"""Execute only a measured, frozen plan; resume completed stages safely."""
import argparse
from pathlib import Path
import time
import signal
import torch
from .common import (read_json, write_json, digest, sha256, load_checkpoint, device_for, Deadline, environment)
from .features import extract_features
from .pilot import runtime_config, code_hash, public_model
from .prepare import verify_prepared
from .training import lm_model, train_lm, train_ner, test_ner


def run(plan_path, extend_hours=0):
    plan = read_json(plan_path)
    identity = plan.pop("plan_id")
    if digest(plan) != identity:
        raise ValueError("Plan was edited after pilot; rerun pilot instead")
    plan["plan_id"] = identity
    if not plan["resolution"]["ready"]:
        raise ValueError("Pilot did not approve the requested budget")
    cfg = runtime_config(plan)
    verify_prepared(cfg["data_dir"])
    if sha256(Path(cfg["data_dir"]) / "manifest.json") != plan["data_sha256"] or code_hash() != plan["code_sha256"]:
        raise ValueError("Code or data changed since pilot; rerun pilot using a fresh run_dir")
    device = device_for(cfg)
    actual_environment = environment(device)
    if any(actual_environment[k] != plan["environment"][k] for k in ("gpu", "torch", "cuda", "packages")):
        raise ValueError("GPU/PyTorch/CUDA differs from pilot; rerun pilot on the target server")
    root = Path(cfg["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    status = read_json(status_path) if status_path.exists() else {"plan_id": identity, "seconds": 0, "complete": False}
    if status["plan_id"] != identity:
        raise ValueError("run_dir already belongs to another plan; select a new run_dir")
    if status["complete"]:
        print("All training and test evaluations already completed.")
        return
    write_json(root / "plan.json", plan)
    total = (cfg["budget"]["hours"] + extend_hours) * 3600
    deadline = Deadline(total - status["seconds"] - plan["pilot_seconds"])
    started = time.monotonic()
    previous_heartbeat = [started - 60]
    def heartbeat():
        now = time.monotonic()
        if now - previous_heartbeat[0] >= 30:
            snapshot = dict(status)
            snapshot["seconds"] = status["seconds"] + now - started
            snapshot["complete"] = False
            write_json(status_path, snapshot)
            previous_heartbeat[0] = now
    deadline.on_check = heartbeat
    heartbeat()
    previous_handler = signal.getsignal(signal.SIGINT)
    def stop_at_boundary(signum, frame):
        deadline.end = 0
        print("Stopping at the next safe checkpoint boundary...", flush=True)
    signal.signal(signal.SIGINT, stop_at_boundary)
    caches = {"static": None}
    try:
        for depth in (1, 2):
            print(f"Training biLM depth {depth}", flush=True)
            checkpoint = train_lm(cfg, depth, plan["resolution"]["lm_steps"], identity, device, deadline)
            model = lm_model(cfg, depth).to(device)
            model.load_state_dict(load_checkpoint(checkpoint, device)["model"])
            cache = root / "features" / f"depth{depth}"
            extract_features(cfg, model, sha256(checkpoint), cache, device, deadline)
            caches[f"depth{depth}"] = cache
            del model
        if "public" in plan["arms"]:
            raw = Path(cfg["raw_dir"])
            for key, filename in (("elmo_options", "elmo_options.json"), ("elmo_weights", "elmo_weights.hdf5")):
                if sha256(raw / filename) != plan["public_assets"][key]["sha256"]:
                    raise ValueError(f"Public asset changed: {key}")
            model = public_model(cfg, device)
            cache = root / "features" / "public"
            extract_features(cfg, model, digest({k: plan["public_assets"][k] for k in ("elmo_options", "elmo_weights")}), cache, device, deadline)
            caches["public"] = cache
            del model
        # Ensure the two arms saw the same training stream and word budget.
        a = read_json(root / "lm_depth1" / "complete.json")
        b = read_json(root / "lm_depth2" / "complete.json")
        if (a["step"], a["tokens"]) != (b["step"], b["tokens"]):
            raise ValueError("Depth ablation token/step budgets diverged")
        for arm in plan["arms"]:
            for seed in cfg["ner"]["seeds"]:
                print(f"Training NER {arm}, seed={seed}", flush=True)
                train_ner(cfg, arm, seed, caches[arm], identity, device, deadline)
        # Only open test labels for scoring after all dev-selected models are fixed.
        for arm in plan["arms"]:
            for seed in cfg["ner"]["seeds"]:
                deadline.check()
                test_ner(cfg, arm, seed, caches[arm], identity, device, deadline)
        status["complete"] = True
    except (TimeoutError, KeyboardInterrupt) as exc:
        status["interrupted"] = type(exc).__name__
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        status["seconds"] += time.monotonic() - started
        status["total_budget_hours"] = cfg["budget"]["hours"] + extend_hours
        write_json(status_path, status)
    print("Completed. Generate results with python -m elmo_project.report --run-dir " + str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--extend-hours", type=float, default=0, help="Explicitly extend total budget beyond the original 24h; recorded in status")
    args = parser.parse_args()
    if args.extend_hours < 0:
        parser.error("--extend-hours must be nonnegative")
    run(args.plan, args.extend_hours)


if __name__ == "__main__":
    main()
