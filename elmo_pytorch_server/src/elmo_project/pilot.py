"""Measure the actual server; freeze one fair, budgeted experiment plan."""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
import torch
from .batches import LMData, NERData, ordered_batches, ner_records, feature_batch
from .common import (load_config, write_json, read_json, sha256, digest, seed_all, device_for,
                     synchronize, environment, Deadline, peak_memory)
from .budget import resolve_budget
from .models import NER
from .pretrained import PublicELMo
from .prepare import verify_prepared
from .training import lm_model, lm_batches, lm_update, ner_update, ner_forward
from .text import TAGS


def code_hash():
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def portable_config(cfg):
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["project_root"])
    for key in ("data_dir", "raw_dir", "run_dir", "pilot_dir"):
        cfg[key] = str(Path(cfg[key]).relative_to(root))
    cfg.pop("project_root")
    return cfg


def runtime_config(plan):
    cfg = copy.deepcopy(plan["config"])
    root = Path(__file__).resolve().parents[2]
    cfg["project_root"] = str(root)
    for key in ("data_dir", "raw_dir", "run_dir", "pilot_dir"):
        cfg[key] = str(root / cfg[key])
    return cfg


def timed_indices(batches, count):
    return [batches[i] for i in np.linspace(0, len(batches) - 1, count, dtype=int)]


def public_model(cfg, device):
    raw = Path(cfg["raw_dir"])
    return PublicELMo(raw / "elmo_options.json", raw / "elmo_weights.hdf5").to(device)


def measure_lm(cfg, depth, device, deadline):
    seed_all(cfg["seed"])
    model = lm_model(cfg, depth).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lm"]["lr"])
    data = LMData(cfg["data_dir"], "train")
    batches = lm_batches(cfg, data)
    samples = timed_indices(batches, cfg["budget"]["pilot_steps"])
    # Warmup is excluded; sampled real batches include both short and long inputs.
    for i in range(cfg["budget"]["pilot_warmup"]):
        deadline.check()
        lm_update(cfg, model, optimizer, data, samples, i * cfg["lm"]["accumulation"], device)
    synchronize(device)
    seconds, tokens = [], []
    for i in range(cfg["budget"]["pilot_steps"]):
        deadline.check()
        start = time.monotonic()
        n, _ = lm_update(cfg, model, optimizer, data, samples, i * cfg["lm"]["accumulation"], device)
        synchronize(device)
        seconds.append(time.monotonic() - start)
        tokens.append(n)
    # Estimate exact full-softmax validation separately.
    valid = LMData(cfg["data_dir"], "valid")
    vb = lm_batches(cfg, valid, False)
    model.eval()
    validation_seconds = []
    with torch.no_grad():
        for indices in timed_indices(vb, min(len(vb), cfg["budget"]["pilot_steps"])):
            deadline.check()
            start = time.monotonic()
            chars, targets, lengths = valid.batch(indices, device)
            model.loss(chars, targets, lengths, cfg["lm"]["softmax_chunk"])
            synchronize(device)
            validation_seconds.append(time.monotonic() - start)
    return {"step_seconds": float(np.mean(seconds)), "words_per_step": float(np.mean(tokens)),
            "validation_seconds": float(np.mean(validation_seconds) * len(vb)),
            "parameters": sum(p.numel() for p in model.parameters())}


def measure_features(cfg, model, device, deadline):
    model.eval()
    rows = sum([ner_records(cfg["data_dir"], split) for split in ("train", "dev", "test")], [])
    batches = ordered_batches([len(r["tokens"]) + 2 for r in rows], cfg["features"]["batch_size"], cfg["features"]["batch_tokens"])
    samples = timed_indices(batches, min(len(batches), cfg["budget"]["pilot_steps"]))
    elapsed = []
    with torch.inference_mode():
        for indices in samples:
            deadline.check()
            start = time.monotonic()
            chars, lengths = feature_batch(rows, indices, getattr(model, "max_chars", cfg["model"]["max_chars"]), device)
            model(chars, lengths).float().cpu().numpy()
            synchronize(device)
            elapsed.append(time.monotonic() - start)
    return float(np.mean(elapsed) * len(batches))


def measure_ner(cfg, feature_shape, device, deadline):
    vectors = torch.from_numpy(np.load(Path(cfg["data_dir"]) / "senna.npy"))
    model = NER(cfg["ner"], vectors, TAGS, feature_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["ner"]["lr"])
    costs = {}
    for split in ("train", "dev", "test"):
        data = NERData(cfg["data_dir"], split, max_chars=cfg["model"]["max_chars"])
        batches = ordered_batches(data.lengths, cfg["ner"]["batch_size"], cfg["ner"]["batch_tokens"])
        samples = timed_indices(batches, min(len(batches), cfg["budget"]["pilot_steps"]))
        seconds = []
        model.train(split == "train")
        for indices in samples:
            deadline.check()
            start = time.monotonic()
            batch = data.batch(indices, device)
            if feature_shape:
                batch["features"] = torch.zeros(*batch["words"].shape, *feature_shape, device=device)
            if split == "train":
                ner_update(cfg, model, optimizer, batch)
            else:
                from .models import mask_from_lengths
                with torch.no_grad():
                    emissions = ner_forward(model, batch)
                    model.crf.decode(emissions, mask_from_lengths(batch["lengths"], batch["words"].shape[1]))
            synchronize(device)
            seconds.append(time.monotonic() - start)
        costs[split] = float(np.mean(seconds) * len(batches))
    return costs



def pilot(cfg):
    manifest = verify_prepared(cfg["data_dir"])
    device = device_for(cfg)
    destination = Path(cfg["pilot_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    if not cfg.get("smoke"):
        from .evaluation import write_predictions, verify_official
        from .text import score
        sample = [{"tokens": ["Alice", "Smith", "in", "Paris"], "tags": ["B-PER", "I-PER", "O", "B-LOC"]}]
        predicted = [["B-PER", "O", "O", "B-LOC"]]
        check_file = destination / "evaluator_check.conll"
        write_predictions(check_file, sample, predicted)
        verify_official(Path(cfg["raw_dir"]) / "conlleval.pl", check_file, score([sample[0]["tags"]], predicted)["ALL"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    deadline = Deadline(cfg["budget"]["pilot_hours"] * 3600)
    measured = {"lm": {}, "features": {}, "ner": {}}
    def progress():
        write_json(destination / "measurements.json", {"environment": environment(device), "measurements": measured})
    for depth in (1, 2):
        print(f"Measuring LM depth {depth}", flush=True)
        measured["lm"][str(depth)] = measure_lm(cfg, depth, device, deadline)
        model = lm_model(cfg, depth).to(device)
        measured["features"][f"depth{depth}"] = measure_features(cfg, model, device, deadline)
        del model
        progress()
    arms = {"static": None, "depth1": [2, cfg["model"]["projection"] * 2], "depth2": [3, cfg["model"]["projection"] * 2]}
    if cfg.get("include_public_elmo", False) and not cfg.get("smoke"):
        print("Measuring original public ELMo", flush=True)
        model = public_model(cfg, device)
        measured["features"]["public"] = measure_features(cfg, model, device, deadline)
        arms["public"] = [model.depth + 1, model.width]
        del model
    for name, shape in arms.items():
        print(f"Measuring NER {name}", flush=True)
        measured["ner"][name] = measure_ner(cfg, shape, device, deadline)
        progress()
    resolution = resolve_budget(cfg, measured)
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["ner"]["max_epochs"] = resolution["ner_max_epochs"]
    plan = {"schema": 1, "config": portable_config(resolved_cfg), "arms": list(arms), "resolution": resolution,
            "measurements": measured, "environment": environment(device), "data_sha256": sha256(Path(cfg["data_dir"]) / "manifest.json"),
            "code_sha256": code_hash(), "pilot_seconds": time.monotonic() - started,
            "peak_vram_bytes": peak_memory(device),
            "public_assets": manifest["artifacts"]}
    plan["plan_id"] = digest(plan)
    write_json(destination / "resolved_plan.json", plan)
    print(f"Ready: {resolution['ready']}; estimated {resolution['estimated_seconds'] / 3600:.2f} hours", flush=True)
    if not resolution["ready"]:
        raise RuntimeError("Pilot rejected this budget: " + "; ".join(resolution["reasons"]))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pilot(load_config(args.config))


if __name__ == "__main__":
    main()
