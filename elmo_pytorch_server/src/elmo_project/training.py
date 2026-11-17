"""Training loops with optimizer/RNG checkpoints and dev-only selection."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from .batches import LMData, NERData, ordered_batches
from .common import (seed_all, rng_state, restore_rng, save_checkpoint, load_checkpoint,
                     log_json, write_json, read_json, synchronize, peak_memory)
from .models import BiLM, NER, mask_from_lengths
from .text import TAGS, score


def amp_context(cfg, device):
    enabled = cfg["lm"]["amp"] and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def lm_model(cfg, depth):
    vocab = read_json(Path(cfg["data_dir"]) / "lm_vocab.json")
    return BiLM(cfg["model"], len(vocab["output"]), depth)


def lm_batches(cfg, data, train=True):
    return ordered_batches(data.lengths, cfg["lm"]["batch_size"], 10**9, cfg["seed"] if train else None)


def lm_update(cfg, model, optimizer, data, batches, micro_index, device):
    optimizer.zero_grad(set_to_none=True)
    count, loss_value = 0, 0.0
    selected = [batches[(micro_index + i) % len(batches)] for i in range(cfg["lm"]["accumulation"])]
    total_predictions = sum(int((data.lengths[indices] - 1).sum()) for indices in selected)
    for indices in selected:
        chars, targets, lengths = data.batch(indices, device)
        with amp_context(cfg, device):
            loss, _, predictions = model.loss(chars, targets, lengths, cfg["lm"]["softmax_chunk"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LM loss")
        (loss * predictions / total_predictions).backward()
        count += int((lengths - 2).sum())  # corpus words, directions counted once
        loss_value += loss.detach().item() * predictions / total_predictions
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["lm"]["clip"], error_if_nonfinite=True)
    optimizer.step()
    return count, loss_value


@torch.no_grad()
def evaluate_lm(cfg, model, data, device, deadline=None):
    model.eval()
    total, counts = [0.0, 0.0], 0
    for indices in lm_batches(cfg, data, False):
        if deadline:
            deadline.check()
        chars, targets, lengths = data.batch(indices, device)
        _, losses, n = model.loss(chars, targets, lengths, cfg["lm"]["softmax_chunk"])
        total[0] += losses[0].item()
        total[1] += losses[1].item()
        counts += n
    nll = [(x / counts) for x in total]
    result = {"forward_nll": nll[0], "backward_nll": nll[1], "mean_nll": sum(nll) / 2,
              "forward_ppl": math.exp(min(nll[0], 700)), "backward_ppl": math.exp(min(nll[1], 700)),
              "predictions_per_direction": counts}
    model.train()
    return result


def train_lm(cfg, depth, steps, fingerprint, device, deadline):
    folder = Path(cfg["run_dir"]) / f"lm_depth{depth}"
    folder.mkdir(parents=True, exist_ok=True)
    completion = folder / "complete.json"
    if completion.exists():
        if read_json(completion)["fingerprint"] != fingerprint:
            raise ValueError("LM completion fingerprint mismatch")
        return folder / "best.pt"
    seed_all(cfg["seed"])
    model = lm_model(cfg, depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lm"]["lr"])
    train, valid = LMData(cfg["data_dir"], "train"), LMData(cfg["data_dir"], "valid")
    batches = lm_batches(cfg, train)
    state = {"step": 0, "tokens": 0, "best": float("inf"), "last_eval_tokens": 0, "seconds": 0.0}
    last = folder / "last.pt"
    if last.exists():
        ckpt = load_checkpoint(last, device)
        if ckpt["fingerprint"] != fingerprint:
            raise ValueError("LM checkpoint belongs to another plan")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        restore_rng(ckpt["rng"])
    start = time.monotonic()
    initial_seconds = state["seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(path):
        state["seconds"] = initial_seconds + time.monotonic() - start
        save_checkpoint(path, {"fingerprint": fingerprint, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "state": state.copy(), "rng": rng_state(), "depth": depth, "model_config": cfg["model"]})

    try:
        while state["step"] < steps and state["tokens"] < cfg["lm"]["max_tokens"]:
            deadline.check()
            model.train()
            tokens, loss = lm_update(cfg, model, optimizer, train, batches, state["step"] * cfg["lm"]["accumulation"], device)
            state["step"] += 1
            state["tokens"] += tokens
            if state["step"] == 1 or state["step"] % 50 == 0:
                synchronize(device)
                log_json(folder / "metrics.jsonl", {"step": state["step"], "tokens": state["tokens"], "train_nll": loss,
                                                     "seconds": initial_seconds + time.monotonic() - start})
            final = state["step"] == steps or state["tokens"] >= cfg["lm"]["max_tokens"]
            if final or state["tokens"] - state["last_eval_tokens"] >= cfg["lm"]["eval_every_tokens"]:
                metrics = evaluate_lm(cfg, model, valid, device, deadline)
                state["last_eval_tokens"] = state["tokens"]
                log_json(folder / "metrics.jsonl", {"step": state["step"], "tokens": state["tokens"], **metrics})
                if metrics["mean_nll"] < state["best"]:
                    state["best"] = metrics["mean_nll"]
                    checkpoint(folder / "best.pt")
            if state["step"] % cfg["lm"]["checkpoint_every_steps"] == 0:
                checkpoint(last)
        # Resume after a timeout during final validation still performs validation.
        if state["last_eval_tokens"] != state["tokens"] or not (folder / "best.pt").exists():
            metrics = evaluate_lm(cfg, model, valid, device, deadline)
            state["last_eval_tokens"] = state["tokens"]
            if metrics["mean_nll"] < state["best"]:
                state["best"] = metrics["mean_nll"]
                checkpoint(folder / "best.pt")
        checkpoint(last)
        write_json(completion, {"fingerprint": fingerprint, **state, "parameters": sum(p.numel() for p in model.parameters()),
                                "peak_vram_bytes": peak_memory(device), "effective_batch_size": cfg["lm"]["batch_size"] * cfg["lm"]["accumulation"],
                                "lr": cfg["lm"]["lr"], "lr_schedule": "constant", "tokens_per_second": state["tokens"] / max(state["seconds"], 1e-6)})
    except (TimeoutError, KeyboardInterrupt):
        checkpoint(last)
        raise
    return folder / "best.pt"


def ner_model(cfg, cache=None):
    vectors = torch.from_numpy(np.load(Path(cfg["data_dir"]) / "senna.npy"))
    shape = read_json(Path(cache) / "manifest.json")["shape"] if cache else None
    return NER(cfg["ner"], vectors, TAGS, shape)


def ner_forward(model, batch):
    return model(batch["words"], batch["chars"], batch["lengths"], batch["features"])


def ner_update(cfg, model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    emissions = ner_forward(model, batch)
    mask = mask_from_lengths(batch["lengths"], batch["words"].shape[1])
    loss = model.loss(emissions, batch["tags"], mask)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite NER loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["ner"]["clip"], error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_ner(cfg, model, data, device, deadline=None):
    model.eval()
    predictions = [None] * len(data)
    for indices in ordered_batches(data.lengths, cfg["ner"]["batch_size"], cfg["ner"]["batch_tokens"]):
        if deadline:
            deadline.check()
        batch = data.batch(indices, device)
        emissions = ner_forward(model, batch)
        decoded = model.crf.decode(emissions, mask_from_lengths(batch["lengths"], batch["words"].shape[1]))
        for i, tags in zip(indices, decoded):
            predictions[i] = [TAGS[t] for t in tags]
    metrics = score([r["tags"] for r in data.records], predictions)
    return metrics, predictions


def train_ner(cfg, arm, seed, cache, fingerprint, device, deadline):
    folder = Path(cfg["run_dir"]) / "ner" / arm / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        if read_json(folder / "complete.json")["fingerprint"] != fingerprint:
            raise ValueError("NER completion fingerprint mismatch")
        return folder
    seed_all(seed)
    model = ner_model(cfg, cache).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["ner"]["lr"])
    train = NERData(cfg["data_dir"], "train", cache, cfg["model"]["max_chars"])
    valid = NERData(cfg["data_dir"], "dev", cache, cfg["model"]["max_chars"])
    state = {"epoch": 0, "batch": 0, "best_f1": -1.0, "bad_epochs": 0, "seconds": 0.0}
    last = folder / "last.pt"
    if last.exists():
        ckpt = load_checkpoint(last, device)
        if ckpt["fingerprint"] != fingerprint:
            raise ValueError("NER checkpoint belongs to another plan")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        restore_rng(ckpt["rng"])
    start, initial_seconds = time.monotonic(), state["seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(path):
        state["seconds"] = initial_seconds + time.monotonic() - start
        save_checkpoint(path, {"fingerprint": fingerprint, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "state": state.copy(), "rng": rng_state()})

    try:
        while state["epoch"] < cfg["ner"]["max_epochs"] and state["bad_epochs"] < cfg["ner"]["patience"]:
            model.train()
            batches = ordered_batches(train.lengths, cfg["ner"]["batch_size"], cfg["ner"]["batch_tokens"], seed + state["epoch"])
            for index in range(state["batch"], len(batches)):
                deadline.check()
                loss = ner_update(cfg, model, optimizer, train.batch(batches[index], device))
                state["batch"] = index + 1
                if state["batch"] % 50 == 0:
                    checkpoint(last)
            metrics, _ = evaluate_ner(cfg, model, valid, device, deadline)
            state["epoch"] += 1
            state["batch"] = 0
            improved = metrics["ALL"]["f1"] > state["best_f1"]
            state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
            if improved:
                state["best_f1"] = metrics["ALL"]["f1"]
                checkpoint(folder / "best.pt")
            log_json(folder / "metrics.jsonl", {"epoch": state["epoch"], "dev": metrics,
                                                "seconds": initial_seconds + time.monotonic() - start})
            checkpoint(last)
        write_json(folder / "complete.json", {"fingerprint": fingerprint, "arm": arm, "seed": seed, **state,
                                               "parameters": sum(p.numel() for p in model.parameters()),
                                               "peak_vram_bytes": peak_memory(device)})
    except (TimeoutError, KeyboardInterrupt):
        checkpoint(last)
        raise
    return folder


def test_ner(cfg, arm, seed, cache, fingerprint, device, deadline):
    folder = Path(cfg["run_dir"]) / "ner" / arm / str(seed)
    if (folder / "test.json").exists():
        saved = read_json(folder / "test.json")
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Test result belongs to another plan")
        return saved
    ckpt = load_checkpoint(folder / "best.pt", device)
    if ckpt["fingerprint"] != fingerprint:
        raise ValueError("Test checkpoint belongs to another plan")
    model = ner_model(cfg, cache).to(device)
    model.load_state_dict(ckpt["model"])
    data = NERData(cfg["data_dir"], "test", cache, cfg["model"]["max_chars"])
    metrics, predicted = evaluate_ner(cfg, model, data, device, deadline)
    from .evaluation import verify_official, write_predictions
    prediction_path = folder / "predictions.conll"
    write_predictions(prediction_path, data.records, predicted)
    if not cfg.get("smoke"):
        verify_official(Path(cfg["raw_dir"]) / "conlleval.pl", prediction_path, metrics["ALL"])
    rows = [{**row, "predicted": pred} for row, pred in zip(data.records, predicted)]
    write_json(folder / "predictions.json", rows)
    result = {"fingerprint": fingerprint, "arm": arm, "seed": seed, "metrics": metrics}
    write_json(folder / "test.json", result)
    return result
