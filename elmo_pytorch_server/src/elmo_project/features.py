"""Fingerprint-checked, disk-backed layer caches shared by all NER seeds."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from .batches import ner_records, feature_batch, ordered_batches
from .common import digest, read_json, write_json, sha256, synchronize


def cache_key(cfg, model_id):
    return digest({"model": model_id, "data": sha256(Path(cfg["data_dir"]) / "manifest.json"),
                   "dtype": cfg["features"]["dtype"], "boundary": "sentence-reset-bos-eos-v1"})


def verify_cache(path, key):
    path = Path(path)
    info = read_json(path / "manifest.json")
    if info["key"] != key:
        raise ValueError(f"Cache fingerprint mismatch: {path}")
    for split, entry in info["splits"].items():
        file = path / f"{split}.npy"
        if sha256(file) != entry["sha256"]:
            raise ValueError(f"Corrupt feature cache: {file}")
    return info


def extract_features(cfg, model, model_id, destination, device, deadline):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    key = cache_key(cfg, model_id)
    if (destination / "manifest.json").exists():
        return verify_cache(destination, key)
    model.eval()
    model.requires_grad_(False)
    start = time.monotonic()
    info = {"key": key, "model_id": model_id, "shape": [model.depth + 1, model.width], "splits": {}}
    max_chars = getattr(model, "max_chars", cfg["model"]["max_chars"])
    for split in ("train", "dev", "test"):
        rows = ner_records(cfg["data_dir"], split)
        lengths = [len(r["tokens"]) for r in rows]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        shape = (int(offsets[-1]), model.depth + 1, model.width)
        target = destination / f"{split}.npy"
        partial = destination / f"{split}.part.npy"
        progress_path = destination / f"{split}.progress.json"
        batches = ordered_batches([n + 2 for n in lengths], cfg["features"]["batch_size"], cfg["features"]["batch_tokens"])
        progress = read_json(progress_path) if progress_path.exists() else {"key": key, "batch": 0}
        if progress["key"] != key:
            raise ValueError("Partial feature cache belongs to another model")
        # A completed split may survive a timeout in a subsequent split.
        if target.exists() and (progress.get("complete") or (progress["batch"] == len(batches) and not partial.exists())):
            info["splits"][split] = {"shape": list(shape), "sha256": sha256(target)}
            continue
        if progress["batch"] > 0 and not partial.exists():
            raise ValueError("Partial cache data missing; use a fresh run directory")
        values = np.lib.format.open_memmap(partial, mode="r+" if partial.exists() else "w+", dtype=cfg["features"]["dtype"], shape=shape)
        for batch_index in range(progress["batch"], len(batches)):
            deadline.check()
            indices = batches[batch_index]
            chars, lens = feature_batch(rows, indices, max_chars, device)
            with torch.inference_mode():
                output = model(chars, lens).float().cpu().numpy()
            for j, i in enumerate(indices):
                values[offsets[i]:offsets[i + 1]] = output[j, 1:lengths[i] + 1]
            values.flush()
            write_json(progress_path, {"key": key, "batch": batch_index + 1})
        del values
        os.replace(partial, target)
        write_json(progress_path, {"key": key, "batch": len(batches), "complete": True})
        info["splits"][split] = {"shape": list(shape), "sha256": sha256(target)}
    synchronize(device)
    info["seconds"] = time.monotonic() - start
    write_json(destination / "manifest.json", info)
    return info
