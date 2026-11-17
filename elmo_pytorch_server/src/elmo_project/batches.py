"""Memory-mapped corpus and bounded, deterministic minibatches."""
import json
from pathlib import Path
import numpy as np
import torch
from .text import TAGS, char_ids, static_lookup
from .common import read_json


class LMData:
    def __init__(self, root, split):
        root = Path(root)
        prefix = root / f"lm_{split}"
        self.inputs = np.memmap(str(prefix) + ".input.bin", mode="r", dtype=np.int32)
        self.targets = np.memmap(str(prefix) + ".target.bin", mode="r", dtype=np.int32)
        self.offsets = np.load(str(prefix) + ".offsets.npy", mmap_mode="r")
        self.lengths = np.load(str(prefix) + ".lengths.npy", mmap_mode="r")
        self.chars = np.load(root / "lm_chars.npy", mmap_mode="r")

    def __len__(self):
        return len(self.offsets)

    def batch(self, indices, device):
        lengths = self.lengths[indices]
        width = int(max(lengths))
        inputs = np.zeros((len(indices), width), dtype=np.int64)
        targets = np.zeros_like(inputs)
        for row, index in enumerate(indices):
            offset, length = int(self.offsets[index]), int(self.lengths[index])
            inputs[row, :length] = self.inputs[offset:offset + length]
            targets[row, :length] = self.targets[offset:offset + length]
        chars = torch.as_tensor(self.chars[inputs].astype(np.int64), device=device)
        return chars, torch.as_tensor(targets, device=device), torch.tensor(lengths.astype(np.int64), device=device)


def ordered_batches(lengths, batch_size, token_limit, seed=None):
    indices = list(range(len(lengths)))
    if seed is not None:
        np.random.default_rng(seed).shuffle(indices)
    # Local bucketing saves padding without making the epoch fully length-sorted.
    result = []
    for offset in range(0, len(indices), batch_size * 20):
        bucket = sorted(indices[offset:offset + batch_size * 20], key=lambda i: lengths[i])
        batch, max_length = [], 0
        for i in bucket:
            next_max = max(max_length, int(lengths[i]))
            if batch and (len(batch) >= batch_size or (len(batch) + 1) * next_max > token_limit):
                result.append(batch)
                batch, max_length = [], 0
            batch.append(i)
            max_length = max(max_length, int(lengths[i]))
        if batch:
            result.append(batch)
    if seed is not None:
        np.random.default_rng(seed + 1).shuffle(result)
    return result


def ner_records(root, split):
    with (Path(root) / f"ner_{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class NERData:
    def __init__(self, root, split, cache=None, max_chars=50):
        self.records = ner_records(root, split)
        self.lengths = [len(r["tokens"]) for r in self.records]
        vocab = {w: i for i, w in enumerate(read_json(Path(root) / "ner_vocab.json"))}
        tag_ids = {tag: i for i, tag in enumerate(TAGS)}
        self.words = [np.asarray([static_lookup(w, vocab) for w in r["tokens"]], dtype=np.int64) for r in self.records]
        self.tags = [np.asarray([tag_ids[t] for t in r["tags"]], dtype=np.int64) for r in self.records]
        self.chars = [np.asarray([char_ids(w, max_chars) for w in r["tokens"]], dtype=np.int64) for r in self.records]
        self.features = None
        if cache:
            self.features = np.load(Path(cache) / f"{split}.npy", mmap_mode="r")
            self.feature_offsets = np.concatenate([[0], np.cumsum(self.lengths)])
            if len(self.features) != sum(self.lengths):
                raise ValueError("Cache token count mismatch")

    def __len__(self):
        return len(self.records)

    def batch(self, indices, device):
        lengths = [self.lengths[i] for i in indices]
        shape = (len(indices), max(lengths))
        words = np.zeros(shape, dtype=np.int64)
        tags = np.zeros(shape, dtype=np.int64)
        chars = np.zeros((*shape, self.chars[0].shape[-1]), dtype=np.int64)
        features = np.zeros((*shape, *self.features.shape[1:]), dtype=np.float32) if self.features is not None else None
        for row, i in enumerate(indices):
            n = lengths[row]
            words[row, :n], tags[row, :n], chars[row, :n] = self.words[i], self.tags[i], self.chars[i]
            if features is not None:
                features[row, :n] = self.features[self.feature_offsets[i]:self.feature_offsets[i + 1]]
        return {"words": torch.as_tensor(words, device=device), "chars": torch.as_tensor(chars, device=device),
                "tags": torch.as_tensor(tags, device=device), "lengths": torch.tensor(lengths, device=device),
                "features": torch.as_tensor(features, device=device) if features is not None else None}


def feature_batch(records, indices, max_chars, device):
    sentences = [["<S>"] + records[i]["tokens"] + ["</S>"] for i in indices]
    lengths = [len(s) for s in sentences]
    chars = np.zeros((len(sentences), max(lengths), max_chars), dtype=np.int64)
    for row, sentence in enumerate(sentences):
        chars[row, :len(sentence)] = [char_ids(w, max_chars) for w in sentence]
    return torch.as_tensor(chars, device=device), torch.tensor(lengths, device=device)
