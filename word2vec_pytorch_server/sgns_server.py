"""Self-contained, server-oriented SGNS and CBOW-NS pipeline."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import platform
import random
import shutil
import socket
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


SEMANTIC_SECTIONS = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("run", "data", "model", "training", "checkpoint", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    return config


def write_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(value: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("sgns_server")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def iter_tokens(path: Path, max_tokens: int | None) -> Iterator[str]:
    emitted = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for token in line.lower().split():
                if max_tokens is not None and emitted >= max_tokens:
                    return
                yield token
                emitted += 1


def flush_ids(buffer: list[int], handle: Any, dtype: np.dtype) -> int:
    if not buffer:
        return 0
    array = np.asarray(buffer, dtype=dtype)
    handle.write(array.tobytes())
    count = len(buffer)
    buffer.clear()
    return count


def preprocess(config: dict[str, Any]) -> dict[str, Any]:
    data, run = config["data"], config["run"]
    corpus = Path(data["corpus_path"])
    if not corpus.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus}")
    output = Path(data["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)
    max_tokens = data.get("max_tokens")
    print(f"Pass 1/2: counting tokens in {corpus}", flush=True)
    counts = Counter(iter_tokens(corpus, max_tokens))
    total_raw = sum(counts.values())
    kept = sorted(
        ((word, count) for word, count in counts.items() if count >= data["min_count"]),
        key=lambda item: (-item[1], item[0]),
    )
    if len(kept) < 2:
        raise ValueError("Preprocessing produced fewer than two vocabulary words")
    words = [word for word, _ in kept]
    frequencies = np.asarray([count for _, count in kept], dtype=np.int64)
    word_to_id = {word: index for index, word in enumerate(words)}
    encoded_total = int(frequencies.sum())
    split = int(encoded_total * (1.0 - data.get("validation_ratio", 0.01)))
    dtype = np.dtype("uint32" if len(words) < 2**32 else "uint64")
    sub = data.get("subsampling", {})
    threshold = float(sub.get("threshold", 1e-5))
    relative = frequencies.astype(np.float64) / encoded_total
    keep_prob = np.minimum(1.0, (np.sqrt(relative / threshold) + 1.0) * (threshold / relative))
    enabled = bool(sub.get("enabled", True))
    rng = np.random.default_rng(run["seed"])
    train_path, validation_path = output / "train_tokens.bin", output / "validation_tokens.bin"
    train_buffer: list[int] = []
    validation_buffer: list[int] = []
    encoded_index = train_count = validation_count = original_train = 0
    buffer_size = 1_000_000
    print("Pass 2/2: encoding and streaming token IDs to disk", flush=True)
    with train_path.open("wb") as train_file, validation_path.open("wb") as validation_file:
        for token in iter_tokens(corpus, max_tokens):
            token_id = word_to_id.get(token)
            if token_id is None:
                continue
            if encoded_index < split:
                original_train += 1
                if not enabled or rng.random() < keep_prob[token_id]:
                    train_buffer.append(token_id)
                    if len(train_buffer) >= buffer_size:
                        train_count += flush_ids(train_buffer, train_file, dtype)
            else:
                validation_buffer.append(token_id)
                if len(validation_buffer) >= buffer_size:
                    validation_count += flush_ids(validation_buffer, validation_file, dtype)
            encoded_index += 1
        train_count += flush_ids(train_buffer, train_file, dtype)
        validation_count += flush_ids(validation_buffer, validation_file, dtype)
    vocabulary = {
        "id_to_word": words,
        "frequencies": frequencies.tolist(),
        "total_token_count": total_raw,
    }
    metadata = {
        "corpus_path": str(corpus),
        "dtype": dtype.name,
        "vocabulary_size": len(words),
        "raw_tokens": total_raw,
        "encoded_tokens": encoded_total,
        "train_tokens_before_subsampling": original_train,
        "train_tokens": train_count,
        "validation_tokens": validation_count,
        "validation_ratio": data.get("validation_ratio", 0.01),
        "subsampling": {
            "enabled": enabled,
            "threshold": threshold,
            "retained_tokens": train_count,
            "removed_tokens": original_train - train_count,
            "retention_ratio": train_count / max(original_train, 1),
        },
    }
    write_json(vocabulary, output / "vocabulary.json")
    write_json(metadata, output / "preprocessing.json")
    np.save(output / "frequencies.npy", frequencies)
    np.save(output / "subsampling_keep_probabilities.npy", keep_prob)
    print(json.dumps(metadata, indent=2))
    return metadata


class Vocabulary:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.id_to_word = payload["id_to_word"]
        self.word_to_id = {word: index for index, word in enumerate(self.id_to_word)}
        self.frequencies = np.asarray(payload["frequencies"], dtype=np.int64)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def size(self) -> int:
        return len(self.id_to_word)


def load_tokens(processed: Path, split: str, metadata: dict[str, Any]) -> np.memmap:
    count = int(metadata[f"{split}_tokens"])
    return np.memmap(processed / f"{split}_tokens.bin", mode="r", dtype=metadata["dtype"], shape=(count,))


def hashed_window(index: int, maximum: int, seed: int) -> int:
    value = (index + seed * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    return 1 + value % maximum


class SkipGramPairs(IterableDataset):
    def __init__(
        self,
        tokens: np.ndarray,
        window_size: int,
        dynamic_window: bool,
        seed: int,
        max_pairs: int | None = None,
    ) -> None:
        self.tokens = tokens
        self.window_size = window_size
        self.dynamic_window = dynamic_window
        self.seed = seed
        self.max_pairs = max_pairs

    def __iter__(self) -> Iterator[tuple[int, int]]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        start = len(self.tokens) * worker_id // workers
        stop = len(self.tokens) * (worker_id + 1) // workers
        produced = 0
        for center_index in range(start, stop):
            window = (
                hashed_window(center_index, self.window_size, self.seed)
                if self.dynamic_window
                else self.window_size
            )
            left = max(0, center_index - window)
            right = min(len(self.tokens), center_index + window + 1)
            center = int(self.tokens[center_index])
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                yield center, int(self.tokens[context_index])
                produced += 1
                if self.max_pairs is not None and produced >= self.max_pairs:
                    return


class CBOWExamples(IterableDataset):
    def __init__(
        self,
        tokens: np.ndarray,
        window_size: int,
        dynamic_window: bool,
        seed: int,
        max_examples: int | None = None,
    ) -> None:
        self.tokens = tokens
        self.window_size = window_size
        self.dynamic_window = dynamic_window
        self.seed = seed
        self.max_examples = max_examples

    def __iter__(self) -> Iterator[tuple[list[int], int]]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        start = len(self.tokens) * worker_id // workers
        stop = len(self.tokens) * (worker_id + 1) // workers
        produced = 0
        for target_index in range(start, stop):
            window = (
                hashed_window(target_index, self.window_size, self.seed)
                if self.dynamic_window
                else self.window_size
            )
            left = max(0, target_index - window)
            right = min(len(self.tokens), target_index + window + 1)
            context = [int(self.tokens[index]) for index in range(left, right) if index != target_index]
            if not context:
                continue
            yield context, int(self.tokens[target_index])
            produced += 1
            if self.max_examples is not None and produced >= self.max_examples:
                return


def collate_cbow(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(context) for context, _ in batch)
    contexts = torch.zeros((len(batch), maximum), dtype=torch.long)
    mask = torch.zeros((len(batch), maximum), dtype=torch.bool)
    targets = torch.empty(len(batch), dtype=torch.long)
    for row, (context, target) in enumerate(batch):
        length = len(context)
        contexts[row, :length] = torch.as_tensor(context, dtype=torch.long)
        mask[row, :length] = True
        targets[row] = target
    return contexts, mask, targets


class SGNS(nn.Module):
    def __init__(self, vocabulary_size: int, dimension: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocabulary_size, dimension)
        self.output_embeddings = nn.Embedding(vocabulary_size, dimension)
        nn.init.uniform_(self.input_embeddings.weight, -0.5 / dimension, 0.5 / dimension)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, centers: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        center = self.input_embeddings(centers)
        positive = self.output_embeddings(positives)
        negative = self.output_embeddings(negatives)
        positive_score = (center * positive).sum(dim=-1)
        negative_score = torch.bmm(negative, center.unsqueeze(-1)).squeeze(-1)
        return -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(dim=1)).mean()


class CBOWNS(nn.Module):
    def __init__(self, vocabulary_size: int, dimension: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocabulary_size, dimension)
        self.output_embeddings = nn.Embedding(vocabulary_size, dimension)
        nn.init.uniform_(self.input_embeddings.weight, -0.5 / dimension, 0.5 / dimension)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        contexts: torch.Tensor,
        context_mask: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        vectors = self.input_embeddings(contexts)
        mask = context_mask.to(vectors.dtype).unsqueeze(-1)
        context_vector = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        positive = self.output_embeddings(positives)
        negative = self.output_embeddings(negatives)
        positive_score = (context_vector * positive).sum(dim=-1)
        negative_score = torch.bmm(negative, context_vector.unsqueeze(-1)).squeeze(-1)
        return -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(dim=1)).mean()


def build_model(config: dict[str, Any], vocabulary_size: int) -> nn.Module:
    architecture = config.get("architecture", "skipgram")
    model_class = {"skipgram": SGNS, "cbow": CBOWNS}.get(architecture)
    if model_class is None:
        raise ValueError("model.architecture must be 'skipgram' or 'cbow'")
    return model_class(vocabulary_size, config["embedding_dim"])


class NegativeSampler:
    def __init__(self, frequencies: np.ndarray, seed: int) -> None:
        weights = torch.as_tensor(frequencies, dtype=torch.float64).pow(0.75)
        self.probabilities = (weights / weights.sum()).float()
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, targets: torch.Tensor, count: int) -> torch.Tensor:
        result = torch.multinomial(
            self.probabilities,
            len(targets) * count,
            replacement=True,
            generator=self.generator,
        ).reshape(len(targets), count)
        targets_cpu = targets.detach().cpu().reshape(-1, 1)
        collisions = result.eq(targets_cpu)
        while collisions.any():
            result[collisions] = torch.multinomial(
                self.probabilities,
                int(collisions.sum()),
                replacement=True,
                generator=self.generator,
            )
            collisions = result.eq(targets_cpu)
        return result


def batch_loss(
    model: nn.Module,
    batch: Any,
    sampler: NegativeSampler,
    negative_count: int,
    device: torch.device,
    architecture: str,
) -> tuple[torch.Tensor, int]:
    if architecture == "skipgram":
        centers, targets = batch
        centers = centers.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        negatives = sampler.sample(targets, negative_count).to(device, non_blocking=True)
        return model(centers, targets, negatives), len(targets)
    contexts, context_mask, targets = batch
    contexts = contexts.to(device, non_blocking=True)
    context_mask = context_mask.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    negatives = sampler.sample(targets, negative_count).to(device, non_blocking=True)
    return model(contexts, context_mask, targets, negatives), len(targets)


def autocast_context(device: torch.device, precision: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
    if dtype is None or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    frequencies: np.ndarray,
    negatives: int,
    device: torch.device,
    precision: str,
    seed: int,
    architecture: str,
) -> float:
    model.eval()
    sampler = NegativeSampler(frequencies, seed + 1)
    total = examples = 0
    for batch in loader:
        with autocast_context(device, precision):
            loss, batch_size = batch_loss(model, batch, sampler, negatives, device, architecture)
        total += float(loss) * batch_size
        examples += batch_size
    model.train()
    return total / max(examples, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    step: int,
    metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "metric": metric,
        },
        temporary,
    )
    temporary.replace(path)


def train(config: dict[str, Any], resume: str | None) -> dict[str, Any]:
    run, data, model_config, training = (
        config["run"], config["data"], config["model"], config["training"]
    )
    set_seed(run["seed"], run.get("deterministic", False))
    processed = Path(data["processed_dir"])
    metadata_path = processed / "preprocessing.json"
    if not metadata_path.is_file():
        preprocess(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocabulary = Vocabulary.load(processed / "vocabulary.json")
    train_tokens = load_tokens(processed, "train", metadata)
    validation_tokens = load_tokens(processed, "validation", metadata)
    architecture = model_config.get("architecture", "skipgram")
    if architecture == "skipgram":
        train_dataset = SkipGramPairs(
            train_tokens,
            model_config["window_size"],
            model_config.get("dynamic_window", True),
            run["seed"],
        )
        validation_dataset = SkipGramPairs(
            validation_tokens,
            model_config["window_size"],
            model_config.get("dynamic_window", True),
            run["seed"],
            training.get("validation_max_pairs"),
        )
        collate_fn = None
    elif architecture == "cbow":
        train_dataset = CBOWExamples(
            train_tokens,
            model_config["window_size"],
            model_config.get("dynamic_window", True),
            run["seed"],
        )
        validation_dataset = CBOWExamples(
            validation_tokens,
            model_config["window_size"],
            model_config.get("dynamic_window", True),
            run["seed"],
            training.get("validation_max_examples"),
        )
        collate_fn = collate_cbow
    else:
        raise ValueError("model.architecture must be 'skipgram' or 'cbow'")
    loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        num_workers=training.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training["batch_size"],
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = resolve_device(training.get("device", "auto"))
    model = build_model(model_config, vocabulary.size).to(device)
    optimizer_name = training.get("optimizer", "adamw").lower()
    optimizer_class = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer_name]
    optimizer_kwargs = {"lr": training["learning_rate"]}
    if optimizer_name in {"adam", "adamw"}:
        optimizer_kwargs["weight_decay"] = training.get("weight_decay", 0.0)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    precision = training.get("precision", "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    sampler = NegativeSampler(vocabulary.frequencies, run["seed"])
    run_dir = Path(run.get("output_dir", "runs")) / run["name"]
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluations").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(
        {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "operating_system": platform.platform(),
            "hostname": socket.gethostname(),
        },
        run_dir / "environment.json",
    )
    logger = configure_logging(run_dir / "train.log")
    start_epoch = step = 0
    best = math.inf
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch, step, best = state["epoch"], state["step"], state.get("metric", math.inf)
        historical_best = run_dir / "checkpoints" / "best.pt"
        if historical_best.is_file():
            best_state = torch.load(historical_best, map_location="cpu", weights_only=False)
            best = min(best, float(best_state.get("metric", math.inf)))
        logger.info("Resumed %s at epoch=%d step=%d", resume, start_epoch, step)
    started = time.perf_counter()
    examples_seen = 0
    last_loss = math.nan
    stop = False
    for epoch in range(start_epoch, training["epochs"]):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                loss, batch_size = batch_loss(
                    model,
                    batch,
                    sampler,
                    model_config["negative_samples"],
                    device,
                    architecture,
                )
            scaler.scale(loss).backward()
            if training.get("gradient_clip_norm") is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            step += 1
            examples_seen += batch_size
            last_loss = float(loss.detach())
            if step % training.get("log_every_steps", 100) == 0:
                elapsed = time.perf_counter() - started
                event = {
                    "event": "train",
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": last_loss,
                    "examples_per_second": examples_seen / max(elapsed, 1e-9),
                    "elapsed_seconds": elapsed,
                    "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                }
                append_jsonl(event, run_dir / "metrics.jsonl")
                logger.info(
                    "epoch=%d step=%d loss=%.4f examples/s=%.0f",
                    epoch + 1,
                    step,
                    last_loss,
                    event["examples_per_second"],
                )
            if training.get("max_steps") is not None and step >= training["max_steps"]:
                stop = True
                break
        metric = validation_loss(
            model,
            validation_loader,
            vocabulary.frequencies,
            model_config["negative_samples"],
            device,
            precision,
            run["seed"],
            architecture,
        )
        append_jsonl({"event": "validation", "epoch": epoch + 1, "step": step, "validation_loss": metric}, run_dir / "metrics.jsonl")
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scaler, epoch + 1, step, metric)
        if metric < best:
            best = metric
            shutil.copy2(run_dir / "checkpoints" / "last.pt", run_dir / "checkpoints" / "best.pt")
        logger.info("epoch=%d validation_loss=%.4f best=%.4f", epoch + 1, metric, best)
        if stop:
            break
    elapsed = time.perf_counter() - started
    summary = {
        "run_status": "completed",
        "epochs_completed": epoch + 1,
        "steps": step,
        "final_training_loss": last_loss,
        "best_validation_loss": best,
        "training_examples_processed": examples_seen,
        "training_throughput_examples_per_second": examples_seen / max(elapsed, 1e-9),
        "total_wall_clock_time": elapsed,
        "vocabulary_size": vocabulary.size,
        "architecture": architecture,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
    }
    write_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    return summary


def load_run(run_dir: Path, checkpoint: str) -> tuple[torch.Tensor, Vocabulary, dict[str, Any]]:
    config = load_config(run_dir / "config.yaml")
    processed = Path(config["data"]["processed_dir"])
    vocabulary = Vocabulary.load(processed / "vocabulary.json")
    model = build_model(config["model"], vocabulary.size)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute() and not checkpoint_path.is_file():
        checkpoint_path = run_dir / "checkpoints" / checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.input_embeddings.weight.detach().cpu(), vocabulary, config


def parse_wordsim(path: Path) -> list[tuple[str, str, float]]:
    examples = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().replace(",", "\t").split()
            if len(fields) < 3:
                continue
            try:
                score = float(fields[-1])
            except ValueError:
                if line_number == 1:
                    continue
                raise
            examples.append((fields[0].lower(), fields[1].lower(), score))
    return examples


def evaluate_wordsim(embeddings: torch.Tensor, vocabulary: Vocabulary, path: Path) -> dict[str, Any]:
    predicted, gold = [], []
    examples = parse_wordsim(path)
    for first, second, score in examples:
        if first not in vocabulary.word_to_id or second not in vocabulary.word_to_id:
            continue
        a = embeddings[vocabulary.word_to_id[first]].numpy()
        b = embeddings[vocabulary.word_to_id[second]].numpy()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        predicted.append(float(np.dot(a, b) / denominator) if denominator else 0.0)
        gold.append(score)
    evaluated = len(predicted)
    return {
        "spearman": float(spearmanr(gold, predicted).statistic) if evaluated >= 2 else 0.0,
        "pearson": float(pearsonr(gold, predicted).statistic) if evaluated >= 2 else 0.0,
        "coverage": evaluated / max(len(examples), 1),
        "evaluated_pairs": evaluated,
        "total_pairs": len(examples),
        "oov_pairs": len(examples) - evaluated,
    }


def parse_analogies(path: Path) -> list[tuple[str, str, str, str, str]]:
    examples, section = [], "unknown"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().lower().split()
            if not fields:
                continue
            if fields[0] == ":":
                section = " ".join(fields[1:])
            elif len(fields) == 4:
                examples.append((*fields, section))
    return examples


@torch.no_grad()
def evaluate_analogies(
    embeddings: torch.Tensor,
    vocabulary: Vocabulary,
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    examples = parse_analogies(path)
    limit = min(config.get("max_vocabulary") or vocabulary.size, vocabulary.size)
    valid: list[tuple[list[int], str]] = []
    skipped = 0
    for a, b, c, expected, section in examples:
        words = (a, b, c, expected)
        if any(word not in vocabulary.word_to_id for word in words):
            skipped += 1
            continue
        ids = [vocabulary.word_to_id[word] for word in words]
        if any(index >= limit for index in ids):
            skipped += 1
            continue
        valid.append((ids, "semantic" if section in SEMANTIC_SECTIONS else "syntactic"))
    device = resolve_device(config.get("device", "auto"))
    raw_candidates = embeddings[:limit].to(device)
    normalized_candidates = F.normalize(raw_candidates, dim=1)
    batch_size = config.get("query_batch_size", 256)
    chunk_size = config.get("candidate_chunk_size", 50_000)
    correct = {"semantic": 0, "syntactic": 0}
    totals = {"semantic": 0, "syntactic": 0}
    for batch_start in range(0, len(valid), batch_size):
        batch = valid[batch_start : batch_start + batch_size]
        ids = torch.tensor([item[0] for item in batch], device=device)
        queries = raw_candidates[ids[:, 1]] - raw_candidates[ids[:, 0]] + raw_candidates[ids[:, 2]]
        queries = F.normalize(queries, dim=1)
        best_scores = torch.full((len(batch),), -torch.inf, device=device)
        best_ids = torch.full((len(batch),), -1, dtype=torch.long, device=device)
        for start in range(0, limit, chunk_size):
            stop = min(start + chunk_size, limit)
            scores = queries @ normalized_candidates[start:stop].T
            for row in range(len(batch)):
                for excluded in ids[row, :3].tolist():
                    if start <= excluded < stop:
                        scores[row, excluded - start] = -torch.inf
            values, positions = scores.max(dim=1)
            improved = values > best_scores
            best_scores[improved] = values[improved]
            best_ids[improved] = positions[improved] + start
        for row, (_, category) in enumerate(batch):
            totals[category] += 1
            correct[category] += int(best_ids[row].item() == ids[row, 3].item())
    evaluated = len(valid)
    return {
        "semantic_accuracy": correct["semantic"] / max(totals["semantic"], 1),
        "syntactic_accuracy": correct["syntactic"] / max(totals["syntactic"], 1),
        "overall_accuracy": sum(correct.values()) / max(evaluated, 1),
        "coverage": evaluated / max(len(examples), 1),
        "evaluated_examples": evaluated,
        "semantic_examples": totals["semantic"],
        "syntactic_examples": totals["syntactic"],
        "skipped_examples": skipped,
        "total_examples": len(examples),
        "candidate_vocabulary": limit,
    }


def evaluate(run_dir: Path, checkpoint: str) -> dict[str, Any]:
    embeddings, vocabulary, config = load_run(run_dir, checkpoint)
    evaluation = config["evaluation"]
    results = {
        "wordsim353": evaluate_wordsim(embeddings, vocabulary, Path(evaluation["wordsim353_path"])),
        "google_analogy": evaluate_analogies(
            embeddings, vocabulary, Path(evaluation["google_analogy_path"]), evaluation
        ),
    }
    write_json(results["wordsim353"], run_dir / "evaluations" / "wordsim353.json")
    write_json(results["google_analogy"], run_dir / "evaluations" / "google_analogy.json")
    print(json.dumps(results, indent=2))
    return results


def download_file(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def download_eval(output: Path) -> None:
    wordsim_zip = output / "wordsim353.zip"
    download_file("https://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip", wordsim_zip)
    download_file(
        "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt",
        output / "questions-words.txt",
    )
    wordsim = output / "wordsim353.tsv"
    if not wordsim.is_file():
        with zipfile.ZipFile(wordsim_zip) as archive:
            member = next(name for name in archive.namelist() if name.endswith("combined.csv"))
            with archive.open(member) as source, wordsim.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Server-oriented SGNS and CBOW-NS")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preprocess", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "train":
            command.add_argument("--resume")
    command = subparsers.add_parser("evaluate")
    command.add_argument("--run-dir", required=True)
    command.add_argument("--checkpoint", default="best.pt")
    command = subparsers.add_parser("download-eval")
    command.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    if args.command == "preprocess":
        preprocess(load_config(args.config))
    elif args.command == "train":
        train(load_config(args.config), args.resume)
    elif args.command == "evaluate":
        evaluate(Path(args.run_dir), args.checkpoint)
    else:
        download_eval(Path(args.output_dir))


if __name__ == "__main__":
    main()
