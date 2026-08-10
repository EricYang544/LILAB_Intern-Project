from pathlib import Path

import pytest
import numpy as np
import yaml

import sgns_server


@pytest.mark.parametrize("architecture", ["skipgram", "cbow"])
def test_end_to_end(tmp_path: Path, architecture: str) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (("king queen man woman " * 20).strip() + "\n")
        + (("paris france berlin germany " * 20).strip() + "\n")
    )
    wordsim = tmp_path / "wordsim.tsv"
    wordsim.write_text("Word 1,Word 2,Human (mean)\nking,queen,9.0\nman,woman,8.0\n")
    analogies = tmp_path / "questions-words.txt"
    analogies.write_text(": family\nman king woman queen\n")
    config = {
        "run": {"name": "smoke", "output_dir": str(tmp_path / "runs"), "seed": 7, "deterministic": True},
        "data": {
            "corpus_path": str(corpus),
            "processed_dir": str(tmp_path / "processed"),
            "sentence_per_line": True,
            "max_tokens": None,
            "min_count": 1,
            "validation_ratio": 0.1,
            "subsampling": {"enabled": False, "threshold": 1e-5},
        },
        "model": {
            "architecture": architecture,
            "embedding_dim": 8,
            "window_size": 2,
            "dynamic_window": True,
            "negative_samples": 2,
            "evaluation_embedding": "input",
        },
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "precision": "fp32",
            "device": "cpu",
            "num_workers": 0,
            "log_every_steps": 100,
            "validation_max_pairs": 100,
            "validation_max_examples": 100,
            "gradient_clip_norm": None,
            "max_steps": 2,
        },
        "checkpoint": {"save_last": True},
        "evaluation": {
            "wordsim353_path": str(wordsim),
            "google_analogy_path": str(analogies),
            "max_vocabulary": None,
            "query_batch_size": 4,
            "candidate_chunk_size": 4,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded = sgns_server.load_config(config_path)
    metadata = sgns_server.preprocess(loaded)
    assert metadata["train_tokens"] > 0
    assert metadata["sentence_boundary_id"] == metadata["vocabulary_size"]
    assert metadata["train_items"] > metadata["train_tokens"]
    summary = sgns_server.train(loaded, resume=None)
    assert summary["steps"] == 2
    assert summary["architecture"] == architecture
    results = sgns_server.evaluate(tmp_path / "runs" / "smoke", "best.pt")
    assert results["wordsim353"]["evaluated_pairs"] == 2
    assert results["google_analogy"]["evaluated_examples"] == 1


def test_sentence_boundary_blocks_cross_sentence_context() -> None:
    boundary = 99
    tokens = np.asarray([1, 2, boundary, 3, 4], dtype=np.uint32)
    skipgram = sgns_server.SkipGramPairs(
        tokens,
        window_size=5,
        dynamic_window=False,
        seed=1,
        sentence_boundary_id=boundary,
    )
    pairs = set(skipgram)
    assert pairs == {(1, 2), (2, 1), (3, 4), (4, 3)}

    cbow = sgns_server.CBOWExamples(
        tokens,
        window_size=5,
        dynamic_window=False,
        seed=1,
        sentence_boundary_id=boundary,
    )
    examples = list(cbow)
    assert examples == [([2], 1), ([1], 2), ([4], 3), ([3], 4)]
