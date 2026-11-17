"""Download public artifacts or use local copies; build immutable prepared data."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import tarfile
import urllib.request
import zipfile

from .common import digest, load_config, read_json, sha256, write_json
from .text import SPECIAL, char_ids, read_conll, static_lookup


def download(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}", flush=True)
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "LILAB-ELMo/0.1"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(4 * 1024 * 1024):
                target.write(chunk)
        if expected and sha256(temporary) != expected:
            raise ValueError(f"Checksum mismatch: {path.name}")
        os.replace(temporary, path)
    checksum = sha256(path)
    if expected and checksum != expected:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return checksum


def extract(archive, destination):
    destination = Path(destination)
    done = destination / ".source.json"
    fingerprint = sha256(archive)
    if done.exists() and read_json(done)["sha256"] == fingerprint:
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    def safe(name):
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"Unsafe archive member: {name}")
        return p

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                safe(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not supported")
            z.extractall(root)
    else:
        with tarfile.open(archive) as t:
            for member in t:
                p = safe(member.name)
                if member.isdir():
                    p.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(member) as source, p.open("wb") as target:
                        while chunk := source.read(4 * 1024 * 1024):
                            target.write(chunk)
                else:
                    raise ValueError("Only ordinary files and directories are accepted")
    write_json(done, {"sha256": fingerprint})


def one(root, pattern):
    matches = sorted(Path(root).rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {root}; found {len(matches)}")
    return matches[0]


def select_lines(paths, token_limit, output, seed, exclude=None):
    """Shuffle official already-shuffled shards; preserve complete sentences."""
    paths = list(sorted(paths))
    random.Random(seed).shuffle(paths)
    total, rows, sources, counts = 0, 0, [], Counter()
    with Path(output).open("w", encoding="utf-8") as out:
        for path in paths:
            last = 0
            with Path(path).open(encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    words = line.split()
                    if not words or (exclude and " ".join(words) in exclude):
                        continue
                    if any(w in SPECIAL for w in words):
                        raise ValueError("Corpus contains reserved special tokens")
                    if total + len(words) > token_limit:
                        continue
                    out.write(" ".join(words) + "\n")
                    counts.update(words)
                    total += len(words)
                    rows += 1
                    last = line_number
                    if total >= token_limit:
                        break
            sources.append({"file": Path(path).name, "through_line": last, "sha256": sha256(path)})
            if total >= token_limit - 100:
                break
    if not total:
        raise ValueError("No LM sentences selected")
    return counts, {"tokens": total, "sentences": rows, "sources": sources, "seed": seed}


def encode_lm(text_path, prefix, input_vocab, output_vocab, max_tokens):
    import numpy as np
    # Binary arrays can be memory-mapped without loading 50M tokens into RAM.
    offsets, lengths, total, word_count = [], [], 0, 0
    with Path(text_path).open(encoding="utf-8") as f, open(str(prefix) + ".input.bin", "wb") as fi, open(str(prefix) + ".target.bin", "wb") as fo:
        for line in f:
            words = line.split()
            for start in range(0, len(words), max_tokens):
                segment = words[start:start + max_tokens]
                tokens = ["<S>"] + segment + ["</S>"]
                ids = np.array([input_vocab[w] for w in tokens], dtype=np.int32)
                target = np.array([output_vocab.get(w, 1) for w in tokens], dtype=np.int32)
                ids.tofile(fi)
                target.tofile(fo)
                offsets.append(total)
                lengths.append(len(tokens))
                total += len(tokens)
                word_count += len(segment)
    np.save(str(prefix) + ".offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(str(prefix) + ".lengths.npy", np.array(lengths, dtype=np.int32))
    return {"segments": len(lengths), "words": word_count, "positions": total}


def prepare(cfg):
    import numpy as np
    root, raw = Path(cfg["data_dir"]), Path(cfg["raw_dir"])
    root.mkdir(parents=True, exist_ok=True)
    request_id = digest({"prepare": cfg["prepare"], "model_max_chars": cfg["model"]["max_chars"],
                         "sources": cfg["sources"], "seed": cfg["seed"], "smoke": cfg.get("smoke", False)})
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["request_id"] != request_id:
            raise ValueError("Prepared data uses different settings; select a new data_dir.")
        verify_prepared(root)
        print(f"Reusing verified data: {root}")
        return manifest
    if cfg.get("smoke"):
        records, train_paths, valid_paths, senna = smoke_sources(raw)
        artifacts = {"synthetic": True}
    else:
        filenames = {"lm": "lm1b.tar.gz", "conll": "conll2003.zip", "senna": "senna.tgz",
                     "elmo_options": "elmo_options.json", "elmo_weights": "elmo_weights.hdf5",
                     "conlleval": "conlleval.pl"}
        artifacts = {}
        for key, filename in filenames.items():
            artifacts[key] = {"url": cfg["sources"][key], "sha256": download(cfg["sources"][key], raw / filename, cfg["sources"].get(key + "_sha256"))}
        for key in ("lm", "conll", "senna"):
            extract(raw / filenames[key], raw / key)
        records = {split: read_conll(one(raw / "conll", name)) for split, name in
                   (("train", "train.txt"), ("dev", "valid.txt"), ("test", "test.txt"))}
        observed = [len(records[s]) for s in ("train", "dev", "test")]
        if observed != cfg["prepare"]["expected_ner_counts"]:
            raise ValueError(f"Unexpected CoNLL sentence counts: {observed}")
        train_paths = sorted((raw / "lm").rglob("training-monolingual.tokenized.shuffled/news.en*"))
        valid_paths = sorted((raw / "lm").rglob("heldout-monolingual.tokenized.shuffled/news.en*"))
        if not train_paths or not valid_paths:
            raise ValueError("Official LM train/heldout shards not found")
        senna = (one(raw / "senna", "words.lst"), one(raw / "senna", "embeddings.txt"))
    for split, rows in records.items():
        with (root / f"ner_{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Exclude exact downstream dev/test sentences from our own LM training.
    excluded = {" ".join(r["tokens"]) for s in ("dev", "test") for r in records[s]}
    counts, selection = select_lines(train_paths, cfg["prepare"]["train_tokens"], root / "lm_train.txt", cfg["seed"], excluded)
    valid_counts, valid_selection = select_lines(valid_paths, cfg["prepare"]["valid_tokens"], root / "lm_valid.txt", cfg["seed"])
    words = SPECIAL + sorted(set(counts) | set(valid_counts))
    inputs = {w: i for i, w in enumerate(words)}
    # Validation-only words receive character input but never enter output vocabulary.
    output_words = SPECIAL + [w for w, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:cfg["prepare"]["vocab_size"]]]
    outputs = {w: i for i, w in enumerate(output_words)}
    write_json(root / "lm_vocab.json", {"input": words, "output": output_words})
    table = np.asarray([char_ids(w, cfg["model"]["max_chars"]) for w in words], dtype=np.int16)
    table[0] = 0
    np.save(root / "lm_chars.npy", table)
    encoded = {s: encode_lm(root / f"lm_{s}.txt", root / f"lm_{s}", inputs, outputs, cfg["prepare"]["max_lm_tokens"]) for s in ("train", "valid")}
    # NER vocabulary: supervised train plus the already-pretrained SENNA lexicon.
    # A dev/test word known to SENNA must not be collapsed to supervised UNK.
    train_types = {w for r in records["train"] for w in r["tokens"]} - set(SPECIAL)
    senna_types = set(Path(senna[0]).read_text(encoding="utf-8").splitlines()) if senna else set()
    ner_words = SPECIAL + sorted((train_types | senna_types) - set(SPECIAL))
    write_json(root / "ner_vocab.json", ner_words)
    matrix = np.random.default_rng(cfg["seed"]).normal(0, 0.05, (len(ner_words), 50)).astype(np.float32)
    lookup = {w: i for i, w in enumerate(ner_words)}
    covered = set()
    if senna:
        with open(senna[0], encoding="utf-8") as wf, open(senna[1], encoding="utf-8") as ef:
            import itertools
            for word, vector in itertools.zip_longest(wf, ef):
                if word is None or vector is None:
                    raise ValueError("SENNA word/vector row mismatch")
                word = word.strip()
                if word in lookup:
                    values = np.fromstring(vector, sep=" ", dtype=np.float32)
                    if values.shape != (50,):
                        raise ValueError("SENNA vectors must have 50 dimensions")
                    matrix[lookup[word]] = values
                    covered.add(word)
    source_lookup = {word: lookup[word] for word in covered}
    for word in train_types - covered:
        index = static_lookup(word, source_lookup)
        if index != 1:
            matrix[lookup[word]] = matrix[index]
            covered.add(word)
    matrix[0] = 0
    np.save(root / "senna.npy", matrix)
    manifest = {"schema": 1, "request_id": request_id, "artifacts": artifacts,
                "selection": selection, "validation_selection": valid_selection, "encoded": encoded,
                "ner_counts": {s: len(rows) for s, rows in records.items()},
                "senna_covered_types": len(covered), "senna_covered_train_types": len(train_types & covered), "ner_train_types": len(train_types), "static_lexicon_types": len(ner_words),
                "preprocessing": "case preserved; LM sentences split into <=max_lm_tokens chunks with fresh boundaries; NER untruncated; exact NER dev/test text excluded from LM train"}
    manifest["files"] = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "manifest.json"}
    write_json(manifest_path, manifest)
    print(f"Prepared {selection['tokens']:,} LM tokens and {len(records['train'])} NER sentences", flush=True)
    return manifest


def verify_prepared(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    for name, checksum in manifest["files"].items():
        if sha256(root / name) != checksum:
            raise ValueError(f"Prepared artifact changed: {name}")
    return manifest


def smoke_sources(raw):
    raw.mkdir(parents=True, exist_ok=True)
    examples = [(["Alice", "works", "at", "Acme", "."], ["B-PER", "O", "O", "B-ORG", "O"]),
                (["Bob", "visits", "Paris", "."], ["B-PER", "O", "B-LOC", "O"]),
                (["Acme", "is", "in", "London", "."], ["B-ORG", "O", "O", "B-LOC", "O"]),
                (["The", "French", "team", "won", "."], ["O", "B-MISC", "O", "O", "O"])]
    records = {s: [{"id": i, "doc_id": i // 2, "tokens": x[0], "tags": x[1]} for i, x in enumerate(examples * n)]
               for s, n in (("train", 4), ("dev", 1), ("test", 1))}
    for s in ("train", "valid"):
        (raw / f"smoke_{s}.txt").write_text(("Alice and Bob work in Paris .\nAcme hired Alice in London .\nThe French team visits Acme .\n" * 100), encoding="utf-8")
    return records, [raw / "smoke_train.txt"], [raw / "smoke_valid.txt"], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
