"""Text and tag handling. No ML dependencies, no tokenizer changes."""
from pathlib import Path

TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
SPECIAL = ["<PAD>", "<UNK>", "<S>", "</S>"]


def char_ids(token, width=50):
    """ELMo byte IDs, shifted by one to reserve zero for padded tokens."""
    if width < 3:
        raise ValueError("Character width must be at least 3")
    payload = [256] if token == "<S>" else [257] if token == "</S>" else list(token.encode("utf-8")[:width - 2])
    values = [258] + payload + [259]
    values += [260] * (width - len(values))
    return [v + 1 for v in values]


def to_iob2(tags):
    result = []
    previous_type = None
    for tag in tags:
        if tag == "O":
            result.append(tag)
            previous_type = None
            continue
        if tag not in TAGS:
            raise ValueError(f"Unexpected NER tag: {tag}")
        prefix, kind = tag.split("-", 1)
        if prefix == "I" and previous_type != kind:
            prefix = "B"
        result.append(f"{prefix}-{kind}")
        previous_type = kind
    return result


def read_conll(path):
    records, tokens, tags = [], [], []
    doc = -1

    def flush():
        if tokens:
            records.append({"id": len(records), "doc_id": doc, "tokens": tokens.copy(), "tags": to_iob2(tags)})
            tokens.clear()
            tags.clear()

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            flush()
        elif parts[0] == "-DOCSTART-":
            flush()
            doc += 1
        else:
            if len(parts) < 2:
                raise ValueError(f"Malformed CoNLL row: {line!r}")
            tokens.append(parts[0])
            tags.append(parts[-1])
    flush()
    if not records:
        raise ValueError(f"Empty CoNLL data: {path}")
    return records


def spans(tags):
    """CoNLL-compatible span extraction: an orphan I-X starts an X span."""
    spans_out, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag != "O" and tag not in TAGS:
            raise ValueError(f"Unexpected tag: {tag}")
        prefix, current = ("O", None) if tag == "O" else tag.split("-", 1)
        if kind is not None and (prefix in ("O", "B") or current != kind):
            spans_out.add((start, i, kind))
            start, kind = None, None
        if current is not None and kind is None:
            start, kind = i, current
    return spans_out


def score(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Sentence count mismatch")
    counts = {kind: [0, 0, 0] for kind in ["ALL", "PER", "ORG", "LOC", "MISC"]}
    for expected, actual in zip(gold, predicted):
        if len(expected) != len(actual):
            raise ValueError("Token count mismatch")
        g, p = spans(expected), spans(actual)
        for kind, totals in counts.items():
            gs = g if kind == "ALL" else {x for x in g if x[2] == kind}
            ps = p if kind == "ALL" else {x for x in p if x[2] == kind}
            for i, n in enumerate((len(gs & ps), len(ps), len(gs))):
                totals[i] += n
    result = {}
    for kind, (correct, pred, actual) in counts.items():
        result[kind] = {"precision": 100 * correct / pred if pred else 0.0,
                        "recall": 100 * correct / actual if actual else 0.0,
                        "f1": 200 * correct / (pred + actual) if pred + actual else 0.0,
                        "correct": correct, "predicted": pred, "gold": actual}
    return result


def static_lookup(word, vocab):
    """Exact, lowercase, then digit-normalized SENNA lookup."""
    import re
    for candidate in (word, word.lower(), re.sub(r"\d", "0", word.lower())):
        if candidate in vocab:
            return vocab[candidate]
    return 1
