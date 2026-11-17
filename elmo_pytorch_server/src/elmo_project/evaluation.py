"""Independent evaluator cross-check and prediction serialization."""
from pathlib import Path
import re
import subprocess


def write_predictions(path, records, predictions):
    with Path(path).open("w", encoding="utf-8") as f:
        for row, predicted in zip(records, predictions):
            for token, gold, pred in zip(row["tokens"], row["tags"], predicted):
                f.write(f"{token} {gold} {pred}\n")
            f.write("\n")


def verify_official(script, prediction_path, expected):
    with Path(prediction_path).open("rb") as source:
        result = subprocess.run(["perl", str(script)], stdin=source, capture_output=True, check=True, timeout=60)
    text = result.stdout.decode()
    match = re.search(r"precision:\s*([\d.]+)%;\s*recall:\s*([\d.]+)%;\s*FB1:\s*([\d.]+)", text)
    if not match:
        raise ValueError(f"Unexpected conlleval output: {text}")
    for key, value in zip(("precision", "recall", "f1"), match.groups()):
        if abs(float(value) - expected[key]) > 0.011:
            raise ValueError(f"Evaluator disagreement: {key}={value}, internal={expected[key]}")
    return text
