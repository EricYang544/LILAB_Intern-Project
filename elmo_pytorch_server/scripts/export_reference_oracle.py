"""Run in a SEPARATE legacy AllenNLP environment on CPU (no project import).

Example: Python 3.8 + allennlp==2.10.1. Provide the same downloaded original
options/weights used by the project. This creates no training results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids


def checksum(path):
    value = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--options", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--output", required=True, help="Path ending in .npz")
args = parser.parse_args()
sentences = [["Alice", "works", "at", "Acme", "."], ["Paris", "."],
             ["The", "bank", "by", "the", "river", "closed", "."],
             ["Café", "東京", "unseenXYZ123", "."]]
model = _ElmoBiLm(args.options, args.weights, requires_grad=False).cpu().eval()
model._elmo_lstm.reset_states()
with torch.no_grad():
    outputs = model(batch_to_ids(sentences))
    values = torch.stack(outputs["activations"], dim=2)
result = {str(i): values[i, 1:len(words) + 1].numpy() for i, words in enumerate(sentences)}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, **result)
path.with_suffix(".json").write_text(json.dumps({"sentences": sentences, "options_sha256": checksum(args.options),
    "weights_sha256": checksum(args.weights), "reference": "AllenNLP _ElmoBiLm; eval; reset states; CPU"}, ensure_ascii=False, indent=2), encoding="utf-8")
print(path)
