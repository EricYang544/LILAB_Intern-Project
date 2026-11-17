"""Compare the HDF5 adapter with outputs from a separate official implementation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from elmo_project.common import read_json, sha256
from elmo_project.pretrained import PublicELMo
from elmo_project.batches import feature_batch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--options", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--oracle", required=True)
args = parser.parse_args()
metadata = read_json(Path(args.oracle).with_suffix(".json"))
if sha256(args.options) != metadata["options_sha256"] or sha256(args.weights) != metadata["weights_sha256"]:
    raise ValueError("Oracle used different public model files")
model = PublicELMo(args.options, args.weights).cpu().eval()
rows = [{"tokens": words} for words in metadata["sentences"]]
chars, lengths = feature_batch(rows, list(range(len(rows))), model.max_chars, torch.device("cpu"))
with torch.inference_mode():
    actual = model(chars, lengths).numpy()
with np.load(args.oracle) as oracle:
    for i, row in enumerate(rows):
        result = actual[i, 1:len(row["tokens"]) + 1]
        np.testing.assert_allclose(result, oracle[str(i)], rtol=1e-4, atol=1e-5)
print("Public HDF5 adapter agrees with the official reference oracle.")
