"""Print runtime details and test actual CUDA forward/backward, not just availability."""
import argparse
import json
import torch
from elmo_project.common import device_for, environment
from elmo_project.models import BiLM

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
device = device_for({"device": "cpu" if args.cpu else "cuda", "smoke": args.cpu})
print(json.dumps(environment(device), indent=2))
model = BiLM({"hidden": 32, "projection": 8, "char_dim": 4, "filters": [[1, 4], [3, 8]], "highways": 1, "dropout": 0}, 20, 2).to(device)
chars = torch.randint(1, 261, (2, 8, 10), device=device)
lengths = torch.tensor([8, 6], device=device)
targets = torch.randint(0, 20, (2, 8), device=device)
loss, _, _ = model.loss(chars, targets, lengths)
loss.backward()
print(f"Projected biLM forward/backward passed. Loss={loss.item():.4f}")
