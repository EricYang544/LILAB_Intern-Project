"""Configuration, atomic artifacts and reproducibility helpers."""
import hashlib
import json
import os
import random
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / cfg.pop("project_root", "..")).resolve()
    for key in ("data_dir", "raw_dir", "run_dir", "pilot_dir"):
        cfg[key] = str((root / cfg[key]).resolve())
    cfg["project_root"] = str(root)
    if cfg["model"]["projection"] >= cfg["model"]["hidden"]:
        raise ValueError("projection must be smaller than hidden")
    if cfg["budget"]["hours"] <= 0 or cfg["prepare"]["train_tokens"] <= 0:
        raise ValueError("Budgets must be positive")
    return cfg


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    import torch
    # Only load locally generated checkpoints; these contain RNG states.
    return torch.load(path, map_location=device, weights_only=False)


def device_for(cfg):
    import torch
    torch.set_num_threads(cfg.get("threads", 4))
    name = cfg["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if not cfg.get("smoke") and device.type != "cuda":
        raise RuntimeError("Production pilot/run requires CUDA. Use the smoke config for CPU checks.")
    if device.type == "cuda":
        x = torch.randn(32, 32, device=device, requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
    return device


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    import platform
    from importlib.metadata import version
    import torch
    return {"packages": {name: version(name) for name in ("torch", "numpy", "PyYAML", "h5py", "matplotlib")}, "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def peak_memory(device):
    import torch
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0


class Deadline:
    def __init__(self, seconds, on_check=None):
        self.end = time.monotonic() + max(0, seconds)
        self.on_check = on_check

    def check(self):
        if self.on_check is not None:
            self.on_check()
        if time.monotonic() >= self.end:
            raise TimeoutError("Run budget exhausted; checkpoint saved, resume with the same command.")
