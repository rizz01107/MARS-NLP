"""Small shared helpers (seeding, hashing, timing, IO)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def get_logger(name: str = "mars") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if installed) torch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional (not needed for baselines)
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch missing
        pass


def stable_seed(*parts: Any) -> int:
    """Deterministic 64-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager measuring wall-clock seconds: ``with timer() as t: ...; t['seconds']``."""
    out = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_fingerprint(df) -> dict:
    """SHA-256 fingerprints of the message-id sets of each split (originals only) - proves the split is unchanged."""
    out = {}
    orig = df[~df["is_perturbed"].astype(bool)]
    for s in ("train", "validation", "test"):
        ids = sorted(orig.loc[orig["split"] == s, "message_id"])
        out[s] = {"n": len(ids), "sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()}
    return out


def environment_info() -> dict:
    """Hardware/software description used to label computational benchmarks."""
    import sklearn

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        info["torch"] = "not installed"
        info["cuda_available"] = False
    try:
        import transformers  # type: ignore

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = "not installed"
    return info
