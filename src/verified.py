"""Access to the owner-supplied verified transformer summary (results/verified_transformer_results.json).

Precedence everywhere in the code base: raw run files (experiments/results/transformer_<tag>.json ...) > this summary.
Fields that exist only in raw files (per-language subsets, per-type robustness, raw predictions) are None here.
"""
from __future__ import annotations

import json
from typing import Optional

from . import config as C

VERIFIED_SUPPLEMENT = C.ROOT_DIR / "results" / "verified_transformer_results.json"
SOURCE_LABEL = "owner-verified Colab T4 run (raw files not in this snapshot)"


def load_supplement() -> Optional[dict]:
    p = VERIFIED_SUPPLEMENT
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def supplement_model(tag: str) -> Optional[dict]:
    sup = load_supplement()
    return sup["transformers"].get(tag) if sup else None
