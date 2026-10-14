"""Shared fixtures. Unit tests use tiny synthetic frames; data-dependent tests skip if the corpus is not built."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


def make_synthetic(n_groups: int = 60, variants_per: int = 3) -> pd.DataFrame:
    """Synthetic corpus in the processed schema (messages are invented placeholders, not dataset rows)."""
    rows = []
    for g in range(n_groups):
        label = "Scam" if g % 3 == 0 else "Legitimate"
        src = C.SRC_UCI if g % 2 == 0 else C.SRC_ZENODO
        for k in range(2):   # two near-duplicate originals per group
            mid = f"S-{g:03d}-{k}"
            rows.append(dict(message_id=mid, message=f"win free prize number {g} variant {k}" if label == "Scam" else f"see you at lunch {g} {k}",
                             cleaned_message=f"win free prize number {g} variant {k}" if label == "Scam" else f"see you at lunch {g} {k}",
                             language=C.LANG_ENGLISH, label=label, source_dataset=src, source_url="u", source_license="l",
                             source_type=C.SOURCE_REAL, message_type="t", is_code_switched=False, is_perturbed=False,
                             perturbation_type=C.NO_PERTURBATION, original_message_id=pd.NA, split="train",
                             group_id=f"G{g:03d}", language_confidence="high"))
    df = pd.DataFrame(rows)
    return df


@pytest.fixture(scope="session")
def synthetic_originals() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def processed_df():
    if not C.PROCESSED_CSV.exists():
        pytest.skip("processed dataset not built (python -m src.build_dataset)")
    from src.data_loader import load_processed

    return load_processed()


@pytest.fixture(scope="session")
def predictor():
    from src.prediction import available_models, load_predictor

    if not any(k.startswith("baseline_") for k in available_models()):
        pytest.skip("no trained baseline model (python -m src.train_baselines)")
    return load_predictor()
