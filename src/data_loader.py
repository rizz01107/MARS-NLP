"""Loading of raw source files and of the processed MARS-NLP corpus."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as C

_BOOL_COLS = ["is_code_switched", "is_perturbed"]
_STR_COLS = ["message_id", "message", "cleaned_message", "language", "label", "source_dataset", "source_url",
             "source_license", "source_type", "message_type", "perturbation_type", "original_message_id",
             "split", "group_id", "language_confidence"]


def load_raw_uci(path: Path = C.UCI_RAW) -> tuple[pd.DataFrame, dict]:
    """Parse the UCI SMS Spam Collection (``label<TAB>message`` per line).

    Returns (dataframe, info). ``info`` records malformed lines and the decoding used.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UCI file not found: {path}. See data/README.md for download instructions.")
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, encoding = raw.decode("latin-1"), "latin-1"
    rows, malformed = [], 0
    for lineno, line in enumerate(text.split("\n"), start=1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        if "\t" not in line:
            malformed += 1
            continue
        label, msg = line.split("\t", 1)
        rows.append({"source_row": lineno, "label_raw": label.strip(), "message": msg})
    df = pd.DataFrame(rows)
    return df, {"encoding": encoding, "malformed_lines": malformed, "n_rows": len(df)}


def load_raw_zenodo(path: Path = C.ZENODO_RAW) -> tuple[pd.DataFrame, dict]:
    """Read the Zenodo Roman-Urdu phishing SMS workbook (first sheet)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zenodo file not found: {path}. See data/README.md for download instructions.")
    df = pd.read_excel(path, sheet_name=0)
    required = {"ID", "Raw_text", "Labels", "Source_Type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Zenodo workbook is missing columns: {sorted(missing)}")
    out = pd.DataFrame({
        "source_row": df["ID"].astype(int),
        "label_raw": df["Labels"].astype(int).astype(str),
        "message": df["Raw_text"].astype("object"),
        "message_type": df["Source_Type"].astype("object"),
        "normalized_text_source": df.get("Normalized_text", pd.Series([None] * len(df))).astype("object"),
    })
    return out, {"n_rows": len(out), "columns": list(df.columns)}


def load_processed(path: Path = C.PROCESSED_CSV) -> pd.DataFrame:
    """Load the processed corpus with correct dtypes (no accidental NaN conversion of text like 'NA')."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Build it with `python -m src.build_dataset`.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, "true": True, "false": False})
    if "label" in df.columns:
        df["label_id"] = df["label"].map(C.LABEL2ID)
    return df


def get_split(df: pd.DataFrame, split: str, include_perturbed: bool = False,
              languages: Optional[list[str]] = None) -> pd.DataFrame:
    """Select a split; by default only genuine originals (``is_perturbed == False``)."""
    if split not in C.SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {C.SPLITS}")
    mask = df["split"] == split
    if not include_perturbed:
        mask &= ~df["is_perturbed"].astype(bool)
    if languages is not None:
        mask &= df["language"].isin(languages)
    return df.loc[mask].reset_index(drop=True)
