"""Error analysis: experiments/results/error_analysis.csv + reports/tables/error_summary.csv.

error_category = "<FP|FN>|<tag;tag;...>" with tags: english / roman_urdu / code_switched / urdu_script / unknown_language,
short_message (<30 chars), ambiguous (confidence < 0.65), perturbed:<type>.
Usage: python -m src.error_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_processed
from .prediction import BaselinePredictor, available_models
from .utils import ensure_dir, get_logger

log = get_logger()
LANG_TAG = {C.LANG_ENGLISH: "english", C.LANG_ROMAN_URDU: "roman_urdu", C.LANG_MIXED: "code_switched",
            C.LANG_URDU: "urdu_script", C.LANG_UNKNOWN: "unknown_language"}
SHORT_CHARS = 30
AMBIGUOUS_CONF = 0.65


def categorise(row) -> str:
    tags = [LANG_TAG.get(row.language, "unknown_language")]
    if len(row.cleaned_message) < SHORT_CHARS:
        tags.append("short_message")
    if row.confidence < AMBIGUOUS_CONF:
        tags.append("ambiguous")
    if row.is_perturbed:
        tags.append(f"perturbed:{row.perturbation_type}")
    kind = "FP" if row.true_id == 0 else "FN"
    return f"{kind}|{';'.join(tags)}"


def run() -> pd.DataFrame:
    df = load_processed()
    test_all = df[df["split"] == "test"].reset_index(drop=True)
    frames = []
    for name, path in available_models().items():
        if not name.startswith("baseline_"):
            continue
        predictor = BaselinePredictor(path, name)
        ids, p, _ = predictor.scores(test_all["cleaned_message"].tolist())
        d = test_all.assign(pred_id=ids, p_scam=p, true_id=test_all["label_id"].values)
        d["confidence"] = np.where(d.pred_id == 1, d.p_scam, 1 - d.p_scam)
        d = d[d.pred_id != d.true_id].copy()
        d["model"] = predictor.meta.get("name", name)
        frames.append(d)
        log.info("%s: %d errors on test (originals+variants)", name, len(d))
    from .make_results import CANONICAL_TRANSFORMERS, load_transformer_result   # executed (non-smoke) runs only

    for tag, name in CANONICAL_TRANSFORMERS.items():
        if load_transformer_result(tag) is None:
            continue
        tp = pd.read_csv(C.RESULTS_DIR / f"transformer_predictions_{tag}.csv")[["message_id", "y_pred", "p_scam"]]
        d = test_all.merge(tp, on="message_id", how="inner")
        d = d.assign(pred_id=d["y_pred"].astype(int), true_id=d["label_id"].values)
        d["confidence"] = np.where(d.pred_id == 1, d.p_scam, 1 - d.p_scam)
        d = d[d.pred_id != d.true_id].copy()
        d["model"] = name
        frames.append(d)
        log.info("%s: %d errors on test (originals+variants)", tag, len(d))
    err = pd.concat(frames, ignore_index=True)
    err["error_category"] = [categorise(r) for r in err.itertuples()]
    out = pd.DataFrame({
        "message": err["message"], "language": err["language"], "true_label": err["true_id"].map(C.ID2LABEL),
        "predicted_label": err["pred_id"].map(C.ID2LABEL), "confidence": err["confidence"].round(4),
        "error_category": err["error_category"], "model": err["model"]})
    ensure_dir(C.RESULTS_DIR)
    out.to_csv(C.RESULTS_DIR / "error_analysis.csv", index=False)
    summarise(err, test_all).to_csv(C.TABLES_DIR / "error_summary.csv", index=False)
    return out


def summarise(err: pd.DataFrame, test_all: pd.DataFrame) -> pd.DataFrame:
    """Error rates by language / length / perturbation for each model (denominators from the test data)."""
    rows = []
    t = test_all.copy()
    t["short"] = t["cleaned_message"].str.len() < SHORT_CHARS
    for model, e in err.groupby("model"):
        for label, mask_t, mask_e in [
            ("originals", ~t.is_perturbed.astype(bool), ~e.is_perturbed.astype(bool)),
            ("perturbed", t.is_perturbed.astype(bool), e.is_perturbed.astype(bool)),
            ("originals: English", (~t.is_perturbed.astype(bool)) & (t.language == C.LANG_ENGLISH), (~e.is_perturbed.astype(bool)) & (e.language == C.LANG_ENGLISH)),
            ("originals: Roman Urdu + Mixed", (~t.is_perturbed.astype(bool)) & t.language.isin([C.LANG_ROMAN_URDU, C.LANG_MIXED]),
             (~e.is_perturbed.astype(bool)) & e.language.isin([C.LANG_ROMAN_URDU, C.LANG_MIXED])),
            ("originals: short (<30 chars)", (~t.is_perturbed.astype(bool)) & t.short, (~e.is_perturbed.astype(bool)) & (e.cleaned_message.str.len() < SHORT_CHARS)),
        ]:
            n, k = int(mask_t.sum()), int(mask_e.sum())
            fp = int(((e.true_id == 0) & mask_e).sum())
            fn = int(((e.true_id == 1) & mask_e).sum())
            rows.append({"model": model, "slice": label, "n": n, "errors": k, "error_rate": k / n if n else np.nan, "FP": fp, "FN": fn})
        pe = e[e.is_perturbed.astype(bool)]
        for ptype, g in pe.groupby("perturbation_type"):
            n = int(((t.perturbation_type == ptype)).sum())
            rows.append({"model": model, "slice": f"perturbed:{ptype}", "n": n, "errors": len(g), "error_rate": len(g) / n if n else np.nan,
                         "FP": int((g.true_id == 0).sum()), "FN": int((g.true_id == 1).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    o = run()
    print(o.error_category.str.split("|").str[0].value_counts())
