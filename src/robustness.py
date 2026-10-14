"""Robustness to controlled text perturbations.

Metric (our own definition, not a standard benchmark metric):
    degradation = original_f1 - perturbed_f1
where both are macro-F1 values computed on the SAME underlying test messages: ``original_f1`` on the original
texts for which the perturbation was applicable, ``perturbed_f1`` on their perturbed variants (paired design).
Also reported: prediction-flip rate = share of pairs whose predicted class changes after perturbation, and the
change in scam-class recall. CIs come from a paired cluster bootstrap over near-duplicate groups.

Usage: python -m src.robustness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import get_split, load_processed
from .evaluation import group_confusions, metric_from_counts, reliability_note, subset_status
from .prediction import BaselinePredictor, available_models
from .utils import ensure_dir, get_logger

log = get_logger()
LANG_SLICES = {"All": None, "English": [C.LANG_ENGLISH], "Roman Urdu + Mixed": [C.LANG_ROMAN_URDU, C.LANG_MIXED]}


def paired_degradation(orig: pd.DataFrame, pert: pd.DataFrame, orig_pred: pd.Series, pert_pred: np.ndarray,
                       n_boot: int = 1000, seed: int = C.SEED) -> dict:
    """Paired degradation between matched originals and perturbed variants (rows aligned by ``orig`` ids)."""
    y = orig["label_id"].values
    g = orig["group_id"].values
    ga = group_confusions(y, orig_pred.values, g)
    gb = group_confusions(y, pert_pred, g, index=ga.index)
    A, B = ga.to_numpy(), gb.to_numpy()
    f_o = float(metric_from_counts(A.sum(0), "macro_f1"))
    f_p = float(metric_from_counts(B.sum(0), "macro_f1"))
    rec_o = float(metric_from_counts(A.sum(0), "recall_scam"))
    rec_p = float(metric_from_counts(B.sum(0), "recall_scam"))
    lo = hi = np.nan
    if len(A) >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(A), size=(n_boot, len(A)))
        d = metric_from_counts(A[idx].sum(1), "macro_f1") - metric_from_counts(B[idx].sum(1), "macro_f1")
        lo, hi = np.quantile(d, [0.025, 0.975])
    return {"n_pairs": int(len(orig)), "n_groups": int(len(A)), "original_f1": f_o, "perturbed_f1": f_p,
            "degradation": f_o - f_p, "degradation_ci_low": float(lo), "degradation_ci_high": float(hi),
            "flip_rate": float((orig_pred.values != pert_pred).mean()), "recall_scam_original": rec_o,
            "recall_scam_perturbed": rec_p}


def robustness_rows(orig_df: pd.DataFrame, pert_df: pd.DataFrame, orig_pred: pd.Series, model_id: str,
                    n_boot: int = 1000) -> list[dict]:
    """Robustness table rows for one model.

    ``orig_df``: original messages (needs message_id, label_id, group_id, language);
    ``pert_df``: perturbed variants with a ``pred`` column (needs original_message_id, perturbation_type);
    ``orig_pred``: predictions on the originals indexed by message_id.
    Language slices are defined by the ORIGINAL message's language so code-switched variants stay paired.
    """
    o_idx = orig_df.set_index("message_id")
    pert = pert_df.assign(orig_language=pert_df["original_message_id"].map(o_idx["language"]))
    rows = []
    for slice_name, langs in LANG_SLICES.items():
        for ptype in list(C.PERTURBATION_TYPES) + ["ALL_TYPES (pooled)"]:
            sub = pert if ptype.startswith("ALL_TYPES") else pert[pert["perturbation_type"] == ptype]
            if langs is not None:
                sub = sub[sub["orig_language"].isin(langs)]
            if len(sub) == 0:
                continue
            o = o_idx.loc[sub["original_message_id"]].reset_index()
            ok, note = subset_status(o["label_id"].values)
            base = {"model_id": model_id, "language_slice": slice_name, "perturbation_type": ptype, "status": note}
            if not ok:
                rows.append({**base, "n_pairs": int(len(sub))})
                continue
            o_pred = orig_pred.loc[sub["original_message_id"]].reset_index(drop=True)
            res = paired_degradation(o, sub.reset_index(drop=True), o_pred, sub["pred"].values, n_boot)
            base["status"] = reliability_note(res["n_groups"])
            rows.append({**base, **res})
    return rows


def evaluate_robustness(predictor, test_orig: pd.DataFrame, test_pert: pd.DataFrame, orig_pred: pd.Series,
                        model_id: str, n_boot: int = 1000) -> list[dict]:
    """Predict the perturbed test variants with ``predictor`` and build the robustness rows."""
    ids, _, _ = predictor.scores(test_pert["cleaned_message"].tolist())
    return robustness_rows(test_orig, test_pert.assign(pred=ids), orig_pred, model_id, n_boot)


def run(n_boot: int = 1000) -> pd.DataFrame:
    df = load_processed()
    test_o = get_split(df, "test")
    test_p = df[(df["split"] == "test") & df["is_perturbed"].astype(bool)].reset_index(drop=True)
    base_preds = pd.read_csv(C.RESULTS_DIR / "baseline_predictions.csv")
    rows = []
    for name, path in available_models().items():
        if not name.startswith("baseline_"):
            continue
        mid = name.replace("baseline_", "")
        pred = base_preds[base_preds.model_id == mid].set_index("message_id")["y_pred"]
        if pred.empty:
            log.warning("No stored test predictions for %s - skipping", mid)
            continue
        log.info("Robustness: %s", mid)
        predictor = BaselinePredictor(path, name)
        rows += evaluate_robustness(predictor, test_o, test_p, pred, mid, n_boot)
    out = pd.DataFrame(rows)
    ensure_dir(C.RESULTS_DIR)
    out.to_csv(C.RESULTS_DIR / "robustness_results.csv", index=False)
    return out


def summarise(rob: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean degradation over the 10 perturbation types (language slice 'All')."""
    sub = rob[(rob.language_slice == "All") & (~rob.perturbation_type.str.startswith("ALL_TYPES")) & rob.degradation.notna()]
    g = sub.groupby("model_id")
    return pd.DataFrame({"mean_degradation": g["degradation"].mean(), "max_degradation": g["degradation"].max(),
                         "n_types": g.size()}).reset_index()


if __name__ == "__main__":
    r = run()
    print(summarise(r).to_string(index=False))
