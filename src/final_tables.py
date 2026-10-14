"""Curated final tables (results/*.csv): built ONLY from saved result files and the owner-verified transformer summary.

Precedence for transformers: raw run files (experiments/results/transformer_<tag>.json ...) > results/verified_transformer_results.json.
Baseline numbers are read from experiments/results/baseline_results.csv (and checked against the owner-supplied table by the QA suite).
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .make_results import load_transformer_result, transformer_robustness_mean
from .utils import load_json
from .verified import SOURCE_LABEL, load_supplement, supplement_model

R = C.RESULTS_DIR
FINAL_DIR = C.ROOT_DIR / "results"
BASELINE_ORDER = [("char_lr", "Char TF-IDF + Logistic Regression"), ("char_svm", "Char TF-IDF + Linear SVM"),
                  ("word_lr", "Word TF-IDF + Logistic Regression"), ("word_svm", "Word TF-IDF + Linear SVM"),
                  ("word_nb", "Word TF-IDF + Naive Bayes"), ("char_nb", "Char TF-IDF + Naive Bayes")]
TX_ORDER = [("mbert", "mBERT"), ("xlmr", "XLM-R")]
BASE_SRC = "experiments/results/baseline_results.csv (recomputed from baseline_predictions.csv by QA)"


def tx_values(tag: str) -> dict | None:
    """Unified transformer values (raw run JSON preferred, else owner-verified summary)."""
    r = load_transformer_result(tag)
    if r is not None:
        o = r["test_overall"]
        rob = transformer_robustness_mean(tag)
        lat = r["latency_computational_benchmark"]
        return {"accuracy": o["accuracy"], "precision": o["precision_scam"], "recall": o["recall_scam"], "macro_f1": o["macro_f1"],
                "weighted_f1": o["weighted_f1"], "ci": [o["macro_f1_ci_low"], o["macro_f1_ci_high"]], "robustness": rob,
                "gpu": r["environment"].get("gpu_name") or r["device"], "train_seconds": r["train_seconds"], "best_epoch": r["best_epoch"],
                "params": r["parameters"]["total"], "size_mb": r["model_size_mb"], "gpu_ms": lat.get("cuda", lat.get(r["device"], {})).get("latency_single_ms_mean"),
                "cpu_ms": lat.get("cpu", {}).get("latency_single_ms_mean"), "n_train": r["n_train"],
                "source": f"experiments/results/transformer_{tag}.json (raw run file)", "raw": True}
    sm = supplement_model(tag)
    if sm is None:
        return None
    return {"accuracy": sm["accuracy"], "precision": sm["precision"], "recall": sm["recall"], "macro_f1": sm["macro_f1"], "weighted_f1": sm["weighted_f1"],
            "ci": sm["macro_f1_ci95"], "robustness": sm["robustness_degradation"], "gpu": sm["gpu"], "train_seconds": sm["train_seconds"],
            "best_epoch": sm["best_epoch"], "params": sm["parameters"], "size_mb": sm["model_size_mb"], "gpu_ms": sm["gpu_latency_ms_per_message"],
            "cpu_ms": sm["cpu_latency_ms_per_message"], "n_train": sm["train_samples"],
            "source": f"results/verified_transformer_results.json ({SOURCE_LABEL})", "raw": False}


def best_observed_baseline(base: pd.DataFrame) -> str:
    ov = base[(base.subset == "Overall") & (~base.aug_trained)]
    return ov.sort_values("macro_f1", ascending=False).iloc[0].model_id


def final_results_df() -> pd.DataFrame:
    base = pd.read_csv(R / "baseline_results.csv")
    ov = base[(base.subset == "Overall") & (~base.aug_trained)].set_index("model_id")
    selected = load_json(C.MODELS_DIR / "best_model.json")["model_id"]
    best_obs = best_observed_baseline(base)
    rows = []
    for mid, name in BASELINE_ORDER:
        r = ov.loc[mid]
        role = []
        if mid == best_obs:
            role.append("highest observed test macro-F1 among the 8 models")
        if mid == selected:
            role.append("validation-selected CPU baseline")
        rows.append({"Model": name, "Family": "CPU baseline (TF-IDF)", "Accuracy": round(r.accuracy, 4), "Precision": round(r.precision_scam, 4),
                     "Recall": round(r.recall_scam, 4), "Macro_F1": round(r.macro_f1, 4), "Weighted_F1": round(r.weighted_f1, 4),
                     "Macro_F1_CI95": f"[{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]", "Role": "; ".join(role),
                     "Hardware": "CPU (1 core)", "Source": BASE_SRC})
    for tag, name in TX_ORDER:
        v = tx_values(tag)
        if v is None:
            rows.append({"Model": name, "Family": "Transformer", "Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", "Macro_F1": "N/A",
                         "Weighted_F1": "N/A", "Macro_F1_CI95": "N/A", "Role": "", "Hardware": "N/A", "Source": "no result available"})
            continue
        rows.append({"Model": name, "Family": "Transformer", "Accuracy": round(v["accuracy"], 4), "Precision": round(v["precision"], 4),
                     "Recall": round(v["recall"], 4), "Macro_F1": round(v["macro_f1"], 4), "Weighted_F1": round(v["weighted_f1"], 4),
                     "Macro_F1_CI95": f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]",
                     "Role": "best-performing transformer" if tag == "xlmr" else "", "Hardware": v["gpu"], "Source": v["source"]})
    return pd.DataFrame(rows)


def robustness_df() -> pd.DataFrame:
    rob = pd.read_csv(R / "robustness_results.csv")
    allr = rob[(rob.language_slice == "All") & (~rob.perturbation_type.str.startswith("ALL_TYPES"))]
    names = dict(BASELINE_ORDER)
    rows = []
    for mid, name in BASELINE_ORDER:
        g = allr[allr.model_id == mid]
        rows.append({"Model": name, "Level": "mean over 10 perturbation types", "Perturbation_Type": "ALL (mean)", "Degradation": round(float(g.degradation.mean()), 4),
                     "CI95_low": "", "CI95_high": "", "n_pairs": "", "n_groups": "", "Source": "experiments/results/robustness_results.csv (mean of per-type values)"})
    for tag, name in TX_ORDER:
        v = tx_values(tag)
        if v is not None and v["robustness"] is not None:
            rows.append({"Model": name, "Level": "mean over 10 perturbation types", "Perturbation_Type": "ALL (mean)", "Degradation": round(float(v["robustness"]), 4),
                         "CI95_low": "", "CI95_high": "", "n_pairs": "", "n_groups": "", "Source": v["source"]})
    for mid, name in BASELINE_ORDER:
        for _, r in allr[allr.model_id == mid].iterrows():
            rows.append({"Model": name, "Level": "per perturbation type (paired, test split)", "Perturbation_Type": r.perturbation_type,
                         "Degradation": round(r.degradation, 4), "CI95_low": round(r.degradation_ci_low, 4), "CI95_high": round(r.degradation_ci_high, 4),
                         "n_pairs": int(r.n_pairs), "n_groups": int(r.n_groups), "Source": "experiments/results/robustness_results.csv"})
    for tag, name in TX_ORDER:
        f = R / f"transformer_robustness_{tag}.csv"
        if load_transformer_result(tag) is not None and f.exists():
            t = pd.read_csv(f)
            t = t[(t.language_slice == "All") & (~t.perturbation_type.str.startswith("ALL_TYPES"))]
            for _, r in t.iterrows():
                rows.append({"Model": name, "Level": "per perturbation type (paired, test split)", "Perturbation_Type": r.perturbation_type,
                             "Degradation": round(r.degradation, 4), "CI95_low": round(r.degradation_ci_low, 4), "CI95_high": round(r.degradation_ci_high, 4),
                             "n_pairs": int(r.n_pairs), "n_groups": int(r.n_groups), "Source": f"experiments/results/transformer_robustness_{tag}.csv"})
    del names
    return pd.DataFrame(rows)


def statistical_df() -> pd.DataFrame:
    sup = load_supplement()
    rows = []
    for c in (sup["statistical_comparisons"] if sup else []):
        rows.append({"Comparison": c["comparison"], "Metric": c["metric"], "A": c["A"], "B": c["B"], "Difference_A_minus_B": c["difference"],
                     "CI95_low": c["ci95_low"], "CI95_high": c["ci95_high"], "CI_includes_zero": c["ci_includes_zero"],
                     "Interpretation": c["interpretation"], "Method": "paired cluster bootstrap over near-duplicate test groups (macro-F1)",
                     "Source": "results/verified_transformer_results.json (owner-verified; raw predictions not in this snapshot)"})
    return pd.DataFrame(rows)


def error_counts_df() -> pd.DataFrame:
    e = pd.read_csv(R / "error_analysis.csv")
    names = pd.read_csv(R / "baseline_results.csv").query("subset == 'Overall'")[["model_id", "model"]].drop_duplicates()
    kind = e.error_category.str.split("|").str[0]
    sup = load_supplement()
    rows = []
    for mid, mname in zip(names.model_id, names.model):
        m = e.model == mname
        rows.append({"Model_ID": mid, "Model": mname, "Errors": int(m.sum()), "False_Positives": int((m & (kind == "FP")).sum()), "False_Negatives": int((m & (kind == "FN")).sum()),
                     "Source": "experiments/results/error_analysis.csv"})
    for tag, name in TX_ORDER:
        n = sup["transformers"][tag]["error_count_test_orig_plus_variants"] if sup else None
        rows.append({"Model_ID": tag, "Model": name, "Errors": n, "False_Positives": "n/a", "False_Negatives": "n/a",
                     "Source": "results/verified_transformer_results.json (FP/FN split needs the raw predictions file)"})
    return pd.DataFrame(rows)
