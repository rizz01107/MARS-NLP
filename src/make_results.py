"""Assemble experiments/results/final_results.csv from the executed experiments (no values are invented).

Usage: python -m src.make_results
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .robustness import summarise
from .verified import SOURCE_LABEL, supplement_model
from .utils import ensure_dir

NOT_RUN = "Not executed in the current environment"
TRANSFORMER_CMD = {
    "mBERT (bert-base-multilingual-cased)": "python -m src.train_transformer --model mbert --epochs 4 --batch-size 32",
    "XLM-RoBERTa (xlm-roberta-base)": "python -m src.train_transformer --model xlmr --epochs 4 --batch-size 32",
}


def _lang_f1(sub: pd.DataFrame, subset: str):
    r = sub[sub.subset == subset]
    if r.empty:
        return "N/A (no data)"
    r = r.iloc[0]
    if pd.isna(r.macro_f1):
        return f"N/A ({r.status})"
    return round(float(r.macro_f1), 4)


def _cv(cv: pd.DataFrame, mid: str, subset: str):
    if cv.empty:
        return "N/A"
    r = cv[(cv.model_id == mid) & (cv.subset == subset)]
    return round(float(r.iloc[0].macro_f1), 4) if len(r) else "N/A (not run for this configuration)"


CANONICAL_TRANSFORMERS = {"mbert": "mBERT (bert-base-multilingual-cased)", "xlmr": "XLM-RoBERTa (xlm-roberta-base)"}
_TX_COLS = ["Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "Macro_F1_CI95", "English_F1", "Urdu_F1", "Roman_Urdu_F1",
            "Mixed_F1", "RomanUrdu_plus_Mixed_F1", "Parameters", "Model_Size", "Inference_Latency", "Robustness_Degradation",
            "GroupCV_Macro_F1", "GroupCV_RomanUrdu_plus_Mixed_F1", "GroupCV_Robustness_Degradation", "Train_Seconds"]


def load_transformer_result(tag: str):
    """Return the canonical, non-smoke, pretrained run JSON for ``tag`` (mbert/xlmr) or None."""
    f = C.RESULTS_DIR / f"transformer_{tag}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    if r.get("smoke_test") or not r.get("pretrained", True) or r.get("config", {}).get("train_languages") or r.get("config", {}).get("augment"):
        return None
    return r


def _sub(r: dict, name: str):
    return next((x for x in r["test_subsets"] if x["subset"] == name), None)


def _sub_f1(r: dict, name: str):
    x = _sub(r, name)
    if x is None:
        return "N/A (no data)"
    v = x.get("macro_f1")
    if v is None or v != v:
        return f"N/A ({x.get('status', 'no data')})"
    return round(float(v), 4)


def transformer_robustness_mean(tag: str):
    f = C.RESULTS_DIR / f"transformer_robustness_{tag}.csv"
    if not f.exists():
        return None
    rs = summarise(pd.read_csv(f))
    return round(float(rs.iloc[0]["mean_degradation"]), 4) if len(rs) else None


def _transformer_rows() -> list[dict]:
    """Rows for fine-tuned transformers: filled ONLY from executed-run files (transformer_<tag>.json + robustness CSV)."""
    rows = []
    for tag, name in CANONICAL_TRANSFORMERS.items():
        r = load_transformer_result(tag)
        cmd = TRANSFORMER_CMD[name]
        sm = supplement_model(tag) if r is None else None
        if sm is not None:
            na = "N/A (per-subset values exist only in the raw run files)"
            rows.append({"Model": name, "Accuracy": sm["accuracy"], "Precision": sm["precision"], "Recall": sm["recall"], "Macro_F1": sm["macro_f1"],
                         "Weighted_F1": sm["weighted_f1"], "Macro_F1_CI95": f"[{sm['macro_f1_ci95'][0]:.3f}, {sm['macro_f1_ci95'][1]:.3f}]",
                         "English_F1": na, "Urdu_F1": "N/A (no Urdu-script data)", "Roman_Urdu_F1": na, "Mixed_F1": na, "RomanUrdu_plus_Mixed_F1": na,
                         "Parameters": sm["parameters"], "Model_Size": f"{sm['model_size_mb']:.1f} MB",
                         "Inference_Latency": f"cuda: {sm['gpu_latency_ms_per_message']:.2f} ms/msg; cpu: {sm['cpu_latency_ms_per_message']:.2f} ms/msg "
                                              f"(single-message computational benchmark; {sm['gpu']})",
                         "Robustness_Degradation": sm["robustness_degradation"], "GroupCV_Macro_F1": "N/A (not run for transformers)",
                         "GroupCV_RomanUrdu_plus_Mixed_F1": "N/A (not run for transformers)", "GroupCV_Robustness_Degradation": "N/A (not run for transformers)",
                         "Train_Seconds": sm["train_seconds"], "Status": f"Reported ({SOURCE_LABEL})",
                         "Notes": f"fine-tuned on {sm['train_samples']} training messages on {sm['gpu']}; best epoch {sm['best_epoch']}; "
                                  "source: results/verified_transformer_results.json"})
            continue
        if r is None:
            rows.append({"Model": name, **{k: "N/A" for k in _TX_COLS}, "Status": NOT_RUN,
                         "Notes": f"Requires GPU/torch + Hugging Face Hub access. Run: {cmd}"})
            continue
        o = r["test_overall"]
        lat = r["latency_computational_benchmark"]
        lat_s = "; ".join(f"{d}: {v['latency_single_ms_mean']:.2f} ms/msg" for d, v in lat.items())
        env = r["environment"]
        rob = transformer_robustness_mean(tag)
        rows.append({"Model": name, "Accuracy": round(o["accuracy"], 4), "Precision": round(o["precision_scam"], 4),
                     "Recall": round(o["recall_scam"], 4), "Macro_F1": round(o["macro_f1"], 4), "Weighted_F1": round(o["weighted_f1"], 4),
                     "Macro_F1_CI95": f"[{o['macro_f1_ci_low']:.3f}, {o['macro_f1_ci_high']:.3f}]",
                     "English_F1": _sub_f1(r, C.LANG_ENGLISH), "Urdu_F1": _sub_f1(r, C.LANG_URDU),
                     "Roman_Urdu_F1": _sub_f1(r, C.LANG_ROMAN_URDU), "Mixed_F1": _sub_f1(r, C.LANG_MIXED),
                     "RomanUrdu_plus_Mixed_F1": _sub_f1(r, "Roman Urdu + Mixed"), "Parameters": r["parameters"]["total"],
                     "Model_Size": f"{r['model_size_mb']:.1f} MB",
                     "Inference_Latency": f"{lat_s} (single-message computational benchmark; {env.get('gpu_name', 'CPU')})",
                     "Robustness_Degradation": rob if rob is not None else "N/A (robustness not run)",
                     "GroupCV_Macro_F1": "N/A (not run for transformers)", "GroupCV_RomanUrdu_plus_Mixed_F1": "N/A (not run for transformers)",
                     "GroupCV_Robustness_Degradation": "N/A (not run for transformers)", "Train_Seconds": round(r["train_seconds"], 1),
                     "Status": "Executed",
                     "Notes": f"fine-tuned on {r['n_train']} training messages; device {r['device']}"
                              + (f" ({env.get('gpu_name')})" if env.get("gpu_name") else "")
                              + f"; best epoch {r['best_epoch']}; RU+Mixed F1 is descriptive only (5 independent test groups)"})
    return rows


def build() -> pd.DataFrame:
    res = pd.read_csv(C.RESULTS_DIR / "baseline_results.csv")
    rob = pd.read_csv(C.RESULTS_DIR / "robustness_results.csv")
    rs = summarise(rob).set_index("model_id")
    cv_path = C.RESULTS_DIR / "group_cv_robustness.csv"
    rs_cv = summarise(pd.read_csv(cv_path)).set_index("model_id") if cv_path.exists() else pd.DataFrame()
    cvr_path = C.RESULTS_DIR / "group_cv_results.csv"
    cv = pd.read_csv(cvr_path) if cvr_path.exists() else pd.DataFrame()
    rows = []
    for mid, g in res.groupby("model_id", sort=False):
        ov = g[g.subset == "Overall"].iloc[0]
        rows.append({
            "Model": ov.model, "Accuracy": round(ov.accuracy, 4), "Precision": round(ov.precision_scam, 4),
            "Recall": round(ov.recall_scam, 4), "Macro_F1": round(ov.macro_f1, 4), "Weighted_F1": round(ov.weighted_f1, 4),
            "Macro_F1_CI95": f"[{ov.macro_f1_ci_low:.3f}, {ov.macro_f1_ci_high:.3f}]",
            "English_F1": _lang_f1(g, C.LANG_ENGLISH), "Urdu_F1": _lang_f1(g, C.LANG_URDU),
            "Roman_Urdu_F1": _lang_f1(g, C.LANG_ROMAN_URDU), "Mixed_F1": _lang_f1(g, C.LANG_MIXED),
            "RomanUrdu_plus_Mixed_F1": _lang_f1(g, "Roman Urdu + Mixed"),
            "Parameters": int(ov.parameters), "Model_Size": f"{ov.model_size_mb:.2f} MB",
            "Inference_Latency": f"{ov.latency_single_ms_mean:.3f} ms/msg (CPU, single-message computational benchmark)",
            "Robustness_Degradation": round(float(rs.loc[mid, "mean_degradation"]), 4) if mid in rs.index else "N/A",
            "GroupCV_Macro_F1": _cv(cv, mid, "Overall"), "GroupCV_RomanUrdu_plus_Mixed_F1": _cv(cv, mid, "Roman Urdu + Mixed"),
            "GroupCV_Robustness_Degradation": round(float(rs_cv.loc[mid, "mean_degradation"]), 4) if mid in rs_cv.index else "N/A",
            "Train_Seconds": round(float(ov.train_seconds), 2), "Status": "Executed",
            "Notes": "Parameters = classifier weights; metrics on test originals; Precision/Recall = Scam class; "
                     "Robustness_Degradation = mean over perturbation types of (original_f1 - perturbed_f1); "
                     "Mixed_F1 and RomanUrdu_plus_Mixed_F1 rest on only 4-5 independent test groups (descriptive only)",
        })
    rows += _transformer_rows()
    out = pd.DataFrame(rows)
    ensure_dir(C.RESULTS_DIR)
    out.to_csv(C.RESULTS_DIR / "final_results.csv", index=False)
    return out


if __name__ == "__main__":
    print(build().to_string(index=False))
