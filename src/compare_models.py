"""Compare executed transformers with the best CPU baseline (selected on VALIDATION macro-F1, not on test).

Outputs (all derived from result files; nothing is filled in for runs that did not execute):
  reports/tables/model_comparison.csv / .md       Model | Macro-F1 | Precision | Recall | English-F1 | RU/Mixed-F1 | Robustness | Status (+CIs)
  experiments/results/transformer_vs_baseline.csv paired cluster-bootstrap differences + McNemar (secondary)
  reports/tables/robustness_comparison.csv        degradation per perturbation type: best baseline vs transformers

Usage: python -m src.compare_models
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .data_loader import get_split, load_processed
from .evaluation import mcnemar_exact, paired_cluster_bootstrap_diff
from .make_results import CANONICAL_TRANSFORMERS, load_transformer_result, transformer_robustness_mean
from .robustness import summarise
from .verified import supplement_model
from .utils import ensure_dir, load_json

R, T = C.RESULTS_DIR, C.TABLES_DIR
RU_NOTE = "descriptive only: 5 independent test groups"


def best_baseline_id() -> str:
    return load_json(C.MODELS_DIR / "best_model.json")["model_id"]


def _fmt(v, nd=4):
    return round(float(v), nd) if v == v and v is not None else "N/A"


def comparison_table() -> pd.DataFrame:
    base = pd.read_csv(R / "baseline_results.csv")
    rob = pd.read_csv(R / "robustness_results.csv")
    bid = best_baseline_id()
    g = base[base.model_id == bid].set_index("subset")
    ov, en, ru = g.loc["Overall"], g.loc[C.LANG_ENGLISH], g.loc["Roman Urdu + Mixed"]
    brob = summarise(rob).set_index("model_id").loc[bid, "mean_degradation"]
    rows = [{"Model": f"{ov.model} (best CPU baseline, selected on validation)", "Macro-F1": _fmt(ov.macro_f1), "Macro-F1 95% CI": f"[{ov.macro_f1_ci_low:.3f}, {ov.macro_f1_ci_high:.3f}]",
             "Precision": _fmt(ov.precision_scam), "Recall": _fmt(ov.recall_scam), "English-F1": _fmt(en.macro_f1),
             "Roman Urdu/Mixed-F1": f"{_fmt(ru.macro_f1)} ({RU_NOTE})", "Robustness Degradation": _fmt(brob), "Status": "Executed (CPU)"}]
    for tag, name in CANONICAL_TRANSFORMERS.items():
        r = load_transformer_result(tag)
        sm = supplement_model(tag) if r is None else None
        if sm is not None:
            rows.append({"Model": name, "Macro-F1": sm["macro_f1"], "Macro-F1 95% CI": f"[{sm['macro_f1_ci95'][0]:.3f}, {sm['macro_f1_ci95'][1]:.3f}]",
                         "Precision": sm["precision"], "Recall": sm["recall"], "English-F1": "N/A (raw run files needed)",
                         "Roman Urdu/Mixed-F1": f"N/A (raw run files needed; {RU_NOTE})", "Robustness Degradation": sm["robustness_degradation"],
                         "Status": f"Executed on {sm['gpu']} (owner-verified; raw files not in this snapshot)"})
            continue
        if r is None:
            rows.append({"Model": name, "Macro-F1": "N/A", "Macro-F1 95% CI": "N/A", "Precision": "N/A", "Recall": "N/A", "English-F1": "N/A",
                         "Roman Urdu/Mixed-F1": "N/A", "Robustness Degradation": "N/A", "Status": "Not executed in the current environment"})
            continue
        o, e, u = r["test_overall"], next(x for x in r["test_subsets"] if x["subset"] == C.LANG_ENGLISH), next(x for x in r["test_subsets"] if x["subset"] == "Roman Urdu + Mixed")
        d = transformer_robustness_mean(tag)
        gpu = r["environment"].get("gpu_name") or r["device"]
        rows.append({"Model": name, "Macro-F1": _fmt(o["macro_f1"]), "Macro-F1 95% CI": f"[{o['macro_f1_ci_low']:.3f}, {o['macro_f1_ci_high']:.3f}]",
                     "Precision": _fmt(o["precision_scam"]), "Recall": _fmt(o["recall_scam"]), "English-F1": _fmt(e["macro_f1"]),
                     "Roman Urdu/Mixed-F1": f"{_fmt(u['macro_f1'])} ({RU_NOTE})", "Robustness Degradation": d if d is not None else "N/A (robustness not run)",
                     "Status": f"Executed ({gpu})"})
    out = pd.DataFrame(rows)
    ensure_dir(T)
    out.to_csv(T / "model_comparison.csv", index=False)
    main_cols = ["Model", "Macro-F1", "Precision", "Recall", "English-F1", "Roman Urdu/Mixed-F1", "Robustness Degradation", "Status"]
    (T / "model_comparison.md").write_text(
        "**Requested comparison table**\n\n" + out[main_cols].to_markdown(index=False) + "\n\n**With 95% cluster-bootstrap CI (overall macro-F1)**\n\n" +
        out[["Model", "Macro-F1", "Macro-F1 95% CI", "Precision", "Recall"]].to_markdown(index=False) + "\n", encoding="utf-8")
    return out


def paired_vs_baseline() -> pd.DataFrame:
    """Paired macro-F1 differences on the SAME test originals (cluster bootstrap over near-duplicate groups)."""
    df = load_processed()
    test = get_split(df, "test")
    bid = best_baseline_id()
    bp = pd.read_csv(R / "baseline_predictions.csv")
    bp = bp[bp.model_id == bid].set_index("message_id")["y_pred"].reindex(test["message_id"])
    preds = {f"best CPU baseline ({bid})": bp.values}
    for tag in CANONICAL_TRANSFORMERS:
        if load_transformer_result(tag) is None:
            continue
        tp = pd.read_csv(R / f"transformer_predictions_{tag}.csv")
        tp = tp[~tp.is_perturbed.astype(bool)].set_index("message_id")["y_pred"].reindex(test["message_id"])
        preds[tag] = tp.values
    keys = list(preds)
    masks = {"Overall": test.index == test.index, C.LANG_ENGLISH: (test.language == C.LANG_ENGLISH).values,
             "Roman Urdu + Mixed": test.language.isin([C.LANG_ROMAN_URDU, C.LANG_MIXED]).values}
    rows = []
    for i, a in enumerate(keys):
        for b in keys[:i]:
            for sname, m in masks.items():
                sub = test[m]
                d = paired_cluster_bootstrap_diff(sub["label_id"].values, preds[a][m], preds[b][m], sub["group_id"].values, "macro_f1", 2000)
                mc = mcnemar_exact(sub["label_id"].values, preds[a][m], preds[b][m])
                rows.append({"A": a, "B": b, "subset": sname, "n": int(len(sub)), "n_groups": d["n_groups"], "macro_f1_diff_A_minus_B": d["diff"],
                             "ci_low": d["ci_low"], "ci_high": d["ci_high"], "ci_excludes_zero": d["ci_excludes_zero"],
                             "mcnemar_p_optimistic_clustered": mc["p_value"],
                             "note": RU_NOTE if sname == "Roman Urdu + Mixed" else ""})
    out = pd.DataFrame(rows)
    out.to_csv(R / "transformer_vs_baseline.csv", index=False)
    return out


def robustness_comparison() -> pd.DataFrame:
    rob = pd.read_csv(R / "robustness_results.csv")
    bid = best_baseline_id()
    parts = [rob[(rob.model_id == bid)].assign(model=f"best CPU baseline ({bid})")]
    for tag in CANONICAL_TRANSFORMERS:
        f = R / f"transformer_robustness_{tag}.csv"
        if load_transformer_result(tag) is not None and f.exists():
            parts.append(pd.read_csv(f).assign(model=tag))
    allr = pd.concat(parts, ignore_index=True)
    allr = allr[(allr.language_slice == "All") & (~allr.perturbation_type.str.startswith("ALL_TYPES"))]
    out = allr.pivot_table(index="perturbation_type", columns="model", values="degradation").reindex(list(C.PERTURBATION_TYPES)).reset_index()
    out.to_csv(T / "robustness_comparison.csv", index=False)
    return out


def run() -> None:
    print(comparison_table().to_string(index=False))
    pv = paired_vs_baseline()
    print(pv.to_string(index=False) if len(pv) else "no executed transformers: no paired comparison")
    print(robustness_comparison().round(4).to_string(index=False))


if __name__ == "__main__":
    run()
