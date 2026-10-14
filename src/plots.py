"""Result figures generated from experiments/results/*.csv (nothing hard-coded). Usage: python -m src.plots"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .utils import ensure_dir

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9})
R = C.RESULTS_DIR


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def model_comparison() -> None:
    res = pd.read_csv(R / "baseline_results.csv")
    ov = res[res.subset == "Overall"].reset_index(drop=True)
    cv = pd.read_csv(R / "group_cv_results.csv")
    cv = cv[cv.subset == "Overall"].set_index("model_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(ov))[::-1]
    ax.errorbar(ov.macro_f1, y, xerr=[ov.macro_f1 - ov.macro_f1_ci_low, ov.macro_f1_ci_high - ov.macro_f1], fmt="o", color="#1f618d",
                capsize=3, label="Fixed split: test macro-F1 (95% cluster-bootstrap CI)")
    m = ov.model_id.isin(cv.index)
    ax.scatter([cv.loc[i, "macro_f1"] for i in ov.model_id[m]], y[m.values], marker="D", color="#c0392b", zorder=5,
               label="5-fold group CV macro-F1 (all originals)")
    ax.set_yticks(y, [n.replace("TFIDF-", "").replace(" (aug-trained)", " [aug]") for n in ov.model])
    ax.set(title="Baseline comparison: macro-F1 on original messages", xlabel="Macro-F1", ylabel="Model")
    ax.legend(loc="lower left")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "results_model_comparison.png")


def robustness_by_type() -> None:
    rob = pd.read_csv(R / "robustness_results.csv")
    sub = rob[(rob.language_slice == "All") & (~rob.perturbation_type.str.startswith("ALL_TYPES"))]
    models = [m for m in ["char_lr", "char_svm", "char_nb", "word_lr", "word_svm", "word_nb"] if m in set(sub.model_id)]
    types = list(C.PERTURBATION_TYPES)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        d = sub[sub.model_id == m].set_index("perturbation_type").reindex(types)
        err = [d.degradation - d.degradation_ci_low, d.degradation_ci_high - d.degradation]
        ax.bar(np.arange(len(types)) + (i - len(models) / 2 + 0.5) * w, d.degradation, w, yerr=err, capsize=1.5, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-0.1, 0.1)
    ax.text(0.99, 0.02, "y-axis clipped to +/-0.10. roman_urdu_spelling_variation applies to only 5 test groups,\nso its bars/CIs (partly off-scale) are descriptive only.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic")
    ax.set_xticks(np.arange(len(types)), [t.replace("_", "\n") for t in types], fontsize=8)
    ax.set(title="Robustness on the fixed test split: degradation = original macro-F1 - perturbed macro-F1",
           xlabel="Perturbation type", ylabel="Degradation (macro-F1 points; 95% cluster-bootstrap CI)")
    ax.legend(title="Model (orig-trained)", ncol=3)
    _save(fig, "results_robustness_by_type.png")


def robustness_cv_heatmap() -> None:
    cv = pd.read_csv(R / "group_cv_robustness.csv")
    cv = cv[~cv.perturbation_type.str.startswith("ALL_TYPES")]
    models = sorted(set(cv.model_id))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, sl in zip(axes, ["English", "Roman Urdu + Mixed"]):
        pv = cv[(cv.language_slice == sl)].pivot_table(index="perturbation_type", columns="model_id", values="degradation").reindex(list(C.PERTURBATION_TYPES))
        im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-0.03, vmax=0.03, aspect="auto")
        ax.set_xticks(range(len(models)), models)
        ax.set_yticks(range(len(pv.index)), pv.index)
        for (i, j), v in np.ndenumerate(pv.values):
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.3f}", ha="center", va="center", fontsize=8)
        ax.set(title=f"{sl}", xlabel="Model")
    axes[0].set_ylabel("Perturbation type")
    fig.colorbar(im, ax=axes, label="Degradation (positive = worse)", shrink=0.8)
    fig.suptitle("Group-CV robustness (out-of-fold): original macro-F1 - perturbed macro-F1", fontsize=13)
    fig.savefig(C.FIGURES_DIR / "results_robustness_cv_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def cross_lingual() -> None:
    x = pd.read_csv(R / "cross_lingual_results.csv")
    x = x[~x.experiment.str.startswith("C-diff")]
    models = list(dict.fromkeys(x.model_id))
    exps = list(dict.fromkeys(x.experiment))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        d = x[x.model_id == m].set_index("experiment").reindex(exps)
        err = [d.macro_f1 - d.macro_f1_ci_low, d.macro_f1_ci_high - d.macro_f1]
        ax.bar(np.arange(len(exps)) + (i - len(models) / 2 + 0.5) * w, d.macro_f1, w, yerr=err, capsize=2, label=m)
    ax.set_xticks(np.arange(len(exps)), [e.replace(": ", ":\n").replace(" -> ", "\n-> ") for e in exps], fontsize=7)
    ax.set(title="Cross-lingual experiments (fixed split). Roman Urdu+Mixed test has only 5 independent groups: wide CIs",
           xlabel="Experiment (train -> test)", ylabel="Macro-F1 (95% cluster-bootstrap CI)", ylim=(0, 1.05))
    ax.legend(title="Model")
    _save(fig, "results_cross_lingual.png")


def efficiency() -> None:
    res = pd.read_csv(R / "baseline_results.csv")
    ov = res[res.subset == "Overall"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(ov.latency_single_ms_mean, ov.model_size_mb, c=ov.macro_f1, cmap="viridis", s=70)
    for _, r in ov.iterrows():
        ax.annotate(r.model_id, (r.latency_single_ms_mean, r.model_size_mb), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, label="Test macro-F1")
    ax.set(title="Computational benchmark (CPU, this machine; not an edge-device measurement)",
           xlabel="Single-message inference latency (ms, mean)", ylabel="Model file size (MB)")
    _save(fig, "results_efficiency.png")


def transformer_comparison() -> None:
    """Best CPU baseline vs executed transformers (skipped when no transformer run exists)."""
    from .make_results import CANONICAL_TRANSFORMERS, load_transformer_result

    tx = {t: load_transformer_result(t) for t in CANONICAL_TRANSFORMERS}
    tx = {t: r for t, r in tx.items() if r is not None}
    if not tx:
        return
    cmp_ = pd.read_csv(C.TABLES_DIR / "model_comparison.csv")
    ci = cmp_["Macro-F1 95% CI"].str.strip("[]").str.split(",", expand=True).astype(float)
    cmp_ = cmp_[cmp_["Macro-F1"] != "N/A"].reset_index(drop=True)
    ci = ci.loc[cmp_.index]
    f1 = cmp_["Macro-F1"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(cmp_))[::-1]
    ax.errorbar(f1, y, xerr=[f1 - ci[0], ci[1] - f1], fmt="o", capsize=4, color="#1f618d")
    ax.set_yticks(y, [m.split(" (")[0] for m in cmp_["Model"]])
    ax.set(title="Test macro-F1 (original messages) with 95% cluster-bootstrap CI", xlabel="Macro-F1", ylabel="Model")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "results_transformer_comparison.png")
    rc = pd.read_csv(C.TABLES_DIR / "robustness_comparison.csv").set_index("perturbation_type")
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.8 / len(rc.columns)
    for i, col in enumerate(rc.columns):
        ax.bar(np.arange(len(rc)) + (i - len(rc.columns) / 2 + 0.5) * w, rc[col], w, label=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(rc)), [t.replace("_", "\n") for t in rc.index], fontsize=8)
    ax.set(title="Robustness degradation (original macro-F1 - perturbed macro-F1), fixed test split",
           xlabel="Perturbation type", ylabel="Degradation (macro-F1 points)")
    ax.legend(title="Model")
    _save(fig, "results_robustness_transformers.png")


def run() -> None:
    ensure_dir(C.FIGURES_DIR)
    for f in (model_comparison, robustness_by_type, robustness_cv_heatmap, cross_lingual, efficiency, transformer_comparison):
        f()


if __name__ == "__main__":
    run()
