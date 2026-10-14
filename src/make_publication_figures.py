"""Publication figures (PNG 300 dpi + PDF) generated ONLY from results/FINAL_RESULTS.csv, results/ROBUSTNESS_RESULTS.csv and
experiments/results/baseline_results.csv. Usage: python -m src.make_publication_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config as C

FIG = C.ROOT_DIR / "figures"
RES = C.ROOT_DIR / "results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"CPU baseline (TF-IDF)": "#4c72b0", "Transformer": "#c44e52"}


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_macro_f1() -> None:
    d = pd.read_csv(RES / "FINAL_RESULTS.csv", keep_default_na=False)
    d = d[d.Macro_F1 != "N/A"].copy()
    d["f1"] = d.Macro_F1.astype(float)
    ci = d.Macro_F1_CI95.str.strip("[]").str.split(",", expand=True).astype(float)
    d["lo"], d["hi"] = ci[0], ci[1]
    d = d.sort_values("f1").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d.f1, color=[COL[f] for f in d.Family], height=0.62, zorder=2)
    ax.errorbar(d.f1, y, xerr=[d.f1 - d.lo, d.hi - d.f1], fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=3)
    for i, r in d.iterrows():
        ax.text(r.hi + 0.004, i, f"{r.f1:.4f}", va="center", fontsize=9)
    ax.set_yticks(y, d.Model)
    ax.set_xlim(0.6, 1.04)
    ax.set_xlabel("Test macro-F1 (935 original messages; error bars: 95% cluster-bootstrap CI)")
    ax.set_title("Macro-F1 of all major models on the fixed, group-level test split")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], labels=list(COL), loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.text(0.0, -0.30, "CIs overlap heavily among the top four models. Paired XLM-R vs selected Char SVM: difference -0.0019, 95% CI [-0.0188, 0.0133] (includes zero).",
            transform=ax.transAxes, fontsize=8, style="italic")
    _save(fig, "model_macro_f1_comparison")


def fig_robustness() -> None:
    r = pd.read_csv(RES / "ROBUSTNESS_RESULTS.csv", keep_default_na=False)
    r = r[r.Level == "mean over 10 perturbation types"]
    pick = [("Char TF-IDF + Linear SVM", "Char TF-IDF\n+ Linear SVM"), ("mBERT", "mBERT"), ("XLM-R", "XLM-R")]
    vals = [float(r[r.Model == m].Degradation.iloc[0]) for m, _ in pick]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar([p[1] for p in pick], vals, color=["#4c72b0", "#c44e52", "#c44e52"], width=0.55, zorder=2)
    ax.axhline(0, color="black", lw=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.0012 if v >= 0 else -0.0012), f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.set_ylim(-0.012, 0.042)
    ax.set_ylabel("Degradation (macro-F1 points)\noriginal - perturbed; lower is better", fontsize=9.5)
    ax.set_title("Robustness to the project's synthetic perturbation suite")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(0.5, -0.2, "Mean over 10 synthetic perturbation types, paired test messages. Negative = perturbed macro-F1 slightly\nhigher than original (not interpreted as a robustness gain). Not a measure of real-world adversarial robustness.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8, style="italic")
    _save(fig, "robustness_degradation")


def fig_pipeline() -> None:
    steps = [("Dataset", "UCI SMS Spam +\nRoman-Urdu phishing SMS\n6,160 originals"), ("Grouped\nsplit", "near-duplicate groups\n70 / 15 / 15\n(test: 935, 778 groups)"),
             ("Synthetic\nperturbations", "10 noise types\n52,079 variants\n(inherit split)"), ("TF-IDF\nbaselines", "char / word\nLR, Linear SVM, NB"),
             ("mBERT /\nXLM-R", "fine-tuned, max 4 epochs\nearly stop on val macro-F1"), ("Evaluation", "macro-F1, precision, recall\n95% cluster-bootstrap CI"),
             ("Robustness\nanalysis", "paired degradation\non perturbed test variants"), ("Error\nanalysis", "per-model error counts,\nFP / FN categories"),
             ("Final QA", "tests, leakage checks,\nchecksum, metric tracing")]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    pos = [(0.15 + i * 2.15, 3.6) for i in range(5)] + [(0.15 + i * 2.15 + 1.07, 1.35) for i in range(4)]
    w, h = 1.95, 1.35
    for i, ((x, y), (t, sub)) in enumerate(zip(pos, steps)):
        c = "#4c72b0" if i < 4 else ("#c44e52" if i == 4 else "#55a868")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=c, lw=1.8))
        ax.text(x + w / 2, y + h - 0.32, f"{i + 1}. {t}", ha="center", va="center", fontsize=9.2, fontweight="bold", color=c, linespacing=1.05)
        ax.text(x + w / 2, y + 0.36, sub, ha="center", va="center", fontsize=7.4)
    arrows = [(pos[i][0] + w, pos[i][1] + h / 2, pos[i + 1][0], pos[i + 1][1] + h / 2) for i in range(4)]
    arrows.append((pos[4][0] + w / 2, pos[4][1], pos[5][0] + w / 2, pos[5][1] + h))
    arrows += [(pos[i][0] + w, pos[i][1] + h / 2, pos[i + 1][0], pos[i + 1][1] + h / 2) for i in range(5, 8)]
    for x0, y0, x1, y1 in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#444444"))
    ax.text(5.5, 5.15, "MARS-NLP experimental pipeline", ha="center", fontsize=13, fontweight="bold")
    ax.text(5.5, 0.55, "Perturbed variants always stay in their original message's split; test originals and test variants are never used for training or model selection.",
            ha="center", fontsize=8.2, style="italic")
    _save(fig, "mars_nlp_pipeline")


def run() -> None:
    for f in (fig_macro_f1, fig_robustness, fig_pipeline):
        f()
    print("figures written to", FIG)


if __name__ == "__main__":
    run()
