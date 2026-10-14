"""Exploratory data analysis: every figure is computed from the processed dataset (nothing hard-coded).

Usage: python -m src.eda
Outputs: reports/figures/eda_*.png and reports/tables/eda_*.csv
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_processed
from .utils import ensure_dir

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10})
LABEL_COLORS = {"Legitimate": "#2874a6", "Scam": "#c0392b"}
SHORT_SRC = {C.SRC_UCI: "UCI SMS Spam", C.SRC_ZENODO: "Zenodo Roman-Urdu phishing"}


def _annotate(ax, fmt="{:d}"):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(int(h)), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=9)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def run() -> dict:
    ensure_dir(C.FIGURES_DIR), ensure_dir(C.TABLES_DIR)
    df = load_processed()
    orig = df[~df["is_perturbed"].astype(bool)].copy()
    var = df[df["is_perturbed"].astype(bool)].copy()
    orig["length"] = orig["cleaned_message"].str.len()
    orig["source"] = orig["source_dataset"].map(SHORT_SRC)
    stats = {}

    # 1 class distribution
    t = orig["label"].value_counts().reindex(C.LABELS)
    t.rename("count").to_frame().assign(share=lambda x: x["count"] / x["count"].sum()).to_csv(C.TABLES_DIR / "eda_class_distribution.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(t.index, t.values, color=[LABEL_COLORS[k] for k in t.index])
    _annotate(ax)
    ax.set(title="Class distribution (original messages)", xlabel="Label", ylabel="Number of messages")
    _save(fig, "eda_class_distribution.png")

    # 2 language distribution
    t = orig["language"].value_counts().reindex(C.LANGUAGES).fillna(0).astype(int)
    t.rename("count").to_frame().to_csv(C.TABLES_DIR / "eda_language_distribution.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(t.index, t.values, color="#566573")
    _annotate(ax)
    ax.set(title="Language distribution (heuristic labels, original messages)", xlabel="Language", ylabel="Number of messages")
    ax.tick_params(axis="x", labelrotation=15)
    _save(fig, "eda_language_distribution.png")

    # 3 class by language
    ct = pd.crosstab(orig["language"], orig["label"]).reindex(C.LANGUAGES).fillna(0).astype(int)
    ct.to_csv(C.TABLES_DIR / "eda_class_by_language.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ct))
    for i, lab in enumerate(C.LABELS):
        ax.bar(x + (i - 0.5) * 0.38, ct.get(lab, 0), 0.38, label=lab, color=LABEL_COLORS[lab])
    ax.set_xticks(x, ct.index)
    _annotate(ax)
    ax.set(title="Class by language (original messages)", xlabel="Language", ylabel="Number of messages")
    ax.legend(title="Label")
    ax.tick_params(axis="x", labelrotation=15)
    _save(fig, "eda_class_by_language.png")

    # 4 message length
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, min(500, orig["length"].max()), 50)
    for lab in C.LABELS:
        ax.hist(orig.loc[orig.label == lab, "length"].clip(upper=bins[-1]), bins=bins, alpha=0.6, label=lab, color=LABEL_COLORS[lab])
    ax.set(title="Message length distribution (characters; values >500 clipped)", xlabel="Characters per message", ylabel="Number of messages")
    ax.legend(title="Label")
    _save(fig, "eda_message_length.png")
    orig.groupby(["source", "label"])["length"].describe().round(1).to_csv(C.TABLES_DIR / "eda_length_stats.csv")

    # 5 source distribution
    ct = pd.crosstab(orig["source"], orig["label"]).reindex(columns=list(C.LABELS)).fillna(0).astype(int)
    ct.to_csv(C.TABLES_DIR / "eda_source_distribution.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(ct))
    for lab in C.LABELS:
        ax.bar(ct.index, ct[lab], bottom=bottom, label=lab, color=LABEL_COLORS[lab])
        for i, v in enumerate(ct[lab]):
            if v > 0:
                ax.text(i, bottom[i] + v / 2, str(int(v)), ha="center", va="center", color="white", fontsize=10)
        bottom += ct[lab].values
    ax.set(title="Source distribution (original messages, after exact-duplicate removal)", xlabel="Source dataset", ylabel="Number of messages")
    ax.legend(title="Label")
    _save(fig, "eda_source_distribution.png")

    # 6 perturbation distribution
    t = var["perturbation_type"].value_counts().reindex(C.PERTURBATION_TYPES).fillna(0).astype(int)
    t.rename("count").to_frame().to_csv(C.TABLES_DIR / "eda_perturbation_distribution.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(t.index[::-1], t.values[::-1], color="#7d6608")
    for i, v in enumerate(t.values[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set(title="Perturbed variants per perturbation type", xlabel="Number of variants", ylabel="Perturbation type")
    _save(fig, "eda_perturbation_distribution.png")

    # 7 code-switch distribution
    cs = pd.DataFrame({"originals": orig["is_code_switched"].astype(bool).value_counts(),
                       "perturbed variants": var["is_code_switched"].astype(bool).value_counts()}).fillna(0).astype(int)
    cs.index = ["code-switched" if i else "not code-switched" for i in cs.index]
    cs.to_csv(C.TABLES_DIR / "eda_code_switch_distribution.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cs.columns))
    w = 0.38
    for i, (idx, row) in enumerate(cs.iterrows()):
        ax.bar(x + (i - 0.5) * w, row.values, w, label=idx, color=["#5d6d7e", "#d68910"][i % 2])
    ax.set_xticks(x, cs.columns)
    _annotate(ax)
    ax.set(title="Code-switch flag distribution", xlabel="Subset", ylabel="Number of messages")
    ax.legend(title="Flag")
    _save(fig, "eda_code_switch_distribution.png")

    # extra: split distribution
    ct = pd.crosstab(orig["split"], orig["label"]).reindex(list(C.SPLITS)).reindex(columns=list(C.LABELS)).fillna(0).astype(int)
    ct.to_csv(C.TABLES_DIR / "eda_split_distribution.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(ct))
    for i, lab in enumerate(C.LABELS):
        ax.bar(x + (i - 0.5) * 0.38, ct[lab], 0.38, label=lab, color=LABEL_COLORS[lab])
    ax.set_xticks(x, ct.index)
    _annotate(ax)
    ax.set(title="Group-level split sizes (original messages)", xlabel="Split", ylabel="Number of messages")
    ax.legend(title="Label")
    _save(fig, "eda_split_distribution.png")

    # template-group structure per source (explains the leakage design)
    g = orig.groupby(["source", "group_id"]).size().rename("size").reset_index()
    summ = g.groupby("source")["size"].agg(n_groups="size", max_group="max", mean_group="mean", n_messages="sum").round(2)
    summ.to_csv(C.TABLES_DIR / "eda_group_structure.csv")
    stats["group_structure"] = summ.to_dict()
    return stats


if __name__ == "__main__":
    print(run())
