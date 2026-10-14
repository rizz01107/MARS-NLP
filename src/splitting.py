"""Group-level stratified splitting (prevents leakage between an original and its variants/near-duplicates).

Methodology
-----------
1. Genuine originals are clustered into *groups* (exact duplicates were removed beforehand; near-duplicates and
   templated messages are merged by :func:`src.preprocessing.near_duplicate_groups`). Each original without a
   near-duplicate is its own group.
2. Whole groups are assigned to train/validation/test with a seeded, size-aware greedy procedure that stratifies
   by (group language, group label) so that class/language proportions are roughly preserved at the message level.
3. Every perturbed variant inherits the split of its original (``original_message_id``), hence of its group.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as C


def _majority(values: pd.Series, prefer: str | None = None) -> str:
    counts = values.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    if prefer is not None and prefer in top:
        return prefer
    return sorted(top)[0]


def assign_groups(group_sizes: Mapping, group_strata: Mapping, fractions: Sequence[float] = C.SPLIT_FRACTIONS,
                  seed: int = C.SEED, split_names: Sequence[str] = C.SPLITS) -> dict:
    """Assign whole groups to splits, stratified; returns ``{group_id: split}``."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    assignment: dict = {}
    for stratum in sorted(set(group_strata.values()), key=str):
        gids = [g for g in group_sizes if group_strata[g] == stratum]
        gids = sorted(gids, key=str)
        rng.shuffle(gids)                       # random tie-breaking
        gids.sort(key=lambda g: -group_sizes[g])  # large groups first (stable sort keeps shuffled ties)
        total = float(sum(group_sizes[g] for g in gids))
        target = {s: f * total for s, f in zip(split_names, fractions)}
        current = {s: 0.0 for s in split_names}
        for g in gids:
            s = max(split_names, key=lambda name: target[name] - current[name])
            assignment[g] = s
            current[s] += group_sizes[g]
    return assignment


DEFAULT_STRATA = ("source_dataset", "label")


def group_stratified_split(originals: pd.DataFrame, group_col: str = "group_id",
                           fractions: Sequence[float] = C.SPLIT_FRACTIONS, seed: int = C.SEED,
                           strata_cols: Sequence[str] = DEFAULT_STRATA) -> pd.Series:
    """Return a Series (index = ``originals.index``) with the split of every original row.

    Groups are stratified by the majority value of ``strata_cols`` (default: source dataset and label; the label
    majority prefers ``Scam`` on ties). Stratifying by *source* rather than detected language keeps each source's
    templates represented in every split even when a source has very few template groups.
    """
    if originals[group_col].isna().any():
        raise ValueError("group_id must be defined for every original message")
    g = originals.groupby(group_col)
    sizes = g.size().to_dict()
    parts = {}
    for col in strata_cols:
        prefer = "Scam" if col == "label" else None
        parts[col] = g[col].agg(lambda s, p=prefer: _majority(s, prefer=p)).to_dict()
    strata = {k: "|".join(str(parts[c][k]) for c in strata_cols) for k in sizes}
    assignment = assign_groups(sizes, strata, fractions, seed)
    return originals[group_col].map(assignment)


def propagate_split_to_variants(df: pd.DataFrame) -> pd.Series:
    """Give each perturbed row the split of its original (via ``original_message_id``)."""
    originals = df.loc[~df["is_perturbed"].astype(bool)].drop_duplicates("message_id").set_index("message_id")["split"]
    out = df["split"].copy()
    mask = df["is_perturbed"].astype(bool)
    out.loc[mask] = df.loc[mask, "original_message_id"].map(originals)
    return out


def check_group_leakage(df: pd.DataFrame, group_col: str = "group_id") -> dict:
    """Return a dict describing leakage; all counts must be zero for a valid split."""
    n_splits = df.groupby(group_col)["split"].nunique()
    leaking_groups = n_splits[n_splits > 1].index.tolist()
    originals = df.loc[~df["is_perturbed"].astype(bool)].drop_duplicates("message_id").set_index("message_id")["split"]
    var = df.loc[df["is_perturbed"].astype(bool)]
    mapped = var["original_message_id"].map(originals)
    variant_mismatch = int((mapped != var["split"]).sum())
    return {"groups_in_multiple_splits": len(leaking_groups), "example_groups": leaking_groups[:5],
            "variants_with_split_different_from_original": variant_mismatch}
