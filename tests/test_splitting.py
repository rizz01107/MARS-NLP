import pandas as pd
import pytest

from src import config as C
from src.splitting import assign_groups, check_group_leakage, group_stratified_split, propagate_split_to_variants
from conftest import make_synthetic


def _with_variants(df):
    var = df.copy()
    var["message_id"] = var["message_id"] + "__v"
    var["original_message_id"] = df["message_id"].values
    var["is_perturbed"] = True
    var["source_type"] = C.SOURCE_AUGMENTED
    var["perturbation_type"] = "minor_noise"
    var["split"] = "train"
    return pd.concat([df, var], ignore_index=True)


def test_group_split_keeps_groups_together():
    df = make_synthetic(80)
    df["split"] = group_stratified_split(df, "group_id")
    assert (df.groupby("group_id")["split"].nunique() == 1).all()


def test_split_fractions_roughly_respected_and_seeded():
    df = make_synthetic(300)
    s1 = group_stratified_split(df, "group_id", seed=1)
    s2 = group_stratified_split(df, "group_id", seed=1)
    s3 = group_stratified_split(df, "group_id", seed=2)
    assert s1.equals(s2) and not s1.equals(s3)
    frac = s1.value_counts(normalize=True)
    for split, target in zip(C.SPLITS, C.SPLIT_FRACTIONS):
        assert abs(frac[split] - target) < 0.05


def test_stratification_both_classes_in_every_split():
    df = make_synthetic(200)
    df["split"] = group_stratified_split(df, "group_id")
    assert all(set(df[df.split == s]["label"]) == set(C.LABELS) for s in C.SPLITS)


def test_variants_inherit_split_no_leakage():
    df = make_synthetic(100)
    df["split"] = group_stratified_split(df, "group_id")
    full = _with_variants(df)
    full["split"] = propagate_split_to_variants(full)
    leak = check_group_leakage(full)
    assert leak["groups_in_multiple_splits"] == 0 and leak["variants_with_split_different_from_original"] == 0


def test_leakage_is_detected():
    df = make_synthetic(30)
    df["split"] = group_stratified_split(df, "group_id")
    full = _with_variants(df)
    full["split"] = propagate_split_to_variants(full)
    i = full.index[full["is_perturbed"]][0]
    full.loc[i, "split"] = "test" if full.loc[i, "split"] != "test" else "train"   # corrupt one variant
    assert check_group_leakage(full)["variants_with_split_different_from_original"] >= 1


def test_missing_group_id_rejected_and_bad_fractions():
    df = make_synthetic(10)
    df.loc[0, "group_id"] = None
    with pytest.raises(ValueError):
        group_stratified_split(df, "group_id")
    with pytest.raises(ValueError):
        assign_groups({"a": 1}, {"a": "x"}, fractions=(0.5, 0.2, 0.1))
