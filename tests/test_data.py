import pandas as pd
import pytest

from src import config as C
from src.data_loader import get_split, load_processed, load_raw_uci
from src.validate_data import validate
from conftest import make_synthetic


def test_missing_processed_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        load_processed(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        load_raw_uci(tmp_path / "nope")


def test_validate_flags_corruption():
    df = make_synthetic(20)
    df["split"] = ["train"] * len(df)
    bad = df.copy()
    bad.loc[0, "label"] = "Maybe"
    bad.loc[1, "message_id"] = bad.loc[2, "message_id"]
    bad.loc[3, "split"] = "dev"
    checks, _ = validate(bad)
    failed = {c.name for c in checks if c.status == "FAIL"}
    assert {"allowed_labels", "unique_message_id", "allowed_split"} <= failed


def test_validate_missing_columns():
    checks, _ = validate(pd.DataFrame({"message": ["x"]}))
    assert checks[0].name == "required_columns" and checks[0].status == "FAIL"


def test_processed_dataset_passes_validation(processed_df):
    checks, _ = validate(processed_df)
    failed = [c for c in checks if c.status == "FAIL"]
    assert not failed, failed


def test_processed_schema_and_values(processed_df):
    df = processed_df
    assert list(df.columns[:14]) == C.README_COLUMNS
    assert set(df["label"]) <= set(C.LABELS) and set(df["language"]) <= set(C.LANGUAGES)
    assert set(df["split"]) == set(C.SPLITS) and set(df["source_type"]) == set(C.SOURCE_TYPES)
    assert df["message_id"].is_unique


def test_language_handling_and_no_urdu_script_claim(processed_df):
    o = processed_df[~processed_df.is_perturbed]
    assert (o.language == C.LANG_URDU).sum() == 0         # documented: no real Urdu-script data in the sources
    assert o[o.language == C.LANG_MIXED].is_code_switched.all()


def test_get_split_originals_only_by_default(processed_df):
    for s in C.SPLITS:
        d = get_split(processed_df, s)
        assert not d.is_perturbed.any() and (d.split == s).all()
    with_var = get_split(processed_df, "test", include_perturbed=True)
    assert with_var.is_perturbed.any()
    with pytest.raises(ValueError):
        get_split(processed_df, "dev")


def test_no_leakage_in_real_data(processed_df):
    from src.splitting import check_group_leakage

    leak = check_group_leakage(processed_df)
    assert leak["groups_in_multiple_splits"] == 0 and leak["variants_with_split_different_from_original"] == 0
    o = processed_df[~processed_df.is_perturbed]
    sets = {s: set(o[o.split == s].cleaned_message) for s in C.SPLITS}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["validation"]) and not (sets["validation"] & sets["test"])


def test_variants_reference_existing_originals_with_same_label(processed_df):
    o = processed_df[~processed_df.is_perturbed].set_index("message_id")
    v = processed_df[processed_df.is_perturbed]
    assert v.original_message_id.isin(o.index).all()
    assert (v.original_message_id.map(o.label) == v.label).all()
    assert (v.original_message_id.map(o.split) == v.split).all()
