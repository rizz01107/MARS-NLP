"""Data integrity + leakage validation.

Usage:  python -m src.validate_data          (exit code 1 if any CRITICAL check fails)
Writes: reports/tables/validation_report.md / .json
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_processed
from .preprocessing import template_key
from .splitting import check_group_leakage
from .utils import ensure_dir, save_json

CRITICAL, WARNING, INFO = "CRITICAL", "WARNING", "INFO"


@dataclass
class Check:
    name: str
    status: str      # PASS / FAIL / WARN / INFO
    severity: str
    detail: str
    count: int = 0


def cross_split_similarity(df: pd.DataFrame, a: str = "test", b: str = "train", max_rows: int = 3000) -> dict:
    """Max cosine similarity (char n-gram TF-IDF) between each original in split ``a`` and any original in ``b``."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    orig = df.loc[~df["is_perturbed"].astype(bool)]
    A, B = orig[orig["split"] == a], orig[orig["split"] == b]
    if len(A) == 0 or len(B) == 0:
        return {"n": 0}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    vec.fit([template_key(t) for t in orig["cleaned_message"]])
    XA = vec.transform([template_key(t) for t in A["cleaned_message"]])
    XB = vec.transform([template_key(t) for t in B["cleaned_message"]]).T.tocsc()
    mx = np.zeros(XA.shape[0])
    for s in range(0, XA.shape[0], 300):
        mx[s:s + 300] = (XA[s:s + 300] @ XB).toarray().max(axis=1)
    return {"n": int(len(mx)), "mean_max_sim": float(mx.mean()), "p95_max_sim": float(np.quantile(mx, 0.95)),
            "share_ge_0.80": float((mx >= 0.80).mean()), "share_ge_0.90": float((mx >= 0.90).mean()),
            "share_ge_0.95": float((mx >= 0.95).mean())}


def validate(df: pd.DataFrame) -> tuple[list[Check], dict]:
    checks: list[Check] = []

    def add(name, ok, detail="", count=0, severity=CRITICAL, info=False):
        status = "INFO" if info else ("PASS" if ok else ("FAIL" if severity == CRITICAL else "WARN"))
        checks.append(Check(name, status, severity, detail if (not ok or info) else (detail or "ok"), int(count)))

    missing_cols = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    add("required_columns", not missing_cols, f"missing: {missing_cols}" if missing_cols else "all present", len(missing_cols))
    if missing_cols:
        return checks, {}

    is_p = df["is_perturbed"].map(lambda v: v if isinstance(v, (bool, np.bool_)) else None)
    bad_bool = int(is_p.isna().sum()) + int(df["is_code_switched"].map(lambda v: not isinstance(v, (bool, np.bool_))).sum())
    add("boolean_columns_valid", bad_bool == 0, f"{bad_bool} invalid boolean values", bad_bool)
    is_p = is_p.fillna(False).astype(bool)
    orig, var = df[~is_p], df[is_p]

    for col in ["message_id", "message", "cleaned_message", "label", "language", "source_type", "split", "group_id",
                "source_dataset", "perturbation_type"]:
        n = int(df[col].isna().sum())
        add(f"no_missing_{col}", n == 0, f"{n} missing", n)
    n = int(var["original_message_id"].isna().sum())
    add("perturbed_rows_have_original_message_id", n == 0, f"{n} perturbed rows without original_message_id", n)
    n = int(orig["original_message_id"].notna().sum())
    add("originals_have_no_original_message_id", n == 0, f"{n} originals with original_message_id (each is its own source)", n, WARNING)

    n = int(df["message_id"].duplicated().sum())
    add("unique_message_id", n == 0, f"{n} duplicated message_id", n)

    bad = df[~df["label"].isin(C.LABELS)]
    add("allowed_labels", bad.empty, f"invalid: {bad['label'].unique()[:5].tolist()}", len(bad))
    bad = df[~df["language"].isin(C.LANGUAGES)]
    add("allowed_languages", bad.empty, f"invalid: {bad['language'].unique()[:5].tolist()}", len(bad))
    bad = df[~df["source_type"].isin(C.SOURCE_TYPES)]
    add("allowed_source_type", bad.empty, f"invalid: {bad['source_type'].unique()[:5].tolist()}", len(bad))
    inconsistent = int(((df["source_type"] == C.SOURCE_AUGMENTED) != is_p).sum())
    add("source_type_consistent_with_is_perturbed", inconsistent == 0, f"{inconsistent} inconsistent rows", inconsistent)
    bad = var[~var["perturbation_type"].isin(C.PERTURBATION_TYPES)]
    add("allowed_perturbation_type", bad.empty, f"invalid: {bad['perturbation_type'].unique()[:5].tolist()}", len(bad))
    bad = orig[orig["perturbation_type"] != C.NO_PERTURBATION]
    add("originals_have_perturbation_type_none", bad.empty, f"{len(bad)} originals with a perturbation type", len(bad))
    bad = df[~df["split"].isin(C.SPLITS)]
    add("allowed_split", bad.empty, f"invalid: {bad['split'].unique()[:5].tolist()}", len(bad))

    ids = set(orig["message_id"])
    bad = var[~var["original_message_id"].isin(ids)]
    add("original_message_id_refers_to_original", bad.empty, f"{len(bad)} variants point to a missing/non-original id", len(bad))

    o_idx = orig.drop_duplicates("message_id").set_index("message_id")   # robust to duplicated ids (reported above)
    m = var[var["original_message_id"].isin(ids)]
    lab_mismatch = int((m["original_message_id"].map(o_idx["label"]) != m["label"]).sum())
    add("variant_label_equals_original_label", lab_mismatch == 0, f"{lab_mismatch} variants with changed label", lab_mismatch)
    grp_mismatch = int((m["original_message_id"].map(o_idx["group_id"]) != m["group_id"]).sum())
    add("variant_group_equals_original_group", grp_mismatch == 0, f"{grp_mismatch} mismatches", grp_mismatch)
    same_text = int((m["message"].values == m["original_message_id"].map(o_idx["message"]).values).sum())
    add("variants_differ_from_original_text", same_text == 0, f"{same_text} identical to original", same_text, WARNING)

    n = int(orig["cleaned_message"].duplicated().sum())
    add("no_duplicate_original_messages", n == 0, f"{n} duplicated original messages", n)
    n_pd = int(var["cleaned_message"].isin(set(orig["cleaned_message"])).sum())
    add("perturbed_texts_not_identical_to_other_originals", n_pd == 0, f"{n_pd} perturbed texts equal some original", n_pd, WARNING)

    leak = check_group_leakage(df)
    add("no_group_in_multiple_splits", leak["groups_in_multiple_splits"] == 0,
        f"{leak['groups_in_multiple_splits']} groups, e.g. {leak['example_groups']}", leak["groups_in_multiple_splits"])
    add("variants_share_split_with_original", leak["variants_with_split_different_from_original"] == 0,
        f"{leak['variants_with_split_different_from_original']} variants", leak["variants_with_split_different_from_original"])
    tr, va, te = (set(orig.loc[orig["split"] == s, "cleaned_message"]) for s in C.SPLITS)
    n_overlap = len(tr & va) + len(tr & te) + len(va & te)
    add("no_exact_text_overlap_between_splits", n_overlap == 0, f"{n_overlap} shared messages", n_overlap)
    cs_mismatch = int((df["is_code_switched"].astype(bool) != (df["language"] == C.LANG_MIXED)).sum())
    add("code_switched_flag_matches_language", cs_mismatch == 0, f"{cs_mismatch} mismatches", cs_mismatch, WARNING)
    for s in C.SPLITS:
        sub = orig[orig["split"] == s]
        both = set(sub["label"]) == set(C.LABELS)
        add(f"split_{s}_has_both_classes", both, f"{s}: labels {sorted(set(sub['label']))}", int(len(sub)))

    stats = {
        "rows": int(len(df)), "originals": int(len(orig)), "variants": int(len(var)),
        "residual_similarity_test_vs_train": cross_split_similarity(df, "test", "train"),
        "residual_similarity_validation_vs_train": cross_split_similarity(df, "validation", "train"),
    }
    rs = stats["residual_similarity_test_vs_train"]
    if rs.get("n"):
        add("residual_near_duplicate_similarity_test_vs_train", True,
            f"mean max-sim={rs['mean_max_sim']:.3f}; share>=0.90: {rs['share_ge_0.90']:.3f}; share>=0.80: {rs['share_ge_0.80']:.3f}",
            int(rs["share_ge_0.90"] * rs["n"]), info=True)
    return checks, stats


def render_report(checks: list[Check], stats: dict) -> str:
    n_fail = sum(c.status == "FAIL" for c in checks)
    n_warn = sum(c.status == "WARN" for c in checks)
    lines = ["# Data validation report", "",
             f"Rows: {stats.get('rows')} (originals {stats.get('originals')}, variants {stats.get('variants')})", "",
             f"**Critical failures: {n_fail} | warnings: {n_warn} | checks: {len(checks)}**", "",
             "| Check | Status | Detail | Count |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {c.status} | {c.detail} | {c.count} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=C.PROCESSED_CSV)
    a = ap.parse_args(argv)
    df = load_processed(a.path)
    checks, stats = validate(df)
    report = render_report(checks, stats)
    ensure_dir(C.TABLES_DIR)
    (C.TABLES_DIR / "validation_report.md").write_text(report, encoding="utf-8")
    save_json({"checks": [asdict(c) for c in checks], "stats": stats}, C.TABLES_DIR / "validation_report.json")
    print(report)
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
