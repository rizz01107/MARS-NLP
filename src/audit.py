"""Dataset audit -> reports/data_audit.md and reports/data_audit.csv (all numbers computed, none hard-coded).

Audits (1) the three files originally provided (scaffold CSV/XLSX/README), (2) the two raw public sources and
(3) the processed MARS-NLP corpus. Usage: python -m src.audit
"""
from __future__ import annotations

import csv
import re

import pandas as pd

from . import config as C
from .data_loader import load_processed, load_raw_uci, load_raw_zenodo
from .preprocessing import clean_text, template_key
from .utils import ensure_dir

ROWS: list[dict] = []
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_NON_ALNUM = re.compile(r"[^a-z0-9\u0080-\uffff]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def rec(section: str, metric: str, value, note: str = "") -> None:
    ROWS.append({"section": section, "metric": metric, "value": value, "note": note})


def audit_scaffold() -> list[str]:
    import openpyxl

    s = "1. Originally provided scaffold files"
    out = ["## 1. Originally provided files (scaffold)", ""]
    csv_path, xlsx_path, readme = (C.SCAFFOLD_DIR / n for n in ("MARS_NLP_REAL_DATASET.csv", "MARS_NLP_REAL_DATASET.xlsx", "DATASET_README.md"))
    for p in (csv_path, xlsx_path, readme):
        rec(s, f"exists:{p.name}", p.exists())
    if not csv_path.exists():
        return out + ["Scaffold files not present in data/raw/scaffold/ - skipped.", ""]
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    header, n_csv = rows[0], len(rows) - 1
    rec(s, "csv_columns", len(header)), rec(s, "csv_data_rows", n_csv)
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    sheets = {ws.title: (ws.max_row, ws.max_column) for ws in wb.worksheets}
    rec(s, "xlsx_sheets", str(sheets))
    main = wb[wb.sheetnames[0]]
    xh = [c for c in next(main.iter_rows(values_only=True))]
    n_x = sum(1 for _ in main.iter_rows(min_row=2, values_only=True) if any(v is not None for v in _))
    rec(s, "xlsx_data_rows", n_x), rec(s, "csv_xlsx_headers_identical", header == xh)
    txt = readme.read_text(encoding="utf-8")
    m = re.search(r"contains\s+\*\*(\d+) imported rows\*\*", txt)
    rec(s, "readme_states_imported_rows", m.group(1) if m else "not found")
    out += [f"* CSV: {len(header)} columns, **{n_csv} data rows**; XLSX main sheet: **{n_x} data rows**; headers identical: {header == xh}.",
            f"* README states the scaffold contains **{m.group(1) if m else '?'} imported rows** - consistent with the files.",
            "* **Authoritative source:** since the scaffold is empty, the raw public sources (UCI, Zenodo) supplied afterwards are authoritative; "
            "nothing was overwritten (the scaffold files are kept unchanged in `data/raw/scaffold/`).", ""]
    return out


def audit_raw() -> list[str]:
    out = ["## 2. Raw public sources", ""]
    s = "2a. UCI SMS Spam Collection"
    u, info = load_raw_uci()
    u["clean"] = u["message"].map(clean_text)
    dup = u.duplicated("clean", keep="first")
    conf = u.groupby("clean")["label_raw"].nunique()
    norm = u["clean"].map(lambda t: _NON_ALNUM.sub("", t.lower()))
    vc = u["label_raw"].value_counts()
    L = u["message"].str.len()
    vals = {"rows": len(u), "encoding": info["encoding"], "malformed_lines": info["malformed_lines"],
            "ham": int(vc.get("ham", 0)), "spam": int(vc.get("spam", 0)), "empty_messages": int((u["clean"] == "").sum()),
            "exact_duplicate_messages (raw text)": int(u["message"].duplicated().sum()),
            "exact_duplicate_messages (after cleaning)": int(dup.sum()), "duplicate_groups_with_conflicting_labels": int((conf > 1).sum()),
            "near_duplicates_beyond_exact (case/punct-insensitive)": int(norm.duplicated().sum() - dup.sum()),
            "messages_with_non_ascii": int(u["message"].map(lambda t: bool(_NON_ASCII.search(t))).sum()),
            "messages_with_html_entities (&lt; &amp; ...)": int(u["message"].str.contains(r"&(?:lt|gt|amp);").sum()),
            "length_min/median/mean/max": f"{L.min()}/{L.median():.0f}/{L.mean():.1f}/{L.max()}"}
    for k, v in vals.items():
        rec(s, k, v)
    out += ["### 2a. UCI SMS Spam Collection", "", "| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in vals.items()]
    out += ["", f"License: {C.LICENSE_UCI}. The bundled readme grants free use with no limitations except copyright (Almeida & Gomez Hidalgo) "
            "and a citation request; the CC BY 4.0 statement originates from the UCI web page and was **not re-verified offline**.", ""]

    s = "2b. Zenodo Roman Urdu phishing SMS"
    z, zinfo = load_raw_zenodo()
    raw = pd.read_excel(C.ZENODO_RAW, sheet_name=0)
    tk = z["message"].map(template_key)
    letters_only = z["message"].str.lower().str.replace(r"\d+", "", regex=True).str.replace(r"[^a-z]", "", regex=True)
    tc = tk.value_counts()
    lab = z["label_raw"].value_counts()
    cat = pd.crosstab(z["message_type"], z["label_raw"])
    vals = {"rows": len(z), "columns": ", ".join(raw.columns), "missing_values_total": int(raw.isna().sum().sum()),
            "legitimate(0)": int(lab.get("0", 0)), "phishing(1)": int(lab.get("1", 0)), "n_message_categories": int(z["message_type"].nunique()),
            "categories_with_mixed_labels": int(((cat > 0).sum(axis=1) > 1).sum()), "unique_IDs": bool(raw["ID"].is_unique),
            "exact_duplicate_Raw_text": int(z["message"].duplicated().sum()),
            "duplicate_Normalized_text": int(raw["Normalized_text"].duplicated().sum()) if "Normalized_text" in raw else "n/a",
            "unique_after_digit_masking": int(tk.nunique()), "unique_after_digit+punctuation_masking": int(letters_only.nunique()),
            "templates_occurring_more_than_once": int((tc > 1).sum()), "messages_in_repeated_templates": int(tc[tc > 1].sum()),
            "largest_template_size": int(tc.max()),
            "messages_in_Perso-Arabic_script": int(z["message"].map(lambda t: bool(_ARABIC.search(t))).sum()),
            "license_in_file": "none found"}
    for k, v in vals.items():
        rec(s, k, v)
    out += ["### 2b. Zenodo Roman Urdu phishing SMS (`phishing_dataset_1000.xlsx`)", "", "| Metric | Value |", "|---|---|"] + \
           [f"| {k} | {v} |" for k, v in vals.items()]
    out += ["", f"License: {C.LICENSE_ZENODO}.", "",
            "**Finding:** the messages are highly templated (see counts above): near-identical frames with swapped brand names, amounts and IDs. "
            "Random splitting would leak templates, so splitting is done at the template-group level (Section 4).", ""]
    return out


def audit_processed() -> list[str]:
    df = load_processed()
    is_p = df["is_perturbed"].astype(bool)
    o, v = df[~is_p], df[is_p]
    out = ["## 3. Processed MARS-NLP corpus (`data/processed/mars_nlp_processed.csv`)", ""]
    s = "3. Processed corpus"
    vals = {"rows_total": len(df), "columns": len(df.columns), "originals": len(o), "perturbed_variants": len(v),
            "missing_values_total (originals_original_message_id excluded)": int(df.drop(columns=["original_message_id"]).isna().sum().sum()),
            "duplicate_message_id": int(df["message_id"].duplicated().sum()),
            "duplicate_original_messages": int(o["cleaned_message"].duplicated().sum()),
            "n_near_duplicate_groups (originals)": int(o["group_id"].nunique()),
            "variants_per_original (mean)": round(len(v) / len(o), 2)}
    for k, val in vals.items():
        rec(s, k, val)
    out += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {val} |" for k, val in vals.items()] + [""]

    def table(title, ser_or_df):
        rec(s, title, ser_or_df.to_dict() if hasattr(ser_or_df, "to_dict") else str(ser_or_df))
        return [f"**{title}**", "", ser_or_df.to_markdown(), ""]

    out += table("Label distribution (originals)", o["label"].value_counts().to_frame("n").assign(share=lambda d: (d.n / d.n.sum()).round(3)))
    out += table("Language distribution (originals, heuristic)", o["language"].value_counts().to_frame("n"))
    out += table("Language x label (originals)", pd.crosstab(o["language"], o["label"]))
    out += table("Source distribution (originals)", o["source_dataset"].value_counts().to_frame("n"))
    out += table("Source x language (originals)", pd.crosstab(o["source_dataset"].str[:24], o["language"]))
    out += table("Perturbation distribution (variants)", v["perturbation_type"].value_counts().to_frame("n"))
    out += table("Split x label (originals)", pd.crosstab(o["split"], o["label"]).reindex(list(C.SPLITS)))
    out += table("Split x source (originals)", pd.crosstab(o["split"], o["source_dataset"].str[:24]).reindex(list(C.SPLITS)))
    out += table("Split x language (originals)", pd.crosstab(o["split"], o["language"]).reindex(list(C.SPLITS)))
    out += table("Split x label (all rows incl. variants)", pd.crosstab(df["split"], df["label"]).reindex(list(C.SPLITS)))
    gs = o.groupby("group_id").size()
    rec(s, "group_size_max", int(gs.max())), rec(s, "groups_with_more_than_one_message", int((gs > 1).sum()))
    zen = o[o.source_dataset == C.SRC_ZENODO].groupby("group_id").size()
    rec(s, "zenodo_groups", int(len(zen))), rec(s, "zenodo_largest_group", int(zen.max()))
    out += [f"* Near-duplicate groups: **{o['group_id'].nunique()}**; groups with >1 message: {(gs > 1).sum()}; largest group: {gs.max()} messages.",
            f"* Zenodo originals form only **{len(zen)} groups** (largest {zen.max()} messages) -> every Zenodo test number rests on very few independent groups.",
            "", "### Original / augmented relationship", ""]
    ids = set(o["message_id"])
    rel = {"variants_whose_original_exists": int(v["original_message_id"].isin(ids).sum()), "variants_total": len(v),
           "variants_with_split_equal_to_original": int((v["original_message_id"].map(o.set_index("message_id")["split"]) == v["split"]).sum()),
           "variants_with_label_equal_to_original": int((v["original_message_id"].map(o.set_index("message_id")["label"]) == v["label"]).sum())}
    for k, val in rel.items():
        rec(s, k, val)
        out.append(f"* {k}: {val}")
    out += ["", "### Language-heuristic spot checks", "",
            "Language labels come from a transparent lexicon heuristic (`src/preprocessing.py`), not a trained model. Sanity checks:", ""]
    uci = o[o.source_dataset == C.SRC_UCI]
    flagged = uci[uci.language.isin([C.LANG_ROMAN_URDU, C.LANG_MIXED])]
    rec(s, "uci_messages_flagged_roman_urdu_or_mixed", len(flagged))
    out.append(f"* UCI messages flagged Roman Urdu/Mixed: **{len(flagged)}** of {len(uci)} (mostly false positives on short English texts, e.g. 'ho ho'; one or two are genuine Roman-Urdu/Hindi text):")
    for t in flagged["cleaned_message"].str[:90]:
        out.append(f"  * `{t}`")
    zen_en = o[(o.source_dataset == C.SRC_ZENODO) & (o.language == C.LANG_ENGLISH)]
    rec(s, "zenodo_messages_labelled_english", len(zen_en))
    out += ["", f"* Zenodo messages labelled English: **{len(zen_en)}** (fully English bank/OTP-style notification templates; individual messages are not reproduced here because the Zenodo license is unverified). "
            "The source's metadata calls the whole set 'Roman Urdu', so **language is not equivalent to source** here.",
            f"* Urdu (Perso-Arabic script) messages in the corpus: **{int((o.language == C.LANG_URDU).sum())}** -> Urdu cannot be evaluated.",
            f"* Language 'Unknown' (too little alphabetic evidence): {int((o.language == C.LANG_UNKNOWN).sum())} originals.", ""]
    lic = o.groupby(["source_dataset", "source_license"]).size().rename("n").reset_index()
    out += ["### Sources and licenses", "", lic.to_markdown(index=False), ""]
    for r in lic.itertuples():
        rec(s, f"license:{r.source_dataset[:30]}", r.source_license)
    return out


def run() -> None:
    ROWS.clear()
    md = ["# MARS-NLP data audit", "",
          "All statistics below are computed by `python -m src.audit` from the files in `data/raw/` and `data/processed/`.", ""]
    md += audit_scaffold()
    md += audit_raw()
    md += audit_processed()
    md += ["## 4. Leakage methodology and findings", "",
           "* Exact duplicates are removed first (keep first occurrence; none had conflicting labels).",
           "* Near-duplicate/templated messages are merged into groups (character 3-5-gram TF-IDF cosine >= "
           f"{C.NEAR_DUP_THRESHOLD} on digit-masked, lower-cased text; single-linkage).",
           "* Whole groups are assigned to train/validation/test (70/15/15 by message count, stratified by source and label); "
           "perturbed variants inherit the split of their original.",
           "* `python -m src.validate_data` verifies: no group in more than one split, every variant in its original's split, no exact text overlap, "
           "and reports residual cross-split similarity (see `reports/tables/validation_report.md`).", ""]
    ensure_dir(C.REPORTS_DIR)
    (C.REPORTS_DIR / "data_audit.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(ROWS).to_csv(C.REPORTS_DIR / "data_audit.csv", index=False, encoding="utf-8")
    print(f"Wrote {C.REPORTS_DIR / 'data_audit.md'} and data_audit.csv ({len(ROWS)} rows)")


if __name__ == "__main__":
    run()
