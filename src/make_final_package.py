"""Build the publication-ready package from SAVED results only (no training, no dataset rebuild, no metric changes).

Writes: results/{FINAL_RESULTS,ROBUSTNESS_RESULTS,STATISTICAL_COMPARISONS,ERROR_COUNTS}.csv, predictions/, figures/ (via
make_publication_figures), reports/{FINAL_REPRODUCIBILITY_REPORT,QA_REPORT}.md, SUBMISSION_CHECKLIST.md, RELEASE_MANIFEST.md and
(optionally) MARS-NLP-release.zip.   Usage: python -m src.make_final_package [--zip]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile

import pandas as pd

from . import config as C
from . import final_tables as F
from .make_report import PENDING_NOTE, REPRO_NOTE, TITLE, pending_table_md
from .utils import sha256_file
from .verified import SOURCE_LABEL, load_supplement

ROOT = C.ROOT_DIR
R, T = C.RESULTS_DIR, C.TABLES_DIR
AUTHOR = ("**Author:** Muhammad Rizwan  \n**Affiliation:** Independent Researcher  \n"
          "**Email:** rizwan01107@gmail.com")
RELEASE_ZIP = ROOT / "MARS-NLP-release.zip"


def md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


# --------------------------------------------------------------------------- CSVs + predictions
def write_results() -> None:
    F.FINAL_DIR.mkdir(exist_ok=True)
    F.final_results_df().to_csv(F.FINAL_DIR / "FINAL_RESULTS.csv", index=False)
    F.robustness_df().to_csv(F.FINAL_DIR / "ROBUSTNESS_RESULTS.csv", index=False)
    F.statistical_df().to_csv(F.FINAL_DIR / "STATISTICAL_COMPARISONS.csv", index=False)
    F.error_counts_df().to_csv(F.FINAL_DIR / "ERROR_COUNTS.csv", index=False)
    (F.FINAL_DIR / "README.md").write_text(
        "# results/ - curated final tables\n\n"
        "| File | Content | Source of values |\n|---|---|---|\n"
        "| `FINAL_RESULTS.csv` | 8-model comparison (test macro-F1, accuracy, precision, recall, weighted-F1, 95% CI) | baselines: `experiments/results/baseline_results.csv`; transformers: raw run JSON if present, else `verified_transformer_results.json` |\n"
        "| `ROBUSTNESS_RESULTS.csv` | degradation = original macro-F1 - perturbed macro-F1 (mean and per perturbation type) | `experiments/results/robustness_results.csv`; transformer means from the verified summary |\n"
        "| `STATISTICAL_COMPARISONS.csv` | paired cluster-bootstrap macro-F1 differences | `verified_transformer_results.json` (owner-verified) |\n"
        "| `ERROR_COUNTS.csv` | per-model error counts on test originals + variants (never aggregated across models) | `experiments/results/error_analysis.csv`; transformer counts from the verified summary |\n"
        "| `verified_transformer_results.json` | owner-supplied verified summary of the Colab Tesla T4 run and QA | provenance stated inside the file |\n\n"
        "`experiments/results/final_results.csv` is the longer machine-generated table (12 baselines incl. augmentation-trained variants + transformers).\n", encoding="utf-8")
    (ROOT / "predictions").mkdir(exist_ok=True)
    shutil.copy2(R / "baseline_predictions.csv", ROOT / "predictions" / "baseline_predictions.csv")
    for f in R.glob("transformer_predictions_*.csv"):
        if "_smoke" not in f.name:
            shutil.copy2(f, ROOT / "predictions" / f.name)
    (ROOT / "predictions" / "README.md").write_text(
        "# predictions/\n\n* `baseline_predictions.csv` - test-original predictions of the 12 CPU baselines (`model_id,message_id,y_true,y_pred,p_scam`); used by the QA suite to recompute every baseline metric.\n"
        "* `transformer_predictions_<tag>.csv` (`mbert`, `xlmr`) - produced by the Colab run for test originals and their perturbed variants. "
        "Copy them here (and into `experiments/results/`) to let `python -m src.qa_check` recompute the transformer metrics; "
        "they are " + ("present." if list((ROOT / "predictions").glob("transformer_predictions_*.csv")) else "**not present in this snapshot**.") + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- reports
def _qa_snapshot() -> dict | None:
    f = T / "qa_report.json"
    return json.loads(f.read_text()) if f.exists() else None


EVIDENCE_NOTE = ("**Evidence check for the raw Colab transformer artifacts (2026-10-08):** the whole workspace was searched for `transformer_*`, mBERT/XLM-R prediction, "
                 "metrics (JSON/CSV) and environment files (`/mnt/user-data/uploads`, `/mnt/user-data/outputs`, `/mnt/transcripts`, the repository, `/home`, `/tmp`). "
                 "**No original Colab outputs were found.** The only matches were temporary pytest fixtures (synthetic test data and a tiny randomly initialised smoke model), "
                 "which are not results and were not used. The raw Colab transformer artifacts were therefore unavailable for independent local verification; nothing was reconstructed "
                 "or invented, and the owner-supplied values in `results/verified_transformer_results.json` are kept unchanged.")


def qa_report() -> str:
    sup = load_supplement()
    qa = sup["qa_owner_verified"]
    snap = _qa_snapshot()
    L = [f"# QA report\n\n{AUTHOR}\n", "## 1. Owner-verified QA (Colab, after both transformers finished)", "",
         f"**{qa['PASS']} PASS, {qa['FAIL']} FAIL, {qa['PENDING']} PENDING, {qa['SKIP']} SKIP** - {qa['pytest_passed']} pytest tests passed.", "",
         *[f"* {c}" for c in qa["checks"]], "",
         f"Provenance: {sup['provenance']}", "", "## 2. Reproducibility note and the 3 PENDING local checks", "", REPRO_NOTE, "", EVIDENCE_NOTE, "", pending_table_md(), "", PENDING_NOTE, "",
         "## 3. Snapshot-local QA re-run (this repository copy)", ""]
    if snap:
        s = snap["summary"]
        L += [f"`python -m src.qa_check` at {snap['generated']}: PASS={s['PASS']} FAIL={s['FAIL']} PENDING={s['PENDING']} SKIP={s['SKIP']}.", "",
              "The 3 PENDING items are listed in Section 2; they stay PENDING until the raw Colab artifacts are copied into this folder "
              "(`experiments/results/transformer_*`, `predictions/transformer_predictions_*.csv`) and `python -m src.qa_check` is re-run.", "", "| Check | Status | Detail |", "|---|---|---|"]
        L += [f"| {c['check']} | {c['status']} | {str(c['detail'])[:140]} |" for c in snap["checks"] if c.get("group", "core") == "core"]
        dc = snap.get("documentation_checks_summary", {})
        L += ["", f"Documentation-wording checks (reported separately; not part of the {s['PASS']}/{s['FAIL']}/{s['PENDING']}/{s['SKIP']} count above): PASS={dc.get('PASS')} FAIL={dc.get('FAIL')}.", "",
              "| Check | Status |", "|---|---|"]
        L += [f"| {c['check']} | {c['status']} |" for c in snap["checks"] if c.get("group") == "documentation"]
    else:
        L += ["No snapshot-local QA report found; run `python -m src.qa_check`."]
    return "\n".join(L) + "\n"


def reproducibility_report() -> str:
    sup = load_supplement()
    b = json.loads((T / "build_summary.json").read_text())
    val = json.loads((T / "validation_report.json").read_text())
    cfg = json.loads((R / "baseline_config.json").read_text())
    fp = json.loads((C.PROCESSED_DIR / "split_fingerprint.json").read_text())
    sha_file = (C.PROCESSED_DIR / "mars_nlp_processed.sha256").read_text().split()[0]
    sha_now = sha256_file(C.PROCESSED_CSV)
    fin = F.final_results_df()
    rob = F.robustness_df()
    base = pd.read_csv(R / "baseline_results.csv")
    ov = base[(base.subset == "Overall") & (~base.aug_trained)].set_index("model_id")
    sel = json.loads((C.MODELS_DIR / "best_model.json").read_text())
    snap = _qa_snapshot()
    tv = {t: F.tx_values(t) for t in ("mbert", "xlmr")}
    pr = sup["training_protocol"]
    d0 = pr["defaults_of_the_documented_command"]
    leak = pd.DataFrame([{"Check": c["name"], "Status": c["status"], "Detail": str(c["detail"])[:110]} for c in val["checks"]
                         if any(k in c["name"] for k in ("leak", "group", "overlap", "split", "unique", "duplicate", "variant", "residual"))])
    rs = val["stats"]["residual_similarity_test_vs_train"]
    models = []
    for mid, m in cfg["models"].items():
        if mid.endswith("_aug"):
            continue
        models.append({"Model": m["name"], "Features": m["features"], "Classifier": m["model"], f"Best {('C/alpha')}": m["best_param"], "Class weight": "balanced" if m["model"] != "NaiveBayes" else "-",
                       "Selected on": "validation macro-F1"})
    tx_tab = pd.DataFrame([{"Model": sup["transformers"][t]["name"], "Train n": v["n_train"], "GPU": v["gpu"], "Best epoch": v["best_epoch"], "Train time (s)": v["train_seconds"],
                            "Params": v["params"], "Size (MB)": v["size_mb"], "GPU ms/msg": v["gpu_ms"], "CPU ms/msg": v["cpu_ms"]} for t, v in tv.items()])
    eff = pd.DataFrame([{"Model": mid, "Params (classifier weights)": int(ov.loc[mid].parameters), "Size (MB)": float(ov.loc[mid].model_size_mb),
                         "Train (s)": float(ov.loc[mid].train_seconds), "CPU single-msg mean (ms)": float(ov.loc[mid].latency_single_ms_mean)} for mid, _ in F.BASELINE_ORDER])
    eff["Model"] = [n for _, n in F.BASELINE_ORDER]
    rob_mean = rob[rob.Level == "mean over 10 perturbation types"][["Model", "Degradation"]]
    svm_types = rob[(rob.Model == "Char TF-IDF + Linear SVM") & (rob.Level.str.startswith("per"))][["Perturbation_Type", "Degradation", "CI95_low", "CI95_high", "n_pairs", "n_groups"]]
    stat = F.statistical_df()[["Comparison", "Difference_A_minus_B", "CI95_low", "CI95_high", "CI_includes_zero"]]
    errs = F.error_counts_df()[["Model", "Errors", "False_Positives", "False_Negatives"]]
    pert = pd.DataFrame([{"Perturbation": k, "Variants": v} for k, v in b["perturbation_counts"].items()])
    src = pd.DataFrame([
        ["Dataset sizes, split sizes, perturbation counts", "`reports/tables/build_summary.json`; recomputed from `data/processed/mars_nlp_processed.csv` by `qa_check`", "recomputed"],
        ["Test groups (778), English 782, Roman Urdu + Mixed 149 / 5 groups", "`data/processed/mars_nlp_processed.csv`; `reports/tables/eda_group_structure.csv`", "recomputed"],
        ["Leakage checks, residual similarity", "`reports/tables/validation_report.json` (`python -m src.validate_data`)", "recomputed"],
        ["Checksum, split fingerprints", "`data/processed/mars_nlp_processed.sha256`, `data/processed/split_fingerprint.json`", "recomputed"],
        ["Baseline metrics (6 models) and CIs", "`experiments/results/baseline_results.csv`, `baseline_predictions.csv`", "recomputed from raw predictions by `qa_check`; equal to the owner table"],
        ["Baseline configurations", "`experiments/results/baseline_config.json`, `experiments/configs/baselines.yaml`", "file"],
        ["Baseline robustness (per type, mean)", "`experiments/results/robustness_results.csv`", "file; Char SVM mean 0.0029 verified"],
        ["Error counts: 12 baselines", "`experiments/results/error_analysis.csv`", "recomputed; equal to the verified counts"],
        ["Transformer metrics, CIs, training time, GPU, latency, parameters, size", f"`results/verified_transformer_results.json` ({SOURCE_LABEL}); raw: `experiments/results/transformer_<tag>.json`", "owner-supplied (raw files absent here)"],
        ["Transformer robustness means (-0.0055, 0.0339), error counts (614, 264)", "`results/verified_transformer_results.json`; raw: `transformer_robustness_<tag>.csv`, `transformer_predictions_<tag>.csv`", "owner-supplied"],
        ["Statistical comparisons (3)", "`results/verified_transformer_results.json`; raw: paired bootstrap on `transformer_predictions_<tag>.csv`", "owner-supplied"],
        ["Owner QA (47 PASS / 0 FAIL / 0 PENDING / 0 SKIP; 63 tests)", "Colab `qa_check` run (`reports/tables/qa_report.json` there)", "owner-supplied"]], columns=["Result", "Source file(s)", "Verification in this snapshot"])
    L = [f"# MARS-NLP final reproducibility report\n\n# {TITLE}\n\n{AUTHOR}\n", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `python -m src.make_final_package` from saved results only (no training or dataset rebuild).", "",
         "## 1. Dataset", "", f"* Original messages: **{b['n_originals']:,}** (UCI SMS Spam Collection + Zenodo Roman-Urdu phishing SMS, after exact-duplicate removal)",
         f"* Synthetic perturbed variants: **{b['n_perturbed']:,}** (derived only from real originals; `source_type = augmented_from_real`)",
         "* No Urdu-script messages; no Urdu-script benchmark is available.", "", "## 2. Train / validation / test split", "",
         f"* Originals: train **{b['split_counts_originals']['train']:,}**, validation **{b['split_counts_originals']['validation']}**, test **{b['split_counts_originals']['test']}** (group-level 70/15/15, stratified by source and label, seed {C.SEED}).",
         "* Test message groups: **778**. Test languages: English **782**; Roman Urdu + Mixed **149**, which come from only **5 independent test groups** and are a stated limitation (descriptive only).", "",
         "## 3. Perturbation types", "", "Spelling variation, character deletion, character substitution, word spacing variation, punctuation variation, casing variation, informal abbreviation, Roman Urdu spelling variation, code switching, minor noise.", "",
         md(pert, "d"), "", "## 4. Data-leakage checks", "", md(leak, ".0f"), "",
         f"Residual similarity of test originals to their most similar training original (char 3-5-gram TF-IDF cosine): mean {rs['mean_max_sim']:.3f}; share >= 0.80: {rs['share_ge_0.80']:.3f}; share >= 0.90: {rs['share_ge_0.90']:.3f}.", "",
         "## 5. Checksum verification", "", f"* Expected SHA-256 (`data/processed/mars_nlp_processed.sha256`): `{sha_file}`", f"* Recomputed at generation time: `{sha_now}` -> **{'MATCH' if sha_now == sha_file else 'MISMATCH'}**",
         "* Every transformer training run verifies this checksum before training and aborts on mismatch.", "", "## 6. Split fingerprints (SHA-256 of sorted original message ids per split)", "",
         md(pd.DataFrame([{"Split": k, "n": v["n"], "sha256": v["sha256"]} for k, v in fp.items()]), "d"), "", "## 7. Model configurations", "",
         "**CPU baselines** (features: `word` = word 1-2-grams, `char` = char_wb 2-5-grams; TF-IDF sublinear, min_df=2, fit on the training split only; LR/SVM class_weight=balanced):", "", md(pd.DataFrame(models), ".4g"), "",
         f"Grids: LR C in {{0.1, 1, 10, 100}}; SVM C in {{0.01, 0.1, 1, 10}}; NB alpha in {{0.01, 0.1, 0.5, 1.0}} (small grids; not exhaustive). The validation-selected baseline is **{sel['name']}** (validation macro-F1 {sel['val_macro_f1']:.4f}).", "",
         f"**Transformers:** mBERT `bert-base-multilingual-cased`, XLM-R `xlm-roberta-base`; at most {pr['max_epochs']} epochs; early stopping on validation macro-F1, patience {pr['patience']}; batch size {pr['batch_size']} (OOM fallback: {pr['oom_fallback']}); "
         f"documented-command defaults lr {d0['learning_rate']}, max length {d0['max_length']}, weight decay {d0['weight_decay']}, warm-up {d0['warmup_ratio']}, seed {d0['seed']}, {d0['loss']}, {d0['optimizer']}.", "",
         "## 8. Final metrics (test, 935 original messages; Precision/Recall = Scam class)", "",
         md(fin[["Model", "Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "Macro_F1_CI95", "Hardware"]]), "",
         "* **Char TF-IDF + Logistic Regression achieved the highest observed test macro-F1 (0.9762).**", "* **Char TF-IDF + Linear SVM was the validation-selected CPU baseline and achieved 0.9741 test macro-F1.**",
         "* **XLM-R was the best-performing transformer with 0.9722 macro-F1.** XLM-R is not claimed to be the best overall model, nor to significantly outperform the selected SVM.", "",
         "## 9. Transformer training details and GPU information", "", md(tx_tab, ".4g"), "",
         "* GPU: **NVIDIA Tesla T4** (Google Colab) for both models. CUDA / PyTorch / transformers versions are recorded in the raw run JSON (`environment`), which is not in this snapshot and is therefore not restated.",
         f"* Provenance of these values: `results/verified_transformer_results.json` ({SOURCE_LABEL}).", "", "## 10. Inference benchmarks (computational benchmarks, not edge-device measurements)", "",
         "CPU baselines (one CPU core, sandbox machine):", "", md(eff, ".4g"), "", "Transformers (single-message latency on the Colab Tesla T4 and on the Colab CPU) are in the table of Section 9. Measurements depend on the hardware/software environment.", "",
         "## 11. Robustness results", "", "`degradation = original macro-F1 - perturbed macro-F1` (mean over 10 synthetic perturbation types, paired test messages; lower is better). Robustness to the project's synthetic perturbation suite - not real-world adversarial robustness.", "",
         md(rob_mean), "", "Per-type values for the selected Char SVM (paired; cluster-bootstrap 95% CI):", "", md(svm_types), "",
         "Per-type values for the transformers require the raw robustness CSVs (not in this snapshot).", "", "## 12. Statistical comparisons (paired cluster bootstrap over near-duplicate test groups)", "", md(stat, ".6f"), "",
         "* XLM-R vs the selected Char SVM: CI includes zero -> no claim of superiority.", "* XLM-R vs mBERT: CI excludes zero (lower bound 0.0017, close to zero).", "* English-only XLM-R vs mBERT: CI includes zero -> no claim of English-only superiority.", "",
         "## 13. Error analysis (counts per model over test originals + variants; never aggregated across models)", "", md(errs, ".0f"), "",
         "## 14. QA results", "", f"* Owner-verified (Colab): **{sup['qa_owner_verified']['PASS']} PASS, {sup['qa_owner_verified']['FAIL']} FAIL, {sup['qa_owner_verified']['PENDING']} PENDING, {sup['qa_owner_verified']['SKIP']} SKIP**; {sup['qa_owner_verified']['pytest_passed']} pytest tests passed.",
         *[f"  * {c}" for c in sup["qa_owner_verified"]["checks"]],
         (f"* Snapshot-local QA of the release package: **{snap['summary']['PASS']} PASS, {snap['summary']['FAIL']} FAIL, {snap['summary']['PENDING']} PENDING, {snap['summary']['SKIP']} SKIP**. See `reports/QA_REPORT.md`." if snap else "* Snapshot-local QA not yet run."), "",
         "The 3 PENDING local checks:", "", pending_table_md(), "", PENDING_NOTE, "", REPRO_NOTE, "",
         "## 15. Known limitations", "",
         "* The dataset is domain-specific (English SMS marketing spam; Pakistani bank/utility-style Roman-Urdu phishing templates).", "* The perturbations are synthetic and are not equivalent to naturally occurring adversarial messages.",
         "* Roman Urdu/Mixed evidence is limited: only **5 independent test groups** support that analysis (descriptive only).", "* There is no comprehensive Urdu-script benchmark.",
         "* Multi-seed variance was not estimated for every model (each transformer was trained once).", "* The hyper-parameter search was not exhaustive.",
         "* Computational measurements depend on the Tesla T4 and the software environment.", "* Transformer interpretability methods such as SHAP/Integrated Gradients were not performed.",
         "* Language labels are heuristic; source, language and label semantics are confounded; the Zenodo license is unverified.",
         "* The mBERT and XLM-R metrics are owner-verified results of the original Colab run; the raw transformer artifacts were unavailable locally, so they have not been independently reproduced here (3 local QA checks PENDING, not failed).", "",
         "## 16. Exact source files for each result", "", md(src, ".0f"), "",
         "## 17. How to re-verify", "", EVIDENCE_NOTE, "", "Copy the Colab files `experiments/results/transformer_{mbert,xlmr}.json`, `transformer_predictions_*.csv`, `transformer_subsets_*.csv`, `transformer_robustness_*.csv` (and optionally the logs) into this repository, then run "
         "`python -m src.qa_check && python -m src.make_final_package`. The code prefers raw run files over the owner-verified summary and recomputes every metric from the raw predictions.", ""]
    return "\n".join(L)


CHECKLIST_ITEMS = [
    ("Final title", True, "exact title in the manuscript, README, reproducibility report (checked by `qa_check`)"),
    ("Author information", True, "name, affiliation, email in the manuscript and main documents (checked by `qa_check`)"),
    ("Abstract", True, "`reports/MARS_NLP_Final_Manuscript.md` - Abstract and Keywords"), ("Introduction", True, "manuscript section 1"), ("Related Work", True, "manuscript section 2; 13 references, all cited"),
    ("Methodology", True, "manuscript section 3 (dataset, splitting, perturbations, TF-IDF, classical models, mBERT, XLM-R)"), ("Dataset description", True, "manuscript sections 3.1-3.2; `reports/data_audit.md`"),
    ("Experimental setup", True, "manuscript section 4; `experiments/configs/`"), ("Results", True, "manuscript section 5; `results/FINAL_RESULTS.csv`"),
    ("Robustness analysis", True, "manuscript section 5.4; `results/ROBUSTNESS_RESULTS.csv`"), ("Statistical analysis", True, "manuscript section 5.3; `results/STATISTICAL_COMPARISONS.csv`"),
    ("Error analysis", True, "manuscript section 5.5; `results/ERROR_COUNTS.csv` (per model, not aggregated)"), ("Limitations", True, "manuscript section 7 (all required limitations; checked by `qa_check`)"),
    ("Conclusion", True, "manuscript section 9"), ("AI-assisted tools disclosure", True, "manuscript section 10 and README disclose ChatGPT and Claude; author responsibility stated"), ("References", True, "manuscript section 11 (core: Devlin, Conneau, Pedregosa; all cited)"), ("Tables", True, "manuscript Tables 1-9; `results/*.csv`"),
    ("Figures", True, "`figures/` (300 dpi PNG + PDF), embedded as Figures 1-3 in the manuscript"), ("Reproducibility information", True, "manuscript section 8; `reports/FINAL_REPRODUCIBILITY_REPORT.md`; `docs/COLAB_GPU.md`; `data/README.md`"),
    ("Code", True, "`src/`, `tests/`, `app/`, `notebooks/`"), ("Requirements", True, "`requirements.txt`, `requirements-colab.txt`, `environment.yml`"), ("README", True, "professor-facing README with all required sections"),
    ("QA report", True, "`reports/QA_REPORT.md`: owner-verified Colab QA 47 PASS / 0 FAIL / 0 PENDING / 0 SKIP; local release-package QA 50 PASS / 0 FAIL / 3 PENDING / 0 SKIP"),
    ("No contradictory metrics", True, "`qa_check`: manuscript and results tables equal `results/*.csv` and the owner-verified table"),
    ("No unsupported claims", True, "automated wording checks (no 'best overall', 'significantly outperforms' or leading-performance claim); the author must still proofread"),
    ("PDF manuscript", True, "`reports/MARS_NLP_Final_Manuscript.pdf` (`scripts/build_manuscript_pdf.sh`)")]
OPEN_ITEMS = ["Raw Colab transformer artifacts (mBERT, XLM-R) were unavailable in the final local workspace: the 3 PENDING local QA checks (missing raw mBERT artifacts, missing raw XLM-R artifacts, no-N/A verification for executed transformer runs) stay PENDING - not failures - until the artifacts are supplied. Copy the Colab raw files (`experiments/results/transformer_*`, `predictions/transformer_predictions_*.csv`, logs) into the repository and run `python -m src.qa_check` so the transformer values are recomputed locally (currently owner-verified summary only).",
              "Confirm the Zenodo dataset license (unverified) and the UCI CC BY 4.0 statement against the live records before redistributing any data.",
              "Regenerate the PDF after manuscript source changes and compare it against the Markdown manuscript.",\n              "Proofread the generated report text; adapt formatting to the target venue.",\n              "Check the target journal’s AI-use disclosure policy and adjust wording before submission.",
              "Optional: multi-seed transformer runs and per-language transformer subsets (needs raw files / new GPU runs)."]


def checklist() -> str:
    L = [f"# Submission checklist\n\n# {TITLE}\n\n{AUTHOR}\n", "Final manuscript: `reports/MARS_NLP_Final_Manuscript.md` (PDF: `reports/MARS_NLP_Final_Manuscript.pdf`). It is unpublished and has not been peer reviewed.", "", "Checked items were verified from files or by `python -m src.qa_check`; evidence in the right column.", "", "| Done | Item | Evidence |", "|---|---|---|"]
    L += [f"| {'[x]' if ok else '[ ]'} | {name} | {ev} |" for name, ok, ev in CHECKLIST_ITEMS]
    L += ["", "## Verification status of the transformer results", "", REPRO_NOTE, "", pending_table_md(), "", PENDING_NOTE,
          "", "## Open items before submission (not done by the automated cleanup)", ""] + [f"- [ ] {x}" for x in OPEN_ITEMS] + [""]
    return "\n".join(L)


# --------------------------------------------------------------------------- manifest + release
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".git", ".venv", "_smoke"}
NOT_IN_RELEASE = ("data/raw/uci/", "data/raw/zenodo/", "models/transformers/", "experiments/results/error_analysis.csv", "reports/tables/explain_examples.csv")
PURPOSES = [("data/raw/uci/", "raw UCI SMS Spam Collection (third-party)"), ("data/raw/zenodo/", "raw Zenodo Roman-Urdu phishing SMS (third-party; license unverified)"),
            ("data/raw/scaffold/", "originally supplied empty scaffold files (audit trail)"), ("data/processed/mars_nlp_processed", "processed corpus (rebuild with `src.build_dataset`)"),
            ("data/processed/", "checksum / split fingerprint guarding the audited split"), ("data/", "data documentation"), ("models/baseline_", "saved CPU baseline artefact (scikit-learn 1.8.0 pickle)"),
            ("models/best_model.json", "validation-selected baseline metadata"), ("models/transformers/", "fine-tuned transformer checkpoint (large)"), ("predictions/", "raw predictions"),
            ("results/", "curated final results"), ("experiments/results/", "raw experiment outputs (audit evidence)"), ("experiments/configs/", "experiment configuration"),
            ("reports/archive/", "superseded reports kept for audit"), ("reports/figures/", "analysis figures"), ("reports/tables/", "generated tables / QA / validation reports"),
            ("reports/", "reports"), ("figures/", "publication figures"), ("notebooks/", "executed analysis notebooks"), ("src/", "source code"), ("tests/", "pytest suite"),
            ("app/", "Streamlit demo"), ("docs/", "documentation"), ("scripts/", "run scripts")]
REQUIRED_REPRO = ("src/", "tests/", "app/", "requirements", "environment.yml", "data/processed/", "data/README.md", "experiments/", "scripts/", "results/verified", "models/baseline_",
                  "models/best_model.json", "docs/COLAB_GPU.md")


def purpose(rel: str) -> str:
    for pre, text in PURPOSES:
        if rel.startswith(pre):
            return text
    return {"README.md": "project README", "COLAB_GPU.md": "pointer to docs/COLAB_GPU.md", "LICENSE": "MIT license (code only)", "SUBMISSION_CHECKLIST.md": "submission checklist",
            "RELEASE_MANIFEST.md": "this manifest", "CITATION.cff": "citation metadata", ".gitignore": "git ignore rules", "requirements.txt": "dependencies", "requirements-colab.txt": "Colab dependencies", "environment.yml": "conda environment"}.get(rel, "project file")


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDE_DIRS) and p.name != RELEASE_ZIP.name:
            yield p


def in_release(rel: str) -> bool:
    return not (rel.startswith(NOT_IN_RELEASE) or rel.startswith("data/processed/mars_nlp_processed"))


def manifest(zip_path_note: str) -> str:
    rows = []
    for p in iter_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel == "RELEASE_MANIFEST.md":
            continue
        req = "yes" if rel.startswith(REQUIRED_REPRO) else ("evidence" if rel.startswith(("experiments/results", "reports/", "results/", "predictions/", "figures/")) else "no")
        rows.append({"path": rel, "purpose": purpose(rel), "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "required_for_reproduction": req,
                     "in_release_archive": "yes" if in_release(rel) else "NO (rebuild/obtain; checksum above)"})
    df = pd.DataFrame(rows)
    big = df[df.size_bytes > 5_000_000]
    L = [f"# Release manifest\n\n{AUTHOR}\n", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. {zip_path_note}", "",
         f"* Files listed: {len(df)}; total size {df.size_bytes.sum() / 1e6:.1f} MB; in release archive: {(df.in_release_archive == 'yes').sum()} files, {df[df.in_release_archive == 'yes'].size_bytes.sum() / 1e6:.1f} MB.",
         "* **Excluded from the archive:** third-party raw data (`data/raw/uci`, `data/raw/zenodo`; Zenodo license unverified), the processed corpus (rebuild with `python -m src.build_dataset`, then compare with the SHA-256 below / `data/processed/mars_nlp_processed.sha256`) and fine-tuned transformer checkpoints (hundreds of MB; reproduce with `docs/COLAB_GPU.md`).",
         "* Also excluded: files containing dataset message text (`experiments/results/error_analysis.csv`, `reports/tables/explain_examples.csv`); regenerate them with `python -m src.error_analysis` / `python -m src.explainability`.",
         "* `required_for_reproduction`: yes = needed to rerun/verify; evidence = saved outputs kept for audit; no = convenience.", "",
         f"Large files (> 5 MB): {', '.join(big.path) or 'none'}", "", df.to_markdown(index=False)]
    return "\n".join(L) + "\n"


def build_release() -> None:
    files = [p for p in iter_files() if in_release(p.relative_to(ROOT).as_posix())]
    with zipfile.ZipFile(RELEASE_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, "MARS-NLP/" + p.relative_to(ROOT).as_posix())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", action="store_true", help="also build MARS-NLP-release.zip")
    a = ap.parse_args(argv)
    write_results()
    from . import make_publication_figures

    make_publication_figures.run()
    from . import make_manuscript

    make_manuscript.main()
    (C.REPORTS_DIR / "QA_REPORT.md").write_text(qa_report(), encoding="utf-8")
    (C.REPORTS_DIR / "FINAL_REPRODUCIBILITY_REPORT.md").write_text(reproducibility_report(), encoding="utf-8")
    (ROOT / "SUBMISSION_CHECKLIST.md").write_text(checklist(), encoding="utf-8")
    (ROOT / "RELEASE_MANIFEST.md").write_text(manifest("Release archive: `MARS-NLP-release.zip`." if a.zip else "Release archive not built in this run (`--zip`)."), encoding="utf-8")
    if a.zip:
        build_release()
        print("release archive:", RELEASE_ZIP, f"{RELEASE_ZIP.stat().st_size / 1e6:.1f} MB")
    print("final package written")


if __name__ == "__main__":
    main()
