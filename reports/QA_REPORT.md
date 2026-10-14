# QA report

**Author:** Muhammad Rizwan  
**Affiliation:** Independent Researcher  
**Email:** rizwan01107@gmail.com

## 1. Owner-verified QA (Colab, after both transformers finished)

**47 PASS, 0 FAIL, 0 PENDING, 0 SKIP** - 63 pytest tests passed.

* no group overlap across splits
* variants remain in their original split
* zero exact cross-split text overlap
* processed CSV checksum verified
* split fingerprints unchanged
* mBERT trained/evaluated
* XLM-R trained/evaluated
* both transformers ran on Tesla T4
* metrics recomputed from raw predictions (70 metric values checked)
* robustness metrics traced
* prediction smoke tests passed
* Streamlit AppTest passed
* headless application startup passed

Provenance: Owner-supplied verified summary of the Google Colab (NVIDIA Tesla T4) transformer run and of the Colab QA run. The raw run files (experiments/results/transformer_<tag>.json, transformer_predictions_<tag>.csv, transformer_robustness_<tag>.csv, models/transformers/<tag>/) were NOT present in the sandbox where the final cleanup was performed, so these values could not be recomputed there. If those raw files are copied into experiments/results/, the code prefers them over this file and `python -m src.qa_check` recomputes and checks every value.

## 2. Reproducibility note and the 3 PENDING local checks

**Reproducibility note (transformers).** mBERT and XLM-R were executed and verified in the original Google Colab GPU run (NVIDIA Tesla T4), which passed the owner's Colab QA (47 PASS, 0 FAIL, 0 PENDING, 0 SKIP). Their final metrics are retained as owner-verified results. The raw transformer artifacts from that run (run JSON, prediction and robustness files, checkpoints, environment information) were unavailable in the final local workspace, so these values have not been independently reproduced or re-verified locally, and complete raw-artifact reproducibility is not claimed for mBERT and XLM-R. No results were reconstructed or invented. This limitation does not change the reported verified metrics.

**Evidence check for the raw Colab transformer artifacts (2026-10-08):** the whole workspace was searched for `transformer_*`, mBERT/XLM-R prediction, metrics (JSON/CSV) and environment files (`/mnt/user-data/uploads`, `/mnt/user-data/outputs`, `/mnt/transcripts`, the repository, `/home`, `/tmp`). **No original Colab outputs were found.** The only matches were temporary pytest fixtures (synthetic test data and a tiny randomly initialised smoke model), which are not results and were not used. The raw Colab transformer artifacts were therefore unavailable for independent local verification; nothing was reconstructed or invented, and the owner-supplied values in `results/verified_transformer_results.json` are kept unchanged.

| # | Local QA check | Status | Meaning |
|---|---|---|---|
| 1 | `transformer mbert: raw result files` | PENDING | missing raw mBERT artifacts (`transformer_mbert.json`, predictions, subsets, robustness CSV, checkpoint) |
| 2 | `transformer xlmr: raw result files` | PENDING | missing raw XLM-R artifacts (`transformer_xlmr.json`, predictions, subsets, robustness CSV, checkpoint) |
| 3 | `transformers: no-N/A check on raw executed runs` | PENDING | no-N/A verification for executed transformer runs: it can only run on the raw run files |

These three local checks are PENDING because the raw Colab artifacts are missing; **they are not failures** (0 local checks failed) and are not converted to PASS without actual raw evidence.

## 3. Snapshot-local QA re-run (this repository copy)

`python -m src.qa_check` at 2026-10-13 21:09:59: PASS=50 FAIL=0 PENDING=3 SKIP=0.

The 3 PENDING items are listed in Section 2; they stay PENDING until the raw Colab artifacts are copied into this folder (`experiments/results/transformer_*`, `predictions/transformer_predictions_*.csv`) and `python -m src.qa_check` is re-run.

| Check | Status | Detail |
|---|---|---|
| syntax: compileall src/app/tests | PASS |  |
| imports: every src module imports (no circular/broken imports) | PASS |  |
| pytest -q | PASS | 68 passed in 18.51s |
| dataset validation (src.validate_data, exit code) | PASS | 0 critical failures |
| leakage: no group in >1 split; every variant in its original's split | PASS | {'groups_in_multiple_splits': 0, 'example_groups': [], 'variants_with_split_different_from_original': 0} |
| leakage: no exact text overlap between splits | PASS | 0 shared |
| data: processed CSV checksum equals the audited file | PASS |  |
| data: split fingerprints unchanged (test split not modified) | PASS |  |
| result files exist (baselines/robustness/cross-lingual/final) | PASS | missing: [] |
| error_analysis.csv has exactly the required 7 columns | PASS | ['message', 'language', 'true_label', 'predicted_label', 'confidence', 'error_category', 'model'] |
| final_results.csv has the required columns | PASS |  |
| transformer mbert: raw result files | PENDING | raw run files are not in this snapshot; values come from the owner-verified summary (copy experiments/results/transformer_* from Colab and r |
| transformer xlmr: raw result files | PENDING | raw run files are not in this snapshot; values come from the owner-verified summary (copy experiments/results/transformer_* from Colab and r |
| report generation: python -m src.make_results | PASS |  |
| report generation: python -m src.compare_models | PASS |  |
| report generation: python -m src.error_analysis | PASS |  |
| report generation: python -m src.plots | PASS |  |
| report generation: python -m src.make_report | PASS |  |
| metrics: every Executed row in final_results.csv recomputed from raw predictions | PASS | 60 values checked |
| transformers: no-N/A check on raw executed runs | PENDING | no raw transformer run files in this snapshot; rows are marked 'Reported' from the owner-verified summary (per-subset cells are N/A by desig |
| report contains every executed model's Macro-F1 exactly as in final_results.csv | PASS | [] |
| README contains the Macro-F1 of all 8 curated models (results/FINAL_RESULTS.csv) | PASS | [] |
| report: executed transformers are not described as 'not executed' | PASS | [] |
| report: states Roman Urdu/Mixed is descriptive (5 independent test groups) | PASS |  |
| final package: required files exist | PASS | missing: [] |
| final results equal the owner-verified table (8 models x 5 metrics) | PASS |  |
| per-model error counts equal the verified counts (14 models) | PASS |  |
| robustness degradation: SVM 0.0029, mBERT -0.0055, XLM-R 0.0339 | PASS | {'Char TF-IDF + Logistic Regression': 0.0064, 'Char TF-IDF + Linear SVM': 0.0029, 'Word TF-IDF + Logistic Regression': -0.0043, 'Word TF-IDF |
| statistical comparisons: 3 rows, CI includes zero / excludes zero / includes zero | PASS |  |
| dataset facts recomputed from processed CSV equal the verified facts | PASS | 6160 / 52031 / 4308 / 917 / 935 / 778 groups / 782 / 149 |
| no stale 'transformers not executed / pending' statements in current documents | PASS | [] |
| aggregated FN/FP totals (2661/2154) are not presented as one model | PASS | [] |
| title, author, affiliation and email consistent in the main documents | PASS | [] |
| README uses the required result wording | PASS | [] |
| no claim that XLM-R is best overall or significantly better than the SVM | PASS | [] |
| required limitations present in README, research report and reproducibility report | PASS | {} |
| references: every listed reference is cited and every citation is listed | PASS | cited-not-listed [], listed-not-cited [] |
| core references present (Devlin BERT, Conneau XLM-R, Pedregosa scikit-learn) | PASS | [] |
| no references to methods/models not used | PASS | [] |
| prediction smoke test: baseline_char_lr | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_char_lr_aug | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_char_nb | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_char_nb_aug | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_char_svm | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_char_svm_aug | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_word_lr | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_word_lr_aug | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_word_nb | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_word_nb_aug | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_word_svm | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| prediction smoke test: baseline_word_svm_aug | PASS | English/Roman Urdu/Urdu ok; empty input rejected cleanly |
| Streamlit AppTest (load + 5 inputs, no exception) | PASS |  |
| Streamlit headless server startup (/_stcore/health == ok) | PASS |  |

Documentation-wording checks (reported separately; not part of the 50/0/3/0 count above): PASS=17 FAIL=0.

| Check | Status |
|---|---|
| transformer reproducibility note present (owner-verified, not independently reproduced locally) | PASS |
| the 3 PENDING checks are identified (mBERT artifacts, XLM-R artifacts, no-N/A verification) | PASS |
| no over-claims (locally reproduced / fully reproducible / pending = failed) | PASS |
| manuscript: all required sections present | PASS |
| manuscript: title, author, affiliation, email | PASS |
| manuscript: every Table-4 metric equals results/FINAL_RESULTS.csv (8 models x 5 metrics) | PASS |
| manuscript: robustness degradation values equal results/ROBUSTNESS_RESULTS.csv | PASS |
| manuscript: statistical values equal results/STATISTICAL_COMPARISONS.csv | PASS |
| manuscript: per-model error counts equal results/ERROR_COUNTS.csv (14 models) | PASS |
| manuscript: dataset facts present (6,160 / 52,031 / 4,308 / 917 / 935 / 778 / 782 / 149 / five groups) | PASS |
| manuscript: transformer details (parameters, size, time, latency) match the verified summary | PASS |
| manuscript: no university name in the manuscript, README, reports or citation file | PASS |
| manuscript: required wording and QA records present | PASS |
| manuscript: no forbidden claims or tooling mentions (state-of-the-art, best overall, significant superiority, aggregated FN/FP) | PASS |
| manuscript: references - all listed are cited, all citations listed, core references present | PASS |
| manuscript: Figures 1-3 embedded (files exist) and referenced | PASS |
| manuscript: PDF exists | PASS |
