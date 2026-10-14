# Submission checklist

# MARS-NLP: Robust Multilingual Scam Message Classification with Character-Level and Transformer-Based NLP

**Author:** Muhammad Rizwan  
**Affiliation:** Independent Researcher  
**Email:** rizwan01107@gmail.com

Final manuscript: `reports/MARS_NLP_Final_Manuscript.md` (PDF: `reports/MARS_NLP_Final_Manuscript.pdf`). It is unpublished and has not been peer reviewed.

Checked items were verified from files or by `python -m src.qa_check`; evidence in the right column.

| Done | Item | Evidence |
|---|---|---|
| [x] | Final title | exact title in the manuscript, README, reproducibility report (checked by `qa_check`) |
| [x] | Author information | name, affiliation, email in the manuscript and main documents (checked by `qa_check`) |
| [x] | Abstract | `reports/MARS_NLP_Final_Manuscript.md` - Abstract and Keywords |
| [x] | Introduction | manuscript section 1 |
| [x] | Related Work | manuscript section 2; 13 references, all cited |
| [x] | Methodology | manuscript section 3 (dataset, splitting, perturbations, TF-IDF, classical models, mBERT, XLM-R) |
| [x] | Dataset description | manuscript sections 3.1-3.2; `reports/data_audit.md` |
| [x] | Experimental setup | manuscript section 4; `experiments/configs/` |
| [x] | Results | manuscript section 5; `results/FINAL_RESULTS.csv` |
| [x] | Robustness analysis | manuscript section 5.4; `results/ROBUSTNESS_RESULTS.csv` |
| [x] | Statistical analysis | manuscript section 5.3; `results/STATISTICAL_COMPARISONS.csv` |
| [x] | Error analysis | manuscript section 5.5; `results/ERROR_COUNTS.csv` (per model, not aggregated) |
| [x] | Limitations | manuscript section 7 (all required limitations; checked by `qa_check`) |
| [x] | Conclusion | manuscript section 9 |
| [x] | References | manuscript section 10 (core: Devlin, Conneau, Pedregosa; all cited) |
| [x] | Tables | manuscript Tables 1-9; `results/*.csv` |
| [x] | Figures | `figures/` (300 dpi PNG + PDF), embedded as Figures 1-3 in the manuscript |
| [x] | Reproducibility information | manuscript section 8; `reports/FINAL_REPRODUCIBILITY_REPORT.md`; `docs/COLAB_GPU.md`; `data/README.md` |
| [x] | Code | `src/`, `tests/`, `app/`, `notebooks/` |
| [x] | Requirements | `requirements.txt`, `requirements-colab.txt`, `environment.yml` |
| [x] | README | professor-facing README with all required sections |
| [x] | QA report | `reports/QA_REPORT.md`: owner-verified Colab QA 47 PASS / 0 FAIL / 0 PENDING / 0 SKIP; local release-package QA 50 PASS / 0 FAIL / 3 PENDING / 0 SKIP |
| [x] | No contradictory metrics | `qa_check`: manuscript and results tables equal `results/*.csv` and the owner-verified table |
| [x] | No unsupported claims | automated wording checks (no 'best overall', 'significantly outperforms' or leading-performance claim); the author must still proofread |
| [x] | AI-assisted tools disclosure | Manuscript section 10 and README describe use of ChatGPT and Claude, with author responsibility stated. |
| [ ] | PDF manuscript synchronized with current Markdown | PDF must be regenerated after the AI-use disclosure and section numbering changes (`scripts/build_manuscript_pdf.sh`). |

## Verification status of the transformer results

**Reproducibility note (transformers).** mBERT and XLM-R were executed and verified in the original Google Colab GPU run (NVIDIA Tesla T4), which passed the owner's Colab QA (47 PASS, 0 FAIL, 0 PENDING, 0 SKIP). Their final metrics are retained as owner-verified results. The raw transformer artifacts from that run (run JSON, prediction and robustness files, checkpoints, environment information) were unavailable in the final local workspace, so these values have not been independently reproduced or re-verified locally, and complete raw-artifact reproducibility is not claimed for mBERT and XLM-R. No results were reconstructed or invented. This limitation does not change the reported verified metrics.

| # | Local QA check | Status | Meaning |
|---|---|---|---|
| 1 | `transformer mbert: raw result files` | PENDING | missing raw mBERT artifacts (`transformer_mbert.json`, predictions, subsets, robustness CSV, checkpoint) |
| 2 | `transformer xlmr: raw result files` | PENDING | missing raw XLM-R artifacts (`transformer_xlmr.json`, predictions, subsets, robustness CSV, checkpoint) |
| 3 | `transformers: no-N/A check on raw executed runs` | PENDING | no-N/A verification for executed transformer runs: it can only run on the raw run files |

These three local checks are PENDING because the raw Colab artifacts are missing; **they are not failures** (0 local checks failed) and are not converted to PASS without actual raw evidence.

## Open items before submission (not done by the automated cleanup)

- [ ] Raw Colab transformer artifacts (mBERT, XLM-R) were unavailable in the final local workspace: the 3 PENDING local QA checks (missing raw mBERT artifacts, missing raw XLM-R artifacts, no-N/A verification for executed transformer runs) stay PENDING - not failures - until the artifacts are supplied. Copy the Colab raw files (`experiments/results/transformer_*`, `predictions/transformer_predictions_*.csv`, logs) into the repository and run `python -m src.qa_check` so the transformer values are recomputed locally (currently owner-verified summary only).
- [ ] Confirm the Zenodo dataset license (unverified) and the UCI CC BY 4.0 statement against the live records before redistributing any data.
- [ ] Regenerate the PDF after the manuscript disclosure/source changes; then compare it against the Markdown manuscript.
- [ ] Proofread the generated report text; adapt formatting to the target venue.
- [ ] Check the target journal's AI-use disclosure policy and adjust the disclosure to its required wording before submission.
- [ ] Optional: multi-seed transformer runs and per-language transformer subsets (needs raw files / new GPU runs).
