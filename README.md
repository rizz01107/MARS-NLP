# MARS-NLP: Robust Multilingual Scam Message Classification with Character-Level and Transformer-Based NLP

**Author:** Muhammad Rizwan  
**Affiliation:** Independent Researcher  
**Email:** rizwan01107@gmail.com

## Project overview
MARS-NLP classifies SMS-style messages as *Scam* (scam/spam/phishing) or *Legitimate* in English, Roman Urdu and code-switched text. It compares character- and word-level TF-IDF baselines (logistic regression, linear SVM, naive Bayes) with fine-tuned mBERT and XLM-R under **group-aware evaluation** and a synthetic perturbation suite. The full manuscript is [`reports/MARS_NLP_Final_Manuscript.md`](reports/MARS_NLP_Final_Manuscript.md) (PDF: [`reports/MARS_NLP_Final_Manuscript.pdf`](reports/MARS_NLP_Final_Manuscript.pdf)); it is a research manuscript prepared for submission and has **not been peer reviewed or published**. All numbers in this README come from saved result files; [`reports/FINAL_REPRODUCIBILITY_REPORT.md`](reports/FINAL_REPRODUCIBILITY_REPORT.md) lists the source file of every value.

## Research motivation and objective
Scam messages exploit urgency and impersonation, and in South Asia they are often written in Roman Urdu (Urdu in Latin script, with no standard spelling) mixed with English. Public SMS corpora contain duplicated and templated messages, so naive splits leak and report near-ceiling scores. The objective is to measure how well TF-IDF baselines and multilingual transformers detect scam messages when evaluation respects near-duplicate/template structure, how they behave under a synthetic perturbation suite, and what the evidence does and does not support.

## Key contributions
* An audited corpus of 6,160 original messages with a **group-level** train/validation/test split; each of the 52,031 synthetic perturbed variants stays in its source message's split (checksum- and fingerprint-verified).
* A ten-type synthetic perturbation suite and a paired robustness measure (degradation = original macro-F1 - perturbed macro-F1).
* A comparison of six TF-IDF classifiers with mBERT and XLM-R using cluster-bootstrap confidence intervals and paired comparisons.
* Transparent reporting of limits: five independent Roman Urdu/Mixed test groups, no Urdu-script data, and transformer metrics that are owner-verified results of the original Colab run.

## Dataset statistics
| Item | Value |
|---|---|
| Original messages | 6,160 (UCI SMS Spam Collection + Zenodo Roman-Urdu phishing SMS) |
| Synthetic perturbed variants | 52,031 |
| Train / validation / test originals | 4,308 / 917 / 935 |
| Test message groups | 778 (English 782 messages; Roman Urdu + Mixed 149 messages from only **5** independent groups) |
| Near-duplicate groups (all originals) | 4,970 (Zenodo: 19) |
| Urdu-script messages | 0 - no Urdu-script benchmark is available |

Sources: UCI SMS Spam Collection (Almeida et al., 2011; CC BY 4.0 per UCI page) and the Roman Urdu Phishing SMS Dataset (Zenodo, DOI 10.5281/zenodo.21810885; **license unverified**). Data are not committed; see `data/README.md`.

## Methodology
### Data splitting
Exact duplicates are removed; near-duplicate/templated messages are merged into groups (character 3-5-gram TF-IDF cosine >= 0.8); whole groups are assigned to train/validation/test (70/15/15, stratified by source and label, seed 42). Perturbed variants inherit their original's split. Verified: no group in more than one split, no exact text overlap across splits, processed-CSV SHA-256 checksum and split fingerprints (`data/processed/`) unchanged.

### Synthetic perturbation strategy
Ten seeded, synthetic noise perturbations derived only from real originals: spelling variation, character deletion, character substitution, word spacing variation, punctuation variation, casing variation, informal abbreviation, Roman Urdu spelling variation, code switching, minor noise. They are **not** equivalent to naturally occurring adversarial messages.

### Models
TF-IDF (character 2-5-grams or word 1-2-grams) with Logistic Regression, Linear SVM or Multinomial Naive Bayes (hyper-parameters chosen on validation macro-F1); mBERT (`bert-base-multilingual-cased`) and XLM-R (`xlm-roberta-base`) fine-tuned on a Google Colab NVIDIA Tesla T4 (max 4 epochs, early stopping on validation macro-F1, patience 2, batch size 32).

## Model comparison (final results)
Fixed test split, 935 original messages (Precision/Recall = Scam class; CI = 95% cluster bootstrap). Source: `results/FINAL_RESULTS.csv`.

| Model                             |   Accuracy |   Precision |   Recall |   Macro-F1 |   Weighted-F1 | Macro-F1 95% CI   |
|:----------------------------------|-----------:|------------:|---------:|-----------:|--------------:|:------------------|
| Char TF-IDF + Logistic Regression |     0.9872 |      0.9931 |   0.9290 |     0.9762 |        0.9870 | [0.952, 0.990]    |
| Char TF-IDF + Linear SVM          |     0.9861 |      0.9931 |   0.9226 |     0.9741 |        0.9859 | [0.949, 0.988]    |
| Word TF-IDF + Logistic Regression |     0.9690 |      0.8938 |   0.9226 |     0.9446 |        0.9692 | [0.857, 0.987]    |
| Word TF-IDF + Linear SVM          |     0.9861 |      0.9931 |   0.9226 |     0.9741 |        0.9859 | [0.948, 0.989]    |
| Word TF-IDF + Naive Bayes         |     0.9369 |      0.7927 |   0.8387 |     0.8885 |        0.9376 | [0.750, 0.967]    |
| Char TF-IDF + Naive Bayes         |     0.9283 |      0.7500 |   0.8516 |     0.8770 |        0.9301 | [0.718, 0.967]    |
| mBERT                             |     0.9369 |      0.8750 |   0.7226 |     0.8772 |        0.9344 | [0.754, 0.970]    |
| XLM-R                             |     0.9850 |      0.9862 |   0.9226 |     0.9722 |        0.9848 | [0.946, 0.987]    |

* **Char TF-IDF + Logistic Regression achieved the highest observed test macro-F1 (0.9762).**
* **Char TF-IDF + Linear SVM was the validation-selected CPU baseline and achieved 0.9741 test macro-F1.**
* **XLM-R was the best-performing transformer with 0.9722 macro-F1.** Its paired difference to the selected SVM is -0.0019 (95% CI [-0.0188, 0.0133], includes zero), so no superiority is claimed; XLM-R vs mBERT is +0.0950 (95% CI [0.0017, 0.2236]); English-only XLM-R vs mBERT +0.0086 (95% CI [-0.0069, 0.0254], includes zero). See `results/STATISTICAL_COMPARISONS.csv`.

![Figure 1. Macro-F1 comparison](figures/model_macro_f1_comparison.png)

## Robustness evaluation
`degradation = original macro-F1 - perturbed macro-F1` (mean over the 10 synthetic perturbation types, paired test messages; lower is better). This is robustness to the project's synthetic perturbation suite, not real-world adversarial robustness. Source: `results/ROBUSTNESS_RESULTS.csv`.

| Model                             |   Degradation |
|:----------------------------------|--------------:|
| Char TF-IDF + Logistic Regression |        0.0064 |
| Char TF-IDF + Linear SVM          |        0.0029 |
| Word TF-IDF + Logistic Regression |       -0.0043 |
| Word TF-IDF + Linear SVM          |        0.0126 |
| Word TF-IDF + Naive Bayes         |       -0.0141 |
| Char TF-IDF + Naive Bayes         |       -0.0356 |
| mBERT                             |       -0.0055 |
| XLM-R                             |        0.0339 |

![Figure 2. Robustness degradation](figures/robustness_degradation.png)

![Figure 3. Experimental pipeline](figures/mars_nlp_pipeline.png)

## Installation and reproducibility
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # tested with Python 3.12; scikit-learn 1.8.0 reproduces the saved baselines
# raw files -> data/raw/uci/SMSSpamCollection, data/raw/zenodo/phishing_dataset_1000.xlsx (see data/README.md)
python -m src.build_dataset && python -m src.validate_data       # deterministic; compare with data/processed/mars_nlp_processed.sha256
python -m src.run_pipeline --skip-build                           # CPU: baselines, robustness, cross-lingual, errors, explainability
pytest -q && python -m src.qa_check                               # tests + complete QA suite
streamlit run app/app.py                                          # research-prototype demo
python -m src.make_final_package                                  # results/, figures/, reports, manifest, manuscript from saved results
```
Do not retrain or rebuild the dataset to "refresh" numbers: the verified results are the saved outputs listed in `reports/FINAL_REPRODUCIBILITY_REPORT.md`.

## Google Colab GPU instructions
Transformers were trained on a Google Colab **Tesla T4** (recommended runtime). Full steps, smoke tests, OOM fallback (batch size 16 must be recorded) and expected outputs: [`docs/COLAB_GPU.md`](docs/COLAB_GPU.md). One-shot: `bash scripts/colab_run_all.sh`.

## Testing and QA
**Original owner-verified Colab QA (after both transformers finished): 47 PASS, 0 FAIL, 0 PENDING, 0 SKIP**; 63 pytest tests passed; no group overlap across splits; variants stay in their original split; zero exact cross-split text overlap; processed-CSV checksum verified; split fingerprints unchanged; mBERT and XLM-R trained/evaluated on a Tesla T4; metrics recomputed from raw predictions (70 values checked); robustness metrics traced; prediction smoke tests, Streamlit AppTest and headless startup passed.

Snapshot-local QA of the release package: **50 PASS, 0 FAIL, 3 PENDING, 0 SKIP**. The 3 PENDING checks:

| # | Local QA check | Status | Meaning |
|---|---|---|---|
| 1 | `transformer mbert: raw result files` | PENDING | missing raw mBERT artifacts (`transformer_mbert.json`, predictions, subsets, robustness CSV, checkpoint) |
| 2 | `transformer xlmr: raw result files` | PENDING | missing raw XLM-R artifacts (`transformer_xlmr.json`, predictions, subsets, robustness CSV, checkpoint) |
| 3 | `transformers: no-N/A check on raw executed runs` | PENDING | no-N/A verification for executed transformer runs: it can only run on the raw run files |

These three local checks are PENDING because the raw Colab artifacts are missing; **they are not failures** (0 local checks failed) and are not converted to PASS without actual raw evidence.

**Reproducibility note (transformers).** mBERT and XLM-R were executed and verified in the original Google Colab GPU run (NVIDIA Tesla T4), which passed the owner's Colab QA (47 PASS, 0 FAIL, 0 PENDING, 0 SKIP). Their final metrics are retained as owner-verified results. The raw transformer artifacts from that run (run JSON, prediction and robustness files, checkpoints, environment information) were unavailable in the final local workspace, so these values have not been independently reproduced or re-verified locally, and complete raw-artifact reproducibility is not claimed for mBERT and XLM-R. No results were reconstructed or invented. This limitation does not change the reported verified metrics.

Details: `reports/QA_REPORT.md`.

## Limitations
* The dataset is domain-specific; the perturbations are synthetic and not equivalent to naturally occurring adversarial messages.
* Roman Urdu/Mixed evidence is limited: only **5 independent test groups** support that analysis (descriptive only); there is no comprehensive Urdu-script benchmark.
* Multi-seed variance was not estimated for every model; the hyper-parameter search was not exhaustive.
* Computational measurements depend on the Tesla T4 / software environment.
* Transformer interpretability methods such as SHAP/Integrated Gradients were not performed.
* Source, language and label semantics are confounded; language labels are heuristic; the Zenodo license is unverified.

## Project structure
```
data/          data README, scaffold audit files, checksum/fingerprints (third-party data are NOT committed; see data/README.md)
models/        saved CPU baselines and transformer checkpoints (not committed; rebuild/obtain as described)
predictions/   raw prediction files (baselines; transformer files from the Colab run go here)
results/       FINAL_RESULTS.csv, ROBUSTNESS_RESULTS.csv, STATISTICAL_COMPARISONS.csv, ERROR_COUNTS.csv, verified_transformer_results.json
experiments/   configs/ and results/ (raw experiment outputs kept for audit; the code reads and writes here)
reports/       MARS_NLP_Final_Manuscript (.md/.pdf), extended technical report, FINAL_REPRODUCIBILITY_REPORT, QA_REPORT, data audit, tables/, figures/
figures/       publication figures (PNG 300 dpi + PDF)
notebooks/     01-07 executed notebooks      src/   pipeline code      tests/   pytest suite      docs/   COLAB_GPU.md
requirements.txt  requirements-colab.txt  environment.yml  CITATION.cff  LICENSE  SUBMISSION_CHECKLIST.md  RELEASE_MANIFEST.md
```

## Citation
If you use this work, cite it as software/manuscript (see `CITATION.cff`): Muhammad Rizwan, *MARS-NLP: Robust Multilingual Scam Message Classification with Character-Level and Transformer-Based NLP*, Independent Researcher (unpublished manuscript). Please also cite the data sources - Almeida, Gomez Hidalgo & Yamakami (2011), DocEng '11; Siddiq, *Roman Urdu Phishing SMS Dataset for Smishing Detection Research*, Zenodo, DOI 10.5281/zenodo.21810885 - and the core methods: Devlin et al. (2019), NAACL-HLT (BERT); Conneau et al. (2020), ACL (XLM-R); Pedregosa et al. (2011), JMLR (scikit-learn). Code license: MIT (code only; the datasets keep their own terms).
