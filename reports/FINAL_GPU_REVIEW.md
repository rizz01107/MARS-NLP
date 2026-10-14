# FINAL GPU REVIEW

Generated 2026-10-13 21:10:00 by `python -m src.make_gpu_review` from result files only.

## STATUS: EXECUTED on NVIDIA Tesla T4 - owner-verified summary

**Provenance:** Owner-supplied verified summary of the Google Colab (NVIDIA Tesla T4) transformer run and of the Colab QA run. The raw run files (experiments/results/transformer_<tag>.json, transformer_predictions_<tag>.csv, transformer_robustness_<tag>.csv, models/transformers/<tag>/) were NOT present in the sandbox where the final cleanup was performed, so these values could not be recomputed there. If those raw files are copied into experiments/results/, the code prefers them over this file and `python -m src.qa_check` recomputes and checks every value.

## 1. GPU and software

GPU: NVIDIA Tesla T4 (Google Colab) for both models. CUDA / PyTorch / transformers versions are recorded in the raw run JSON (`experiments/results/transformer_<tag>.json`, field `environment`), which is not part of this snapshot; they are therefore not restated here.

## 2. Models and training configuration

| Model                                | HF id                        |   train n |   best epoch |   max epochs |   patience |   batch |    params |   size MB |
|:-------------------------------------|:-----------------------------|----------:|-------------:|-------------:|-----------:|--------:|----------:|----------:|
| mBERT (bert-base-multilingual-cased) | bert-base-multilingual-cased |      4308 |            2 |            4 |          2 |      32 | 177854978 |     681.3 |
| XLM-RoBERTa (xlm-roberta-base)       | xlm-roberta-base             |      4308 |            4 |            4 |          2 |      32 | 278045186 |    1077   |

Protocol: early stopping on validation macro-F1; OOM fallback = batch size 16 (must be recorded explicitly). Defaults of the documented command: lr 2e-05, max length 128, weight decay 0.01, warm-up 0.1, seed 42, class-weighted cross-entropy, AdamW, linear warm-up/decay.

## 3. Training time

| Model                                |   train seconds |
|:-------------------------------------|----------------:|
| mBERT (bert-base-multilingual-cased) |          287.50 |
| XLM-RoBERTa (xlm-roberta-base)       |          295.60 |

## 4. Final test metrics (fixed split, 935 original messages)

| Model                                |   Accuracy |   Precision(Scam) |   Recall(Scam) |   Macro-F1 |   Weighted-F1 | Macro-F1 95% CI   |
|:-------------------------------------|-----------:|------------------:|---------------:|-----------:|--------------:|:------------------|
| mBERT (bert-base-multilingual-cased) |     0.9369 |            0.8750 |         0.7226 |     0.8772 |        0.9344 | [0.754, 0.970]    |
| XLM-RoBERTa (xlm-roberta-base)       |     0.9850 |            0.9862 |         0.9226 |     0.9722 |        0.9848 | [0.946, 0.987]    |

Roman Urdu / Mixed values are descriptive only (5 independent test groups); per-language transformer values need the raw run files.

## 5. Robustness (degradation = original macro-F1 - perturbed macro-F1; mean over 10 synthetic perturbation types)

| Model                                |   Degradation |
|:-------------------------------------|--------------:|
| mBERT (bert-base-multilingual-cased) |       -0.0055 |
| XLM-RoBERTa (xlm-roberta-base)       |        0.0339 |

## 6. Comparison with the best CPU baseline (validation-selected Char TF-IDF + Linear SVM)

**Requested comparison table**

| Model                                                              |   Macro-F1 |   Precision |   Recall | English-F1                 | Roman Urdu/Mixed-F1                                                     |   Robustness Degradation | Status                                                                       |
|:-------------------------------------------------------------------|-----------:|------------:|---------:|:---------------------------|:------------------------------------------------------------------------|-------------------------:|:-----------------------------------------------------------------------------|
| TFIDF-char + LinearSVM (best CPU baseline, selected on validation) |     0.9741 |      0.9931 |   0.9226 | 0.9594                     | 1.0 (descriptive only: 5 independent test groups)                       |                   0.0029 | Executed (CPU)                                                               |
| mBERT (bert-base-multilingual-cased)                               |     0.8772 |      0.875  |   0.7226 | N/A (raw run files needed) | N/A (raw run files needed; descriptive only: 5 independent test groups) |                  -0.0055 | Executed on NVIDIA Tesla T4 (owner-verified; raw files not in this snapshot) |
| XLM-RoBERTa (xlm-roberta-base)                                     |     0.9722 |      0.9862 |   0.9226 | N/A (raw run files needed) | N/A (raw run files needed; descriptive only: 5 independent test groups) |                   0.0339 | Executed on NVIDIA Tesla T4 (owner-verified; raw files not in this snapshot) |

**With 95% cluster-bootstrap CI (overall macro-F1)**

| Model                                                              |   Macro-F1 | Macro-F1 95% CI   |   Precision |   Recall |
|:-------------------------------------------------------------------|-----------:|:------------------|------------:|---------:|
| TFIDF-char + LinearSVM (best CPU baseline, selected on validation) |     0.9741 | [0.949, 0.988]    |      0.9931 |   0.9226 |
| mBERT (bert-base-multilingual-cased)                               |     0.8772 | [0.754, 0.970]    |      0.875  |   0.7226 |
| XLM-RoBERTa (xlm-roberta-base)                                     |     0.9722 | [0.946, 0.987]    |      0.9862 |   0.9226 |


| comparison                                                          |   difference |   ci95_low |   ci95_high | ci_includes_zero   |
|:--------------------------------------------------------------------|-------------:|-----------:|------------:|:-------------------|
| XLM-R vs selected Char TF-IDF + Linear SVM (overall test, macro-F1) |      -0.0019 |    -0.0188 |      0.0133 | True               |
| XLM-R vs mBERT (overall test, macro-F1)                             |       0.0950 |     0.0017 |      0.2236 | False              |
| XLM-R vs mBERT (English-only test, macro-F1)                        |       0.0086 |    -0.0069 |      0.0254 | True               |

## 7. Efficiency (computational benchmark on the Tesla T4 / Colab CPU; NOT an edge-device benchmark)

| Model                                |   GPU ms/msg |   CPU ms/msg |
|:-------------------------------------|-------------:|-------------:|
| mBERT (bert-base-multilingual-cased) |         9.52 |       125.65 |
| XLM-RoBERTa (xlm-roberta-base)       |        10.25 |       117.48 |

## Limitations

* Roman Urdu/Mixed evaluation rests on 5 independent test groups (descriptive only); no real Urdu-script data exists in the sources.
* One seed per model; no seed-variance estimate. Hyper-parameters were fixed a priori (not tuned) except early stopping on validation macro-F1; the hyper-parameter search was not exhaustive.
* Source, language and label semantics are confounded (UCI marketing spam vs Zenodo phishing); Zenodo license unverified.
* Perturbations are synthetic random noise, not optimised adversarial attacks and not equivalent to naturally occurring adversarial messages; SHAP/Integrated Gradients for transformers were not performed.
* The dataset is domain-specific; no Urdu-script benchmark exists in the sources.
* Latency numbers depend on the Tesla T4 / software environment (computational benchmarks).

## Warnings / errors found in run logs

The Colab run logs (`experiments/results/logs/`) are not part of this snapshot, so no warning/error lines can be reported from them here. (Copy them in and re-run `python -m src.make_gpu_review` to list them.)

## QA suite

`python -m src.qa_check` at 2026-10-13 21:09:59: PASS=50 FAIL=0 PENDING=3 SKIP=0; transformers executed per QA: none.

Failed or pending checks:

* [PENDING] transformer mbert: raw result files - raw run files are not in this snapshot; values come from the owner-verified summary (copy experiments/results/transformer_* from Colab and re-run to verify)
* [PENDING] transformer xlmr: raw result files - raw run files are not in this snapshot; values come from the owner-verified summary (copy experiments/results/transformer_* from Colab and re-run to verify)
* [PENDING] transformers: no-N/A check on raw executed runs - no raw transformer run files in this snapshot; rows are marked 'Reported' from the owner-verified summary (per-subset cells are N/A by design)

## Confirmation that results were actually executed

Both transformers were trained and evaluated on a Google Colab Tesla T4 (owner-verified Colab QA: 47 PASS, 0 FAIL, 0 PENDING, 0 SKIP; 63 pytest tests passed; 70 metric values recomputed from raw predictions). The values in this review come from `results/verified_transformer_results.json` (owner-verified Colab T4 run (raw files not in this snapshot)); the raw run files are not in this snapshot, so the snapshot-local QA marks the transformer file checks PENDING. Copy the Colab `experiments/results/transformer_*` files into this folder and run `python -m src.qa_check` to re-verify locally.
