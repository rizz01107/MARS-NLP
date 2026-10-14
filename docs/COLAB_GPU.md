# GPU phase on Google Colab (how the transformer results were produced)

mBERT and XLM-R were fine-tuned and evaluated on a Google Colab **NVIDIA Tesla T4** GPU following these steps. The final
numbers are in `results/FINAL_RESULTS.csv` and documented in `reports/FINAL_REPRODUCIBILITY_REPORT.md`. This page lets you
repeat or audit the run. **Do not retrain to "refresh" numbers** - the verified results are the saved outputs.

## 0. Hardware
Use **Runtime -> Change runtime type -> Hardware accelerator: GPU**. A **Tesla T4 (16 GB)** was used and is recommended; any
CUDA GPU works but timings and latency will differ (they depend on the GPU and software environment). Without a GPU the code
falls back to CPU, which is far too slow for full training.

## 1. Get the project into Colab (pick ONE)
**A. Upload a zip** that contains `data/processed/` (the release archive does not ship third-party data; use the full project zip you hold):
```python
from google.colab import files
files.upload()                       # choose the project zip
```
```bash
!unzip -q -o MARS-NLP*.zip -d /content
%cd /content/MARS-NLP
```
**B. Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
!unzip -q -o "/content/drive/MyDrive/<your-zip>.zip" -d /content
%cd /content/MARS-NLP
```
The processed dataset must be the audited one. **Do not run `src.build_dataset` in Colab.**

## 2. Environment setup and dependencies
```bash
!pip install -q -r requirements-colab.txt
```
`scikit-learn` is pinned to **1.8.0** (the version the saved CPU baselines were pickled with). If pip changed it, restart the
session (**Runtime -> Restart session**) and `%cd /content/MARS-NLP` again.

## 3. Preflight (must print `PREFLIGHT OK`)
```bash
!python -m src.gpu_preflight
```
Checks GPU/CUDA/torch/transformers, Hugging Face access for both tokenizers, the SHA-256 of `data/processed/mars_nlp_processed.csv`,
the train/validation/test message-id fingerprints and the scikit-learn version.

## 4. Smoke tests (pipeline checks only - NOT results)
```bash
!python -m src.train_transformer --model mbert --smoke --robustness
!python -m src.train_transformer --model xlmr  --smoke --robustness
```
Real pretrained models, 256 training messages, 1 epoch, at most 128 validation/test messages. Outputs are tagged `*_smoke`
(`experiments/results/transformer_<model>_smoke.*`, `models/_smoke/`). **Smoke outputs are never used by `make_results`, the
report, the figures or the QA suite, and their metrics are meaningless.**

## 5. Full training (these produce the final results)
```bash
!python -m src.train_transformer --model mbert --epochs 4 --batch-size 32 --robustness
!python -m src.train_transformer --model xlmr  --epochs 4 --batch-size 32 --robustness
```
* **Maximum 4 epochs**, **early stopping on validation macro-F1 with patience = 2**, model selection on validation only; the test split is evaluated once.
* **Batch size 32.** If the GPU runs out of memory, use `--batch-size 16` and **record this explicitly** in the report (the run JSON stores the configuration).
* Other settings (documented-command defaults): lr 2e-5, max length 128, weight decay 0.01, 10% warm-up, seed 42, class-weighted cross-entropy.

## 6. One-shot alternative (stops at the first failure and keeps logs)
```bash
!bash scripts/colab_run_all.sh
```

## 7. Regenerate results and run QA
```bash
!python -m src.make_results
!python -m src.compare_models
!python -m src.error_analysis
!python -m src.plots
!python -m src.make_report
!python -m src.qa_check              # pytest, validation, leakage, metric traceability, report checks, Streamlit, predictions
!python -m src.make_final_package    # results/, figures/, reports, checklist, manifest from the saved results
```

## 8. Where the outputs are saved
**Final results (full training runs, tag `mbert` / `xlmr`):**

| File | Content |
|---|---|
| `experiments/results/transformer_<tag>.json` | configuration, GPU/CUDA/torch versions, parameters, training time, history, metrics, latency, data checksum, split fingerprint |
| `experiments/results/transformer_predictions_<tag>.csv` | `message_id,is_perturbed,y_true,y_pred,p_scam` for all test originals and their variants (copy to `predictions/`) |
| `experiments/results/transformer_subsets_<tag>.csv` | Overall / English / Roman Urdu / Mixed / Roman Urdu + Mixed / per-source metrics with 95% CIs |
| `experiments/results/transformer_robustness_<tag>.csv` | paired degradation per perturbation type and language slice |
| `models/transformers/<tag>/` | fine-tuned model and tokenizer (hundreds of MB; not committed) |

Derived: `results/FINAL_RESULTS.csv`, `results/ROBUSTNESS_RESULTS.csv`, `results/STATISTICAL_COMPARISONS.csv`, `results/ERROR_COUNTS.csv`,
`figures/*.png|pdf`, `reports/*.md`, `README.md`.

**Smoke-test outputs (never final):** any file whose name or folder contains `_smoke` (`experiments/results/transformer_*_smoke*`, `models/_smoke/`).

## 9. How to verify the results
```bash
!python -m src.qa_check
```
It recomputes accuracy, precision, recall, macro-F1 and weighted-F1 for every executed model from its raw predictions file and
compares them with the result tables, checks the checksum/fingerprints (test split unchanged), leakage, report consistency and
that no `N/A` remains for an executed transformer. Expected: `FAIL=0` and `transformers executed: ['mbert', 'xlmr']` once the raw
Colab files are present in `experiments/results/`.
