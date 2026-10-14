# results/ - curated final tables

| File | Content | Source of values |
|---|---|---|
| `FINAL_RESULTS.csv` | 8-model comparison (test macro-F1, accuracy, precision, recall, weighted-F1, 95% CI) | baselines: `experiments/results/baseline_results.csv`; transformers: raw run JSON if present, else `verified_transformer_results.json` |
| `ROBUSTNESS_RESULTS.csv` | degradation = original macro-F1 - perturbed macro-F1 (mean and per perturbation type) | `experiments/results/robustness_results.csv`; transformer means from the verified summary |
| `STATISTICAL_COMPARISONS.csv` | paired cluster-bootstrap macro-F1 differences | `verified_transformer_results.json` (owner-verified) |
| `ERROR_COUNTS.csv` | per-model error counts on test originals + variants (never aggregated across models) | `experiments/results/error_analysis.csv`; transformer counts from the verified summary |
| `verified_transformer_results.json` | owner-supplied verified summary of the Colab Tesla T4 run and QA | provenance stated inside the file |

`experiments/results/final_results.csv` is the longer machine-generated table (12 baselines incl. augmentation-trained variants + transformers).
