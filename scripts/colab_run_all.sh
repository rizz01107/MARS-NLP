#!/usr/bin/env bash
# Full GPU phase. Run from the repository root:  bash scripts/colab_run_all.sh
# Stops at the first failure (set -e + pipefail) and keeps every log in experiments/results/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/results/logs
mkdir -p "$LOG"

echo "== 0. preflight (GPU, hub access, data checksum, sklearn version) =="
python -m src.gpu_preflight 2>&1 | tee "$LOG/00_preflight.log"

echo "== 1. smoke tests (real pretrained models, 256 train msgs, 1 epoch; outputs marked smoke_test, never used for results) =="
python -m src.train_transformer --model mbert --smoke --robustness 2>&1 | tee "$LOG/01_smoke_mbert.log"
python -m src.train_transformer --model xlmr  --smoke --robustness 2>&1 | tee "$LOG/02_smoke_xlmr.log"

echo "== 2. full training (4 epochs, early stopping on validation macro-F1) + robustness =="
python -m src.train_transformer --model mbert --epochs 4 --batch-size 32 --robustness 2>&1 | tee "$LOG/03_train_mbert.log"
python -m src.train_transformer --model xlmr  --epochs 4 --batch-size 32 --robustness 2>&1 | tee "$LOG/04_train_xlmr.log"

echo "== 3. regenerate results, comparison, error analysis, plots, report =="
python -m src.make_results      2>&1 | tee "$LOG/05_make_results.log"
python -m src.compare_models    2>&1 | tee "$LOG/06_compare_models.log"
python -m src.error_analysis    2>&1 | tee "$LOG/07_error_analysis.log"
python -m src.plots             2>&1 | tee "$LOG/08_plots.log"
python -m src.make_report       2>&1 | tee "$LOG/09_make_report.log"

echo "== 4. complete QA suite (also regenerates reports/FINAL_GPU_REVIEW.md) =="
python -m src.qa_check 2>&1 | tee "$LOG/10_qa_check.log"
echo "ALL STEPS FINISHED - read reports/FINAL_GPU_REVIEW.md and check the QA summary line above."
