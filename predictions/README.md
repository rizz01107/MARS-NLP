# predictions/

* `baseline_predictions.csv` - test-original predictions of the 12 CPU baselines (`model_id,message_id,y_true,y_pred,p_scam`); used by the QA suite to recompute every baseline metric.
* `transformer_predictions_<tag>.csv` (`mbert`, `xlmr`) - produced by the Colab run for test originals and their perturbed variants. Copy them here (and into `experiments/results/`) to let `python -m src.qa_check` recompute the transformer metrics; they are **not present in this snapshot**.
