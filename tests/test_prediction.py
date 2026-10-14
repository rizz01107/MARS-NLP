import numpy as np
import pytest

from src import config as C
from src.evaluation import (cluster_bootstrap_ci, compute_metrics, mcnemar_exact, metric_from_counts,
                            paired_cluster_bootstrap_diff, subset_status)
from src.prediction import InvalidInput, available_models, load_predictor, predict_message, validate_input
from src import train_transformer as T


def test_prediction_format(predictor):
    r = predict_message("URGENT! You have won a prize. Call 09061234567 to claim.", predictor)
    assert r.label in C.LABELS and r.label_id in (0, 1) and C.ID2LABEL[r.label_id] == r.label
    assert 0.0 <= r.confidence <= 1.0 and 0.0 <= r.p_scam <= 1.0
    assert r.language in C.LANGUAGES and r.model_name
    assert isinstance(r.as_dict(), dict)


@pytest.mark.parametrize("bad", ["", "   ", None, "\u200b\u200b", "\U0001F600\U0001F600", 123])
def test_invalid_input_raises_friendly_error(predictor, bad):
    with pytest.raises(InvalidInput):
        predict_message(bad, predictor)


def test_urdu_input_does_not_crash_and_is_flagged(predictor):
    r = predict_message("\u0622\u067e \u06a9\u0627 \u0627\u06a9\u0627\u0624\u0646\u0679 \u0628\u0644\u0627\u06a9 \u06c1\u0648 \u06af\u06cc\u0627 \u06c1\u06d2", predictor)
    assert r.language == C.LANG_URDU and not r.supported_language and r.warnings


def test_roman_urdu_input(predictor):
    r = predict_message("Aap ka account block ho gaya hai. Foran PIN bhejein warna band kar diya jayega.", predictor)
    assert r.language in (C.LANG_ROMAN_URDU, C.LANG_MIXED) and r.supported_language


def test_odd_text_inputs_never_crash(predictor):
    for t in ["<script>alert(1)</script>", "a" * 20000, "mixed \u0622\u067e and English", "NULL", "\n\n\tfree\n\n", "\x00free\x07"]:
        r = predict_message(t, predictor)
        assert r.label in C.LABELS
    assert predict_message("a" * 20000, predictor).truncated


def test_model_loading_and_missing_model(predictor):
    models = available_models()
    assert models and predictor.name in models
    again = load_predictor(next(k for k in models if k.startswith("baseline_")))
    ids, p, kind = again.scores(["free prize call now", "see you at lunch"])
    assert ids.shape == (2,) and p.shape == (2,) and kind in ("probability", "margin_sigmoid")
    with pytest.raises(FileNotFoundError):
        load_predictor("definitely_not_a_model")


def test_validate_input_truncation():
    cleaned, trunc = validate_input("x" * (C.MAX_INPUT_CHARS + 10))
    assert trunc and len(cleaned) == C.MAX_INPUT_CHARS


def test_metrics_and_bootstrap_consistency():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    m = compute_metrics(y, p)
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 3 and abs(m["accuracy"] - 0.75) < 1e-9
    counts = np.array([m["tn"], m["fp"], m["fn"], m["tp"]], float)
    assert abs(float(metric_from_counts(counts, "macro_f1")) - m["macro_f1"]) < 1e-9
    groups = np.arange(len(y))
    ci = cluster_bootstrap_ci(y, p, groups, "macro_f1", n_boot=200)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"] or np.isclose(ci["point"], ci["ci_low"]) or np.isclose(ci["point"], ci["ci_high"])
    d = paired_cluster_bootstrap_diff(y, p, y, groups, n_boot=200)
    assert d["diff"] < 0 and mcnemar_exact(y, p, y)["p_value"] <= 1.0
    ok, note = subset_status(np.array([0, 1] * 3))
    assert not ok and "insufficient" in note


def test_transformer_helpers_without_torch():
    assert T.resolve_model_name("mbert") == "bert-base-multilingual-cased" and T.resolve_model_name("xlmr") == "xlm-roberta-base"
    assert T.label_maps()[1] == C.LABEL2ID
    assert T.select_device("cpu") == "cpu" and T.select_device() in ("cpu", "cuda")   # never raises without CUDA
    w = T.class_weights([0, 0, 0, 1])
    assert w[1] > w[0]
    args = T.build_arg_parser().parse_args(["--model", "xlmr", "--smoke", "--train-languages", "English"])
    assert T.run_tag(args) == "xlmr_train-English_smoke"


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="torch not installed (GPU run is pending)")
def test_transformer_requires_hub_or_local_model():
    assert T._require_torch()
