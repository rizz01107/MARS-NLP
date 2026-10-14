"""Transformer pipeline tests. The end-to-end test uses a tiny RANDOM local model (no downloads); its outputs are marked
smoke_test and must never be treated as results. Synthetic fixtures are used only to test the table-building code."""
import importlib.util
import json

import pandas as pd
import pytest

from src import config as C
from src import make_results as MR
from src import verified as V
from src import train_transformer as T

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("tokenizers") is not None


def _fake_run(tag, smoke=False, pretrained=True, device="cuda"):
    sub = lambda n, f1: {"subset": n, "n": 100, "n_groups": 30, "macro_f1": f1, "macro_f1_ci_low": f1 - .02, "macro_f1_ci_high": f1 + .01,
                         "accuracy": .9, "precision_scam": .91, "recall_scam": .8, "weighted_f1": .9, "status": "ok"}
    return {"model": MR.CANONICAL_TRANSFORMERS[tag], "tag": tag, "smoke_test": smoke, "pretrained": pretrained, "config": {"epochs": 4},
            "device": device, "environment": {"gpu_name": "TEST-GPU"}, "parameters": {"total": 1000}, "model_size_mb": 1.0,
            "train_seconds": 5.0, "best_epoch": 2, "n_train": 10,
            "test_overall": sub("Overall", .9), "test_subsets": [sub("Overall", .9), sub(C.LANG_ENGLISH, .88), sub("Roman Urdu + Mixed", 1.0)],
            "latency_computational_benchmark": {device: {"latency_single_ms_mean": 5.0}}}


def test_make_results_ignores_smoke_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(V, "VERIFIED_SUPPLEMENT", tmp_path / "absent.json")   # no owner-verified summary in this test
    (tmp_path / "transformer_mbert.json").write_text(json.dumps(_fake_run("mbert", smoke=True)))
    (tmp_path / "transformer_xlmr.json").write_text(json.dumps(_fake_run("xlmr", pretrained=False)))
    assert MR.load_transformer_result("mbert") is None and MR.load_transformer_result("xlmr") is None
    rows = MR._transformer_rows()
    assert all(r["Status"] == MR.NOT_RUN and r["Macro_F1"] == "N/A" for r in rows)


def test_make_results_fills_only_from_executed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(V, "VERIFIED_SUPPLEMENT", tmp_path / "absent.json")
    (tmp_path / "transformer_mbert.json").write_text(json.dumps(_fake_run("mbert")))
    pd.DataFrame([{"language_slice": "All", "perturbation_type": "casing_variation", "model_id": "mbert", "degradation": 0.01}]).to_csv(
        tmp_path / "transformer_robustness_mbert.csv", index=False)
    rows = {r["Model"]: r for r in MR._transformer_rows()}
    ok, pend = rows[MR.CANONICAL_TRANSFORMERS["mbert"]], rows[MR.CANONICAL_TRANSFORMERS["xlmr"]]
    assert ok["Status"] == "Executed" and ok["Macro_F1"] == 0.9 and ok["Robustness_Degradation"] == 0.01 and ok["English_F1"] == 0.88
    assert pend["Status"] == MR.NOT_RUN and pend["Macro_F1"] == "N/A"


def test_run_tag_and_smoke_never_canonical():
    a = T.build_arg_parser()
    assert T.run_tag(a.parse_args(["--model", "mbert"])) == "mbert"
    assert T.run_tag(a.parse_args(["--model", "xlmr", "--smoke"])) != "xlmr"
    assert T.run_tag(a.parse_args(["--model", "mbert", "--tiny-offline"])).endswith("_smoke")


def test_checksum_guard_detects_modified_data(tmp_path, monkeypatch, processed_df):
    sha = tmp_path / "x.sha256"
    monkeypatch.setattr(C, "PROCESSED_DIR", tmp_path)
    fake_csv = tmp_path / "data.csv"
    fake_csv.write_text("a,b\n1,2\n")
    monkeypatch.setattr(C, "PROCESSED_CSV", fake_csv)
    (tmp_path / "mars_nlp_processed.sha256").write_text("0" * 64 + "  data.csv\n")
    with pytest.raises(RuntimeError, match="MISMATCH"):
        T.verify_data_integrity()
    del sha


@pytest.mark.skipif(not HAS_TORCH, reason="torch/transformers/tokenizers not installed")
def test_tiny_offline_end_to_end_pipeline(processed_df, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")      # keep the repo's result folders clean
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path / "models")
    args = T.build_arg_parser().parse_args(["--model", "mbert", "--tiny-offline", "--robustness", "--batch-size", "16", "--max-length", "48"])
    # This smoke test uses a random local model; test the pipeline without
    # depending on the production dataset's audit checksum.
    monkeypatch.setattr(T, "verify_data_integrity", lambda: {"sha256": "smoke-test-only", "verified": True})
    res = T.train(args)
    assert res["smoke_test"] and not res["pretrained"] and res["tag"].endswith("_smoke")
    assert res["data_integrity"]["verified"] and res["device"] in ("cpu", "cuda")
    for f in ("transformer_{t}.json", "transformer_predictions_{t}.csv", "transformer_subsets_{t}.csv", "transformer_robustness_{t}.csv"):
        assert (C.RESULTS_DIR / f.format(t=res["tag"])).exists()
    pred = pd.read_csv(C.RESULTS_DIR / f"transformer_predictions_{res['tag']}.csv")
    assert set(pred.columns) == {"message_id", "is_perturbed", "y_true", "y_pred", "p_scam"} and pred.p_scam.between(0, 1).all()
    # variants stay in the test split and refer to evaluated originals
    test = processed_df[processed_df.split == "test"].set_index("message_id")
    assert pred.message_id.isin(test.index).all()
    assert MR.load_transformer_result(res["tag"]) is None            # smoke runs can never become reported results
    from src.prediction import load_predictor, predict_message

    r = predict_message("free prize call now", load_predictor(str(C.MODELS_DIR / "_smoke" / res["tag"])))
    assert r.label in C.LABELS and r.confidence_type == "probability"


# ---------------------------------------------------------------- owner-verified final package
def test_verified_summary_has_required_values():
    sup = V.load_supplement()
    assert sup and sup["transformers"]["xlmr"]["macro_f1"] == 0.9722 and sup["transformers"]["mbert"]["macro_f1"] == 0.8772
    assert sup["transformers"]["xlmr"]["gpu"] == "NVIDIA Tesla T4" and sup["transformers"]["mbert"]["parameters"] == 177854978
    assert sup["qa_owner_verified"]["PASS"] == 47 and sup["qa_owner_verified"]["FAIL"] == 0


def test_supplement_fallback_marks_rows_as_reported_not_executed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)              # no raw run files
    rows = {r["Model"]: r for r in MR._transformer_rows()}
    x = rows[MR.CANONICAL_TRANSFORMERS["xlmr"]]
    assert x["Macro_F1"] == 0.9722 and x["Status"].startswith("Reported") and x["Robustness_Degradation"] == 0.0339


def test_final_tables_match_owner_table_and_saved_baselines():
    from src import final_tables as F

    fin = F.final_results_df().set_index("Model")
    for row in V.load_supplement()["owner_final_table"]:
        for k in ("Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1"):
            assert abs(float(fin.loc[row["Model"], k]) - row[k]) < 1e-4, (row["Model"], k)
    assert fin.loc["Char TF-IDF + Logistic Regression", "Role"].startswith("highest observed")
    assert "validation-selected" in fin.loc["Char TF-IDF + Linear SVM", "Role"]
    assert "best-performing transformer" in fin.loc["XLM-R", "Role"]


@pytest.mark.skipif(not (C.RESULTS_DIR / "error_analysis.csv").exists(),\n                    reason="error_analysis.csv is generated locally and excluded because it contains third-party message text")\ndef test_error_counts_per_model_not_aggregated():
    from src import final_tables as F

    ec = F.error_counts_df().set_index("Model_ID")["Errors"].astype(int).to_dict()
    assert ec == V.load_supplement()["error_counts_test_originals_plus_variants"]


def test_robustness_means_match_verified_values():
    from src import final_tables as F

    m = F.robustness_df().query("Level == 'mean over 10 perturbation types'").set_index("Model")["Degradation"].astype(float)
    assert abs(m["Char TF-IDF + Linear SVM"] - 0.0029) < 1e-4 and abs(m["mBERT"] + 0.0055) < 1e-4 and abs(m["XLM-R"] - 0.0339) < 1e-4
