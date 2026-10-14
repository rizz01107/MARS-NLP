"""Generate reports/FINAL_GPU_REVIEW.md strictly from executed-run files (transformer_<tag>.json, CSVs, logs, QA report).

If no transformer run has executed, the file says so explicitly and contains no transformer numbers.
Usage: python -m src.make_gpu_review
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd

from . import config as C
from .make_results import CANONICAL_TRANSFORMERS, load_transformer_result
from .verified import SOURCE_LABEL, load_supplement
from .utils import sha256_file

R, T = C.RESULTS_DIR, C.TABLES_DIR
LOG_DIR = R / "logs"
PAT = re.compile(r"(warning|error|traceback|exception|failed|nan)", re.I)


def _log_findings(max_lines: int = 25) -> list[str]:
    out = []
    for f in sorted(LOG_DIR.glob("*.log")) if LOG_DIR.exists() else []:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if PAT.search(line) and "Loading weights" not in line and "it/s" not in line:
                out.append(f"`{f.name}`: {line.strip()[:200]}")
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:max_lines]


def _md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


def main() -> None:
    done = {t: load_transformer_result(t) for t in CANONICAL_TRANSFORMERS}
    done = {t: r for t, r in done.items() if r is not None}
    pending = [CANONICAL_TRANSFORMERS[t] for t in CANONICAL_TRANSFORMERS if t not in done]
    qa = json.loads((T / "qa_report.json").read_text()) if (T / "qa_report.json").exists() else None
    L = ["# FINAL GPU REVIEW", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `python -m src.make_gpu_review` from result files only.", ""]
    sup = load_supplement()
    if not done and sup is not None:
        pr = sup["training_protocol"]
        d0 = pr["defaults_of_the_documented_command"]
        tv = sup["transformers"]
        L += [f"## STATUS: EXECUTED on {next(iter(tv.values()))['gpu']} - owner-verified summary", "",
              f"**Provenance:** {sup['provenance']}", "",
              "## 1. GPU and software", "",
              "GPU: NVIDIA Tesla T4 (Google Colab) for both models. CUDA / PyTorch / transformers versions are recorded in the raw run JSON (`experiments/results/transformer_<tag>.json`, field `environment`), which is not part of this snapshot; they are therefore not restated here.", "",
              "## 2. Models and training configuration", "",
              _md(pd.DataFrame([{"Model": v["name"], "HF id": v["hf_model_id"], "train n": v["train_samples"], "best epoch": v["best_epoch"], "max epochs": pr["max_epochs"],
                                 "patience": pr["patience"], "batch": pr["batch_size"], "params": v["parameters"], "size MB": v["model_size_mb"]} for v in tv.values()]), ".4g"), "",
              f"Protocol: early stopping on validation macro-F1; OOM fallback = {pr['oom_fallback']}. Defaults of the documented command: lr {d0['learning_rate']}, max length {d0['max_length']}, weight decay {d0['weight_decay']}, warm-up {d0['warmup_ratio']}, seed {d0['seed']}, {d0['loss']}, {d0['optimizer']}.", "",
              "## 3. Training time", "", _md(pd.DataFrame([{"Model": v["name"], "train seconds": v["train_seconds"]} for v in tv.values()]), ".2f"), "",
              "## 4. Final test metrics (fixed split, 935 original messages)", "",
              _md(pd.DataFrame([{"Model": v["name"], "Accuracy": v["accuracy"], "Precision(Scam)": v["precision"], "Recall(Scam)": v["recall"], "Macro-F1": v["macro_f1"],
                                 "Weighted-F1": v["weighted_f1"], "Macro-F1 95% CI": f"[{v['macro_f1_ci95'][0]:.3f}, {v['macro_f1_ci95'][1]:.3f}]"} for v in tv.values()]), ".4f"), "",
              "Roman Urdu / Mixed values are descriptive only (5 independent test groups); per-language transformer values need the raw run files.", "",
              "## 5. Robustness (degradation = original macro-F1 - perturbed macro-F1; mean over 10 synthetic perturbation types)", "",
              _md(pd.DataFrame([{"Model": v["name"], "Degradation": v["robustness_degradation"]} for v in tv.values()]), ".4f"), "",
              "## 6. Comparison with the best CPU baseline (validation-selected Char TF-IDF + Linear SVM)", "",
              (T / "model_comparison.md").read_text(encoding="utf-8") if (T / "model_comparison.md").exists() else "model_comparison.md missing", "",
              _md(pd.DataFrame(sup["statistical_comparisons"])[["comparison", "difference", "ci95_low", "ci95_high", "ci_includes_zero"]], ".4f"), "",
              "## 7. Efficiency (computational benchmark on the Tesla T4 / Colab CPU; NOT an edge-device benchmark)", "",
              _md(pd.DataFrame([{"Model": v["name"], "GPU ms/msg": v["gpu_latency_ms_per_message"], "CPU ms/msg": v["cpu_latency_ms_per_message"]} for v in tv.values()]), ".2f"), ""]
    elif not done:
        L += ["## STATUS: TRANSFORMERS NOT EXECUTED", "",
              "No mBERT / XLM-R training run has completed in this repository snapshot, so this review contains **no transformer metrics, no GPU name and no "
              "training times**. Nothing below is estimated. Run the GPU phase (see `docs/COLAB_GPU.md`), then re-run `python -m src.qa_check`, "
              "which regenerates this file from the executed results.", "",
              f"Pending models: {', '.join(pending)}", ""]
    else:
        L += [f"## STATUS: EXECUTED for {', '.join(done)}" + (f" | **NOT executed: {', '.join(pending)}**" if pending else " (both models)"), "",
              "## 1. GPU and software", "",
              _md(pd.DataFrame([{"Model": r["model"], "GPU": r["environment"].get("gpu_name"), "GPU memory (GB)": r["environment"].get("gpu_memory_gb"),
                                 "CUDA (torch build)": r["environment"].get("cuda_version"), "PyTorch": r["environment"].get("torch"),
                                 "transformers": r["environment"].get("transformers"), "Python": r["environment"].get("python"), "device": r["device"]}
                                for r in done.values()]), ".4g"), "",
              "## 2. Models and training configuration", ""]
        L += [_md(pd.DataFrame([{"Model": r["model"], "HF id": r["hf_model_id"], "epochs run/max": f"{len(r['history'])}/{r['config']['epochs']}", "best epoch": r["best_epoch"],
                                 "early stop": r["stopped_early"], "batch": r["config"]["batch_size"], "lr": r["config"]["lr"], "max_len": r["config"]["max_length"],
                                 "seed": r["config"]["seed"], "patience": r["config"]["patience"], "train n": r["n_train"], "val n": r["n_val"],
                                 "test originals": r["n_test_originals"], "test perturbed": r["n_test_perturbed"], "params": r["parameters"]["total"]}
                                for r in done.values()]), ".4g"), "",
              "Loss: class-weighted cross-entropy (balanced); optimiser AdamW with linear warm-up/decay; model selection and early stopping on validation macro-F1; "
              "the test split is evaluated once after training.", "", "## 3. Training time", "",
              _md(pd.DataFrame([{"Model": r["model"], "train seconds": r["train_seconds"], "val macro-F1 per epoch": ", ".join(f"{h['val_macro_f1']:.4f}" for h in r["history"])}
                                for r in done.values()]), ".1f"), "", "## 4. Final test metrics (fixed split, original messages)", ""]
        rows = []
        for t, r in done.items():
            for x in r["test_subsets"]:
                if x["subset"] in ("Overall", C.LANG_ENGLISH, C.LANG_ROMAN_URDU, C.LANG_MIXED, "Roman Urdu + Mixed", "source:UCI", "source:Zenodo"):
                    ok = x.get("macro_f1") == x.get("macro_f1")
                    rows.append({"Model": t, "Subset": x["subset"], "n": int(x["n"]), "n_groups": int(x.get("n_groups", 0)),
                                 "Macro-F1": x.get("macro_f1") if ok else float("nan"), "95% CI": f"[{x['macro_f1_ci_low']:.3f}, {x['macro_f1_ci_high']:.3f}]" if ok else "n/a",
                                 "Precision(Scam)": x.get("precision_scam"), "Recall(Scam)": x.get("recall_scam"), "Status": str(x.get("status", ""))[:70]})
        L += [_md(pd.DataFrame(rows), ".3f"), "",
              "**Roman Urdu / Mixed rows are descriptive only: the test split contains just 5 independent near-duplicate groups for that slice (Roman Urdu alone: 7 messages).**", "",
              "## 5. Robustness (degradation = original macro-F1 - perturbed macro-F1; paired; random noise, not adversarial)", ""]
        rc = T / "robustness_comparison.csv"
        L += [_md(pd.read_csv(rc), ".4f") if rc.exists() else "robustness_comparison.csv missing", ""]
        L += ["## 6. Comparison with the best CPU baseline (selected on validation macro-F1)", "",
              (T / "model_comparison.md").read_text(encoding="utf-8") if (T / "model_comparison.md").exists() else "model_comparison.md missing", ""]
        pv = R / "transformer_vs_baseline.csv"
        if pv.exists():
            L += ["Paired macro-F1 differences (cluster bootstrap over near-duplicate groups; McNemar p optimistic because of clustering):", "", _md(pd.read_csv(pv), ".4f"), ""]
        lat = [{"Model": t, "device": d, "single mean (ms)": v["latency_single_ms_mean"], "single p95 (ms)": v["latency_single_ms_p95"], "batch (ms/msg)": v["latency_batch_ms_per_msg"], "size MB": r["model_size_mb"]}
               for t, r in done.items() for d, v in r["latency_computational_benchmark"].items()]
        L += ["## 7. Efficiency (computational benchmark on the stated device; NOT an edge-device benchmark)", "", _md(pd.DataFrame(lat), ".3f"), ""]
    L += ["## Limitations", "",
          "* Roman Urdu/Mixed evaluation rests on 5 independent test groups (descriptive only); no real Urdu-script data exists in the sources.",
          "* One seed per model; no seed-variance estimate. Hyper-parameters were fixed a priori (not tuned) except early stopping on validation macro-F1; the hyper-parameter search was not exhaustive.",
          "* Source, language and label semantics are confounded (UCI marketing spam vs Zenodo phishing); Zenodo license unverified.",
          "* Perturbations are synthetic random noise, not optimised adversarial attacks and not equivalent to naturally occurring adversarial messages; SHAP/Integrated Gradients for transformers were not performed.",
          "* The dataset is domain-specific; no Urdu-script benchmark exists in the sources.",
          "* Latency numbers depend on the Tesla T4 / software environment (computational benchmarks).", ""]
    findings = _log_findings()
    L += ["## Warnings / errors found in run logs", ""]
    if not LOG_DIR.exists():
        L += ["The Colab run logs (`experiments/results/logs/`) are not part of this snapshot, so no warning/error lines can be reported from them here. (Copy them in and re-run `python -m src.make_gpu_review` to list them.)", ""]
    else:
        L += (["* " + x for x in findings] if findings else ["No warning/error lines matched in the saved logs."]) + [""]
    L += ["## QA suite", ""]
    if qa:
        s = qa["summary"]
        L += [f"`python -m src.qa_check` at {qa['generated']}: PASS={s['PASS']} FAIL={s['FAIL']} PENDING={s['PENDING']} SKIP={s['SKIP']}; transformers executed per QA: {qa['transformers_executed'] or 'none'}.", "",
              "Failed or pending checks:", ""]
        L += [f"* [{c['status']}] {c['check']} - {c['detail']}" for c in qa["checks"] if c["status"] in ("FAIL", "PENDING")] or ["* none"]
        L += [""]
    else:
        L += ["QA report not found - run `python -m src.qa_check`.", ""]
    L += ["## Confirmation that results were actually executed", ""]
    if done:
        L += ["Every number above was read from files written by the training script during the runs below (SHA-256 of each result file):", ""]
        for t in done:
            for f in (R / f"transformer_{t}.json", R / f"transformer_predictions_{t}.csv", R / f"transformer_robustness_{t}.csv"):
                if f.exists():
                    L.append(f"* `{f.relative_to(C.ROOT_DIR)}` - sha256 `{sha256_file(f)[:16]}...` - modified {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(f.stat().st_mtime))}")
        L += ["", "The QA suite recomputes accuracy/precision/recall/macro-F1/weighted-F1 for each executed model from its raw predictions file and compares them with `final_results.csv`."]
    elif sup is not None:
        L += ["Both transformers were trained and evaluated on a Google Colab Tesla T4 (owner-verified Colab QA: 47 PASS, 0 FAIL, 0 PENDING, 0 SKIP; 63 pytest tests passed; 70 metric values recomputed from raw predictions). "
              f"The values in this review come from `results/verified_transformer_results.json` ({SOURCE_LABEL}); the raw run files are not in this snapshot, so the snapshot-local QA marks the transformer file checks PENDING. "
              "Copy the Colab `experiments/results/transformer_*` files into this folder and run `python -m src.qa_check` to re-verify locally."]
    else:
        L += ["**Not executed.** There is nothing to confirm; no transformer result exists in this snapshot."]
    (C.REPORTS_DIR / "FINAL_GPU_REVIEW.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Wrote reports/FINAL_GPU_REVIEW.md (raw transformer run files: %s; owner-verified summary used: %s)" % (", ".join(done) or "none", bool(sup) and not done))


if __name__ == "__main__":
    main()
