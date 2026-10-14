"""Generate reports/MARS_NLP_Research_Report.md and README.md from the executed results (numbers are read from
experiments/results/*.csv and reports/tables/*; narrative claims were checked against them).

Usage: python -m src.make_report
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

R, T = C.RESULTS_DIR, C.TABLES_DIR
TITLE = "MARS-NLP: Robust Multilingual Scam Message Classification with Character-Level and Transformer-Based NLP"


REPRO_NOTE = ("**Reproducibility note (transformers).** mBERT and XLM-R were executed and verified in the original Google Colab GPU run (NVIDIA Tesla T4), "
              "which passed the owner's Colab QA (47 PASS, 0 FAIL, 0 PENDING, 0 SKIP). Their final metrics are retained as owner-verified results. "
              "The raw transformer artifacts from that run (run JSON, prediction and robustness files, checkpoints, environment information) were unavailable "
              "in the final local workspace, so these values have not been independently reproduced or re-verified locally, and complete raw-artifact "
              "reproducibility is not claimed for mBERT and XLM-R. No results were reconstructed or invented. This limitation does not change the reported verified metrics.")
PENDING_ROWS = [
    ("1", "`transformer mbert: raw result files`", "PENDING", "missing raw mBERT artifacts (`transformer_mbert.json`, predictions, subsets, robustness CSV, checkpoint)"),
    ("2", "`transformer xlmr: raw result files`", "PENDING", "missing raw XLM-R artifacts (`transformer_xlmr.json`, predictions, subsets, robustness CSV, checkpoint)"),
    ("3", "`transformers: no-N/A check on raw executed runs`", "PENDING", "no-N/A verification for executed transformer runs: it can only run on the raw run files")]
PENDING_NOTE = "These three local checks are PENDING because the raw Colab artifacts are missing; **they are not failures** (0 local checks failed) and are not converted to PASS without actual raw evidence."


def pending_table_md() -> str:
    return "| # | Local QA check | Status | Meaning |\n|---|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in PENDING_ROWS)


def local_qa_line() -> str:
    f = T / "qa_report.json"
    if not f.exists():
        return "Snapshot-local QA of the release package: see `reports/QA_REPORT.md`."
    q = json.loads(f.read_text())["summary"]
    return f"Snapshot-local QA of the release package: **{q['PASS']} PASS, {q['FAIL']} FAIL, {q['PENDING']} PENDING, {q['SKIP']} SKIP**."


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def load() -> dict:
    d = {"base": pd.read_csv(R / "baseline_results.csv"), "rob": pd.read_csv(R / "robustness_results.csv"),
         "xl": pd.read_csv(R / "cross_lingual_results.csv"), "cv": pd.read_csv(R / "group_cv_results.csv"),
         "cvr": pd.read_csv(R / "group_cv_robustness.csv"), "err": pd.read_csv(R / "error_summary.csv") if (R / "error_summary.csv").exists() else pd.read_csv(T / "error_summary.csv"),
         "final": pd.read_csv(R / "final_results.csv", keep_default_na=False),
         "build": json.load(open(T / "build_summary.json")), "val": json.load(open(T / "validation_report.json")),
         "cfg": json.load(open(R / "baseline_config.json")), "groups": pd.read_csv(T / "eda_group_structure.csv"),
         "audit": pd.read_csv(C.REPORTS_DIR / "data_audit.csv")}
    return d


def fmt_ci(r) -> str:
    return f"{r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]"


def build_numbers(d: dict) -> dict:
    b, cv, base = d["build"], d["cv"], d["base"]
    ov = base[(base.subset == "Overall")]
    orig_trained = ov[~ov.aug_trained]
    top = orig_trained.sort_values("macro_f1", ascending=False).iloc[0]
    cvo = cv[cv.subset == "Overall"].sort_values("macro_f1", ascending=False).iloc[0]
    cvru = cv[cv.subset == "Roman Urdu + Mixed"].set_index("model_id")
    sp = b["split_counts_originals"]
    return {"top": top, "cvo": cvo, "cvru": cvru, "sp": sp, "b": b}


def tables(d: dict) -> dict:
    base, rob, xl, cv, cvr = d["base"], d["rob"], d["xl"], d["cv"], d["cvr"]
    ov = base[base.subset == "Overall"].copy()
    t_overall = pd.DataFrame({"Model": ov.model.str.replace("TFIDF-", ""), "Best param": ov.best_param,
                              "Val macro-F1": ov.val_macro_f1, "Accuracy": ov.accuracy, "Precision (Scam)": ov.precision_scam,
                              "Recall (Scam)": ov.recall_scam, "Macro-F1": ov.macro_f1, "Weighted-F1": ov.weighted_f1,
                              "Macro-F1 95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(ov.macro_f1_ci_low, ov.macro_f1_ci_high)]})
    langs = base[base.model_id.isin(["char_lr", "word_lr", "char_svm", "word_svm"]) & base.subset.isin(
        ["English", "Roman Urdu", "Mixed/Code-Switched", "Roman Urdu + Mixed", "source:UCI", "source:Zenodo"])]
    t_lang = pd.DataFrame({"Model": langs.model_id, "Subset": langs.subset, "n": langs.n.astype(int), "n_groups": langs.n_groups.astype(int),
                           "Macro-F1": langs.macro_f1, "95% CI": [("n/a" if pd.isna(a) else f"[{a:.3f}, {b:.3f}]") for a, b in zip(langs.macro_f1_ci_low, langs.macro_f1_ci_high)],
                           "Status": langs.status.str.replace("descriptive only: ", "descr. only: ").str[:48]})
    cvt = cv[cv.subset.isin(["Overall", "English", "Roman Urdu", "Mixed/Code-Switched", "Roman Urdu + Mixed", "source:Zenodo", "source:UCI"])]
    t_cv = pd.DataFrame({"Model": cvt.model_id, "Subset": cvt.subset, "n": cvt.n.astype(int), "n_groups": cvt.n_groups.astype(int),
                         "Macro-F1": cvt.macro_f1, "95% CI": [("n/a" if pd.isna(a) else f"[{a:.3f}, {b:.3f}]") for a, b in zip(cvt.macro_f1_ci_low, cvt.macro_f1_ci_high)]})
    xlt = xl.copy()
    t_xl = pd.DataFrame({"Model": xlt.model_id, "Experiment": xlt.experiment, "n_test": xlt.n_test.astype(int), "n_groups": xlt.n_groups.astype(int),
                         "Macro-F1 / diff": xlt.macro_f1, "95% CI": [("n/a" if pd.isna(a) else f"[{a:.3f}, {b:.3f}]") for a, b in zip(xlt.macro_f1_ci_low, xlt.macro_f1_ci_high)]})
    r_all = rob[(rob.language_slice == "All") & (~rob.perturbation_type.str.startswith("ALL_TYPES")) & rob.model_id.isin(["char_lr", "word_lr", "char_svm", "word_svm"])]
    t_rob = r_all.pivot_table(index="perturbation_type", columns="model_id", values="degradation").reindex(list(C.PERTURBATION_TYPES)).reset_index()
    t_rob.columns.name = None
    n_pairs = r_all[r_all.model_id == "char_lr"].set_index("perturbation_type")[["n_pairs", "n_groups"]]
    t_rob.insert(1, "n_pairs (groups)", [f"{int(n_pairs.loc[p, 'n_pairs'])} ({int(n_pairs.loc[p, 'n_groups'])})" for p in t_rob.perturbation_type])
    rs = rob[(rob.language_slice == "All") & (~rob.perturbation_type.str.startswith("ALL_TYPES"))].groupby("model_id").degradation.mean().rename("mean_degradation_fixed_split")
    rcv = cvr[(cvr.language_slice == "All") & (~cvr.perturbation_type.str.startswith("ALL_TYPES"))].groupby("model_id").degradation.mean().rename("mean_degradation_group_cv")
    rcv_ru = cvr[(cvr.language_slice == "Roman Urdu + Mixed") & (~cvr.perturbation_type.str.startswith("ALL_TYPES"))].groupby("model_id").degradation.mean().rename("mean_degradation_group_cv_RU+Mixed")
    t_rsum = pd.concat([rs, rcv, rcv_ru], axis=1).reset_index().rename(columns={"index": "model_id"})
    cvru_t = cvr[(cvr.language_slice == "Roman Urdu + Mixed") & (cvr.model_id == "char_lr") & (~cvr.perturbation_type.str.startswith("ALL_TYPES"))]
    t_rcv_ru = pd.DataFrame({"Perturbation": cvru_t.perturbation_type, "n_pairs": cvru_t.n_pairs.astype(int), "n_groups": cvru_t.n_groups.astype(int),
                             "Degradation": cvru_t.degradation, "95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(cvru_t.degradation_ci_low, cvru_t.degradation_ci_high)],
                             "Flip rate": cvru_t.flip_rate})
    eff = ov[~ov.aug_trained][["model", "parameters", "n_tfidf_features", "model_size_mb", "train_seconds", "latency_single_ms_mean", "latency_single_ms_p95", "latency_batch_ms_per_msg"]]
    t_eff = eff.rename(columns={"model": "Model", "parameters": "Classifier weights", "n_tfidf_features": "TF-IDF features", "model_size_mb": "Size (MB)",
                                "train_seconds": "Train (s)", "latency_single_ms_mean": "Single msg mean (ms)", "latency_single_ms_p95": "Single msg p95 (ms)",
                                "latency_batch_ms_per_msg": "Batch (ms/msg)"})
    t_eff["Model"] = t_eff["Model"].str.replace("TFIDF-", "")
    fin = d["final"][["Model", "Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "English_F1", "Roman_Urdu_F1", "RomanUrdu_plus_Mixed_F1", "Parameters", "Model_Size", "Robustness_Degradation", "Status"]].copy()
    fin["Model"] = fin["Model"].str.replace("TFIDF-", "")
    fin["Status"] = fin["Status"].str.replace("Not executed in the current environment", "Not executed")
    for c in ["Roman_Urdu_F1", "RomanUrdu_plus_Mixed_F1"]:
        fin[c] = fin[c].astype(str).str.replace(r"N/A \(insufficient.*\)", "N/A (n=7)", regex=True)
    fin["RomanUrdu_plus_Mixed_F1"] = [v if not str(v).replace(".", "", 1).isdigit() else f"{v}*" for v in fin["RomanUrdu_plus_Mixed_F1"]]
    fin = fin.rename(columns={"RomanUrdu_plus_Mixed_F1": "RU+Mixed_F1*"})
    from .final_tables import error_counts_df

    ec = error_counts_df()
    t_err = pd.DataFrame({"Model": ec.Model.str.replace("TFIDF-", ""), "Errors (test originals + variants)": ec.Errors, "FP": ec.False_Positives, "FN": ec.False_Negatives})
    return {"errors": t_err, "overall": t_overall, "lang": t_lang, "cv": t_cv, "xl": t_xl, "rob": t_rob, "rsum": t_rsum, "rcv_ru": t_rcv_ru, "eff": t_eff, "final": fin}


REFS = """1. Almeida, T. A., Gomez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11)*.
2. Siddiq, G. *Roman Urdu Phishing SMS Dataset for Smishing Detection Research*. Zenodo. DOI 10.5281/zenodo.21810885. (Citation details as given in the provided dataset README; license unverified.)
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
4. Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
5. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
6. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020: System Demonstrations*.
7. Belinkov, Y., & Bisk, Y. (2018). Synthetic and natural noise both break neural machine translation. *ICLR 2018*.
8. Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR 2019*.
9. Lee, K., Ippolito, D., Nystrom, A., et al. (2022). Deduplicating training data makes language models better. *ACL 2022*.
10. Elangovan, A., He, J., & Verspoor, K. (2021). Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation. *EACL 2021*.
11. Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical Society: Series B*, 69(3), 369-390.
12. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
13. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
14. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*.
"""



def tx_sections() -> dict:
    """Transformer-dependent text. Everything is derived from executed-run files; otherwise explicit 'not executed' text."""
    from .make_results import CANONICAL_TRANSFORMERS, load_transformer_result

    runs = {t: load_transformer_result(t) for t in CANONICAL_TRANSFORMERS}
    done = {t: r for t, r in runs.items() if r is not None}
    cmd = ("```\npython -m src.train_transformer --model mbert --epochs 4 --batch-size 32 --robustness\n"
           "python -m src.train_transformer --model xlmr  --epochs 4 --batch-size 32 --robustness\n"
           "python -m src.make_results && python -m src.compare_models && python -m src.make_report\n```")
    out = {"n_done": len(done), "all_done": len(done) == len(CANONICAL_TRANSFORMERS)}
    missing = [CANONICAL_TRANSFORMERS[t] for t in CANONICAL_TRANSFORMERS if t not in done]
    from .final_tables import tx_values
    from .verified import SOURCE_LABEL, load_supplement

    sup = load_supplement()
    if not done and sup is not None:
        V = {t: tx_values(t) for t in CANONICAL_TRANSFORMERS}
        mb, xl = V["mbert"], V["xlmr"]
        base = pd.read_csv(R / "baseline_results.csv")
        ov = base[(base.subset == "Overall") & (~base.aug_trained)].sort_values("macro_f1", ascending=False)
        obs = ov.iloc[0]
        sel = ov.set_index("model_id").loc[json.load(open(C.MODELS_DIR / "best_model.json"))["model_id"]]
        st = {c["comparison"]: c for c in sup["statistical_comparisons"]}
        c_svm = next(c for k, c in st.items() if "selected" in k)
        c_mx = next(c for k, c in st.items() if "mBERT (overall" in k)
        c_en = next(c for k, c in st.items() if "English-only" in k)
        rob_all = pd.read_csv(R / "robustness_results.csv")
        rob_all = rob_all[(rob_all.language_slice == "All") & (~rob_all.perturbation_type.str.startswith("ALL_TYPES")) & (rob_all.model_id == "char_svm")]
        svm_deg = float(rob_all.degradation.mean())
        proto = sup["training_protocol"]
        d0 = proto["defaults_of_the_documented_command"]
        cfg = pd.DataFrame([{"Model": v["name"], "HF id": v["hf_model_id"], "Train n": v["train_samples"], "Best epoch": v["best_epoch"], "Max epochs": proto["max_epochs"],
                             "Early stop": f"val macro-F1, patience {proto['patience']}", "Batch": proto["batch_size"], "GPU": v["gpu"],
                             "Params": v["parameters"], "Size (MB)": v["model_size_mb"], "Train time (s)": v["train_seconds"]} for v in sup["transformers"].values()])
        met = pd.DataFrame([{"Model": v["name"], "Accuracy": v["accuracy"], "Precision (Scam)": v["precision"], "Recall (Scam)": v["recall"], "Macro-F1": v["macro_f1"],
                             "Weighted-F1": v["weighted_f1"], "Macro-F1 95% CI": f"[{v['macro_f1_ci95'][0]:.3f}, {v['macro_f1_ci95'][1]:.3f}]"} for v in sup["transformers"].values()])
        lat = pd.DataFrame([{"Model": v["name"], "GPU ms/msg (T4)": v["gpu_latency_ms_per_message"], "CPU ms/msg": v["cpu_latency_ms_per_message"], "Params": v["parameters"],
                             "Size (MB)": v["model_size_mb"], "Train (s)": v["train_seconds"]} for v in sup["transformers"].values()])
        cmp_md = (T / "model_comparison.md").read_text(encoding="utf-8") if (T / "model_comparison.md").exists() else ""
        cmp_s = lambda c: f"{c['difference']:+.4f} (95% CI [{c['ci95_low']:.4f}, {c['ci95_high']:.4f}]; the interval {'includes' if c['ci_includes_zero'] else 'excludes'} zero)"
        out.update(
            hdr="Transformer results: mBERT and XLM-R were trained and evaluated on a Google Colab NVIDIA Tesla T4 GPU (owner-verified summary; see Section 12 for provenance).",
            abs=(f"Char TF-IDF + Logistic Regression achieved the highest observed test macro-F1 ({obs.macro_f1:.4f}). Char TF-IDF + Linear SVM was the validation-selected CPU baseline "
                 f"and achieved {sel.macro_f1:.4f} test macro-F1. XLM-R was the best-performing transformer with {xl['macro_f1']:.4f} macro-F1 (95% CI [{xl['ci'][0]:.3f}, {xl['ci'][1]:.3f}]) "
                 f"versus {mb['macro_f1']:.4f} for mBERT (95% CI [{mb['ci'][0]:.3f}, {mb['ci'][1]:.3f}]); the paired XLM-R minus SVM difference is {cmp_s(c_svm)}, so no superiority of XLM-R over the selected SVM is claimed. "
                 f"Mean degradation under the project's synthetic perturbation suite was {svm_deg:.4f} (Char SVM), "
                 f"{mb['robustness']:+.4f} (mBERT) and {xl['robustness']:+.4f} (XLM-R)."),
            sec12=(f"mBERT (`bert-base-multilingual-cased`) and XLM-R (`xlm-roberta-base`) were fine-tuned and evaluated on a Google Colab **NVIDIA Tesla T4** GPU "
                   f"(`bash scripts/colab_run_all.sh`; protocol: at most {proto['max_epochs']} epochs, early stopping on validation macro-F1 with patience {proto['patience']}, batch size {proto['batch_size']}; "
                   f"documented-command defaults: lr {d0['learning_rate']}, max length {d0['max_length']}, weight decay {d0['weight_decay']}, warm-up {d0['warmup_ratio']}, seed {d0['seed']}, class-weighted cross-entropy). "
                   f"**Provenance:** the values below are the owner-verified results of the original Colab run (`results/verified_transformer_results.json`, {SOURCE_LABEL}). "
                   + REPRO_NOTE + " If the raw run files are copied into `experiments/results/`, `python -m src.qa_check` recomputes and checks every value.\n\n"
                   + md(cfg, ".4g") + "\n\nTest metrics (fixed split, 935 original messages, 778 groups; Precision/Recall refer to the Scam class):\n\n" + md(met, ".4f")
                   + "\n\nComparison with the CPU baselines (best CPU baseline = validation-selected Char TF-IDF + Linear SVM). Per-language transformer F1 values are not part of the verified summary; "
                     "the Roman Urdu/Mixed slice is descriptive anyway (5 independent test groups):\n\n" + cmp_md
                   + "\n\nPaired macro-F1 differences (cluster bootstrap over near-duplicate test groups; see `results/STATISTICAL_COMPARISONS.csv`):\n\n"
                   + f"* XLM-R vs selected Char SVM: {cmp_s(c_svm)} - no evidence of a difference; XLM-R is **not** claimed to outperform the SVM.\n"
                   + f"* XLM-R vs mBERT: {cmp_s(c_mx)} - the interval excludes zero but its lower bound is very close to zero.\n"
                   + f"* English-only XLM-R vs mBERT: {cmp_s(c_en)} - no evidence of an English-only difference.\n"),
            rob=("Transformers on the same synthetic perturbation suite (degradation = original macro-F1 - perturbed macro-F1; mean over the 10 perturbation types, paired test messages; lower is better): "
                 f"mBERT {mb['robustness']:+.4f}, XLM-R {xl['robustness']:+.4f}, versus {svm_deg:+.4f} for the selected Char SVM. "
                 "XLM-R shows the largest degradation of the three. mBERT's slightly negative value (perturbed macro-F1 marginally higher than original) is not interpreted as a robustness gain. "
                 "This describes robustness to the project's synthetic perturbation suite only, not real-world adversarial robustness; per-type transformer values require the raw robustness files."),
            eff="Transformer computational benchmark (Tesla T4 GPU and the Colab CPU; single-message latency, not an edge-device measurement; GPU/CPU software environment is recorded in the raw run JSON):\n\n" + md(lat, ".4g"),
            cmp="", disc5=(f"XLM-R ({xl['macro_f1']:.4f}) did not differ detectably from the validation-selected Char SVM ({cmp_s(c_svm)}) and was weaker than the highest observed baseline; mBERT ({mb['macro_f1']:.4f}) was clearly lower than XLM-R overall ({cmp_s(c_mx)}), "
                           f"while the English-only XLM-R vs mBERT difference is not detectable ({cmp_s(c_en)}). On a test set this small and template-dominated, differences whose interval includes zero should not be read as a ranking."),
            lim_extra="* Each transformer was trained once; variance across seeds is not reported. The transformer values are owner-verified results of the original Colab run and have not been independently reproduced locally (raw transformer artifacts were unavailable in the final workspace; 3 local QA checks are therefore PENDING, not failed); per-language transformer subsets and per-type transformer robustness need those raw files.",
            concl=(f"XLM-R was the best-performing transformer ({xl['macro_f1']:.4f} macro-F1) but did not detectably outperform the validation-selected Char TF-IDF + Linear SVM ({sel.macro_f1:.4f}); "
                   f"mBERT reached {mb['macro_f1']:.4f}. The ranking among the top models is not statistically established. The transformer metrics are owner-verified results of the original Colab run, not independently reproduced locally."),
            future="Run multiple seeds and the zero-shot language splits for the transformers; ")
        return out
    if not done:
        out.update(hdr="Transformer results: **not executed in the current environment** (see Section 12).",
                   abs="Multilingual transformers (mBERT, XLM-R) are implemented but **not executed** here (no GPU, no torch, model hub unreachable); no transformer numbers are reported.",
                   sec12="**Not executed in the current environment.** No mBERT/XLM-R result exists and none is reported. `src/train_transformer.py` implements the full pipeline (pretrained `bert-base-multilingual-cased` / `xlm-roberta-base`, GPU detection with CPU fallback, fixed seeds, attention masks, dynamic padding/truncation, label mapping shared with the baselines, early stopping on validation macro-F1, parameter counts, latency benchmarks, robustness evaluation, data-checksum and group-leakage guards). The pipeline was exercised on CPU with a tiny randomly initialised local model (pipeline test only; its outputs are marked `smoke_test` and are never used for results). To run (e.g. Colab GPU):\n" + cmd,
                   rob="", eff="No GPU or transformer measurements exist.", cmp="", disc5="Whether multilingual transformers transfer is the key open question and requires the pending GPU runs.",
                   lim_extra="* Transformers, and therefore GPU/CPU transformer latency, were **not executed** in this snapshot.",
                   concl="Transformer results remain pending.", future="Run mBERT/XLM-R (with robustness and zero-shot splits) on a GPU; ")
        return out
    first = next(iter(done.values()))
    env = first["environment"]
    cfg_rows = []
    sub_rows = []
    for t, r in done.items():
        c, e = r["config"], r["environment"]
        cfg_rows.append({"Model": r["model"], "HF id": r["hf_model_id"], "Epochs run / max": f"{len(r['history'])} / {c['epochs']}", "Best epoch": r["best_epoch"],
                         "Batch": c["batch_size"], "LR": c["lr"], "Max len": c["max_length"], "Seed": c["seed"], "Train n": r["n_train"],
                         "Params": r["parameters"]["total"], "Train time (s)": round(r["train_seconds"], 1), "Device": e.get("gpu_name") or r["device"]})
        for x in r["test_subsets"]:
            if x["subset"] in ("Overall", C.LANG_ENGLISH, C.LANG_ROMAN_URDU, C.LANG_MIXED, "Roman Urdu + Mixed"):
                ok = x.get("macro_f1") == x.get("macro_f1")
                sub_rows.append({"Model": t, "Subset": x["subset"], "n": int(x["n"]), "n_groups": int(x.get("n_groups", 0)),
                                 "Macro-F1": x.get("macro_f1") if ok else float("nan"),
                                 "95% CI": f"[{x['macro_f1_ci_low']:.3f}, {x['macro_f1_ci_high']:.3f}]" if ok else "n/a",
                                 "Precision (Scam)": x.get("precision_scam"), "Recall (Scam)": x.get("recall_scam"),
                                 "Status": str(x.get("status", ""))[:60]})
    cmp_md = (T / "model_comparison.md").read_text(encoding="utf-8") if (T / "model_comparison.md").exists() else ""
    pv = pd.read_csv(R / "transformer_vs_baseline.csv") if (R / "transformer_vs_baseline.csv").exists() else pd.DataFrame()
    sent = []
    for _, r in pv.iterrows():
        if r["subset"] == "Overall":
            sent.append(f"* {r['A']} vs {r['B']}: macro-F1 difference {r['macro_f1_diff_A_minus_B']:+.3f} (95% cluster-bootstrap CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]; the interval {'excludes' if r['ci_excludes_zero'] else 'includes'} zero; McNemar p={r['mcnemar_p_optimistic_clustered']:.3g}, optimistic because test messages are clustered).")
    pv_md = pv.assign(**{c: pv[c].round(4) for c in ["macro_f1_diff_A_minus_B", "ci_low", "ci_high"]}).fillna("").to_markdown(index=False) if len(pv) else ""
    rcmp = pd.read_csv(T / "robustness_comparison.csv") if (T / "robustness_comparison.csv").exists() else pd.DataFrame()
    rob_md = ("Degradation by perturbation type, best CPU baseline vs transformers (fixed split, 'All' slice; positive = worse; transformer rows are from executed runs):\n\n"
              + rcmp.to_markdown(index=False, floatfmt=".4f") + "\n") if len(rcmp) else ""
    lat_rows = []
    for t, r in done.items():
        for dev, v in r["latency_computational_benchmark"].items():
            lat_rows.append({"Model": t, "Device": dev, "Params": r["parameters"]["total"], "Size (MB)": round(r["model_size_mb"], 1), "Train (s)": round(r["train_seconds"], 1),
                             "Single msg mean (ms)": round(v["latency_single_ms_mean"], 2), "Single msg p95 (ms)": round(v["latency_single_ms_p95"], 2), "Batch (ms/msg)": round(v["latency_batch_ms_per_msg"], 3)})
    gpu = env.get("gpu_name") or "CPU"
    out.update(
        hdr=f"Transformer results: executed for {', '.join(done)} (GPU: {gpu}); " + (f"**not executed for: {', '.join(missing)}**" if missing else "both models executed") + " (see Section 12).",
        abs=f"Fine-tuned mBERT and XLM-R results are reported for {', '.join(done)} (executed on {gpu}); see Sections 12, 14 and 18 for metrics, intervals and the paired comparison with the best CPU baseline."
            + (f" Not executed: {', '.join(missing)}." if missing else ""),
        sec12=("Executed runs (configuration, training time and environment are read from `experiments/results/transformer_<tag>.json`; "
               f"Python {env.get('python')}, torch {env.get('torch')}, CUDA {env.get('cuda_version')}, transformers {env.get('transformers')}, GPU {gpu}"
               f"{' with ' + str(env.get('gpu_memory_gb')) + ' GB' if env.get('gpu_memory_gb') else ''}):\n\n" + md(pd.DataFrame(cfg_rows), ".4g")
               + "\n\nTest metrics on the fixed split (originals). **Roman Urdu / Mixed subsets are descriptive only (5 independent test groups; Roman Urdu alone has 7 messages):**\n\n"
               + md(pd.DataFrame(sub_rows), ".3f")
               + ("\n\nNot executed: " + ", ".join(missing) + ".\n" if missing else "\n")
               + "\nComparison with the best CPU baseline (selected on validation macro-F1):\n\n" + cmp_md
               + ("\nPaired differences (cluster bootstrap over near-duplicate groups; McNemar is secondary):\n\n" + "\n".join(sent) + "\n\n" + pv_md if len(pv) else "")),
        rob=rob_md, eff=("Transformer computational benchmark (single-message latency on the stated device; not an edge-device measurement):\n\n" + md(pd.DataFrame(lat_rows), ".3g")),
        cmp="", disc5="Transformer vs baseline differences are reported with paired cluster-bootstrap intervals in Section 12; on a test set this small and template-dominated, differences whose interval includes zero should not be read as a ranking.",
        lim_extra=(f"* Not executed: {', '.join(missing)}." if missing else ""),
        concl=("Transformer results are reported for " + ", ".join(done) + "." if not missing else "Transformer results are reported for " + ", ".join(done) + "; " + ", ".join(missing) + " remain pending."),
        future="Run multiple seeds and the zero-shot language splits for the transformers; ")
    return out


def report(d: dict, t: dict, n: dict) -> str:
    tx = tx_sections()
    b, top, cvo, cvru, sp = n["b"], n["top"], n["cvo"], n["cvru"], n["sp"]
    cfg_env = d["cfg"]["environment"]
    g = d["groups"].set_index("source")
    zen_groups, zen_max = int(g.loc["Zenodo Roman-Urdu phishing", "n_groups"]), int(g.loc["Zenodo Roman-Urdu phishing", "max_group"])
    val = d["val"]["stats"]["residual_similarity_test_vs_train"]
    xl = d["xl"]
    cdiff = xl[xl.experiment.str.startswith("C-diff")]
    zs = xl[xl.experiment.str.contains("zero-shot")]
    return f"""# MARS-NLP: Robust Multilingual Scam Message Classification with Character-Level and Transformer-Based NLP

**Author:** Muhammad Rizwan  \n**Affiliation:** Independent Researcher  \n**Email:** rizwan01107@gmail.com

*Generated by `python -m src.make_report` from executed experiments. {tx['hdr']}*

## 1. Abstract
We study scam message classification across English, Roman Urdu and code-switched SMS using two public corpora (UCI SMS Spam Collection; a Roman-Urdu phishing SMS dataset from Zenodo). A dataset audit showed that the originally supplied corpus files were empty, that the Zenodo data are heavily templated ({zen_groups} near-duplicate groups for 1,000 messages), and that no real Urdu-script messages exist in the sources. We therefore split at the level of near-duplicate groups, generate synthetic noise perturbations only from real originals (variants share their original's split), and report 95% cluster-bootstrap intervals. {tx['abs']} The Roman Urdu/Mixed test slice rests on only 5 independent test groups and is descriptive. The perturbation suite is synthetic and is not equivalent to naturally occurring adversarial messages.

## 2. Introduction
SMS scams and unsolicited spam are a persistent security problem, and South-Asian users routinely write Urdu in Latin script (Roman Urdu) mixed with English. Spelling in Roman Urdu is unstandardised, which makes both lexical and learned detectors potentially brittle. Public resources for this setting are small and heterogeneous, so careful evaluation design matters at least as much as model choice.

## 3. Problem Statement
Binary classification of an SMS message as *Scam* (scam/spam/phishing) or *Legitimate* (ham), evaluated overall and per language, and under controlled text perturbations that mimic spelling variation, noise and code-switching.

## 4. Research Question
(RQ1) How well do TF-IDF baselines detect scam/spam in English, Roman Urdu and code-switched SMS when evaluation respects near-duplicate/template structure? (RQ2) How much does performance degrade under controlled perturbations? (RQ3) Does training on one language transfer to the other (TF-IDF zero-shot)? (RQ4) How do fine-tuned mBERT and XLM-R compare with the CPU baselines in accuracy, robustness to the synthetic perturbation suite and computational cost?

## 5. Motivation
Reported accuracy on SMS spam corpora is often near-ceiling, partly because of duplicated and templated messages (cf. work on duplicated data and leakage in NLP evaluation [9, 10]). A reproducible pipeline that exposes this structure, quantifies uncertainty honestly and separates what was and was not run is a prerequisite for credible multilingual robustness claims.

## 6. Related Work
The UCI SMS Spam Collection [1] is a standard English benchmark. Pretrained multilingual encoders - mBERT, the multilingual variant of BERT [3], and XLM-R [4] - are fine-tuned here as comparison models; their behaviour on romanised, non-standard orthography is an empirical question. Sensitivity of neural text models to noise is documented in [7, 8]; our perturbations are random synthetic noise, not optimised adversarial attacks. Duplicate and leakage effects on evaluation are discussed in [9, 10]. For uncertainty with clustered samples we use a cluster bootstrap [11, 12]; paired comparisons use McNemar's test [13] only as a secondary check. Explanations use linear decompositions and LIME [14] for the TF-IDF baselines. Implementation relies on scikit-learn [5] and Hugging Face Transformers [6].

## 7. Dataset
| Source | Raw rows | After cleaning + exact dedup | License |
|---|---|---|---|
| SMS Spam Collection (UCI) [1] | {b['uci_raw_rows']} | {b['exact_dedup_per_source'][C.SRC_UCI]['n_after']} | {C.LICENSE_UCI} |
| Roman Urdu Phishing SMS (Zenodo) [2] | {b['zenodo_raw_rows']} | {b['exact_dedup_per_source'][C.SRC_ZENODO]['n_after']} | {C.LICENSE_ZENODO} |

Processed corpus: **{b['n_originals']} originals** and **{b['n_perturbed']} perturbed variants** ({b['n_total']} rows). Originals by split: train {sp['train']}, validation {sp['validation']}, test {sp['test']}. Language labels (heuristic, see Section 9): {b['language_counts_originals']}. **There are no Urdu-script messages**, so Urdu cannot be evaluated; the Urdu columns in all result tables are N/A. Real Urdu-script data would need to come from an additional, licensed source.

## 8. Dataset Audit
Full audit: `reports/data_audit.md`, `reports/data_audit.csv`. Key findings (all computed):
* The three originally provided files were an **empty scaffold** (0 data rows in CSV and XLSX; the README states 0 imported rows). The raw public files supplied afterwards are authoritative; the scaffold is kept unchanged in `data/raw/scaffold/`.
* UCI: 5,574 rows, 4,827 ham / 747 spam; 403 exact duplicate messages in the raw text (414 after cleaning), none with conflicting labels; further near-duplicates beyond exact matches exist.
* Zenodo: 1,000 rows, 595 legitimate / 405 phishing, 16 message categories, no exact duplicates, but only {zen_groups} near-duplicate template groups (largest: {zen_max} messages). Its license field is absent from the file and **unverified**.
* Source and language are partly confounded (UCI is English; Zenodo is Roman Urdu *and* English bank/OTP templates), and the meaning of the positive class differs (UCI spam = unsolicited marketing; Zenodo phishing = fraud).
* Validation (`python -m src.validate_data`): {len(d['val']['checks'])} checks, 0 critical failures; one warning (14 perturbed texts coincide with another original's text). Residual similarity of test messages to their most similar training message: mean {val['mean_max_sim']:.2f}, share >= 0.80: {val['share_ge_0.80']:.3f}.

## 9. Preprocessing
`message` always keeps the raw text; `cleaned_message` applies Unicode NFKC, HTML-entity decoding, zero-width/control-character removal (ZWNJ/ZWJ preserved for Urdu) and whitespace normalisation, with conservative mojibake repair. Case, punctuation and spelling variation are retained (TF-IDF lower-cases internally). Labels are normalised to Scam/Legitimate. Exact duplicates are removed (keep first; none conflicted). **Near-duplicates are not removed but grouped** (character 3-5-gram TF-IDF cosine >= {C.NEAR_DUP_THRESHOLD}, single linkage) -> {b['n_groups']} groups. **Language** is assigned by a transparent function-word heuristic (not a trained language-ID model; spot checks in the audit show false positives on short English texts), not from the source name.

**Leakage prevention.** Whole groups are assigned to train/validation/test (70/15/15 by message count; stratified by source and label, seed {C.SEED}); each perturbed variant inherits its original's split. Verified by `src.validate_data` and unit tests: no group in more than one split, no variant outside its original's split, no exact text overlap across splits.

## 10. Experimental Setup
Environment: Python {cfg_env['python']}, scikit-learn {cfg_env['scikit_learn']}, {cfg_env['cpu_count']} CPU core(s), torch: {cfg_env['torch']}. Hyper-parameters are chosen on the validation originals (macro-F1); the test split is used once. Metrics: accuracy, Scam-class precision/recall, macro-F1 (primary, because of class imbalance), weighted-F1. Intervals: 95% percentile cluster bootstrap over near-duplicate groups (1,000 resamples). Subsets with fewer than {C.MIN_EVAL_N} samples or fewer than {C.MIN_EVAL_PER_CLASS} per class are not reported; subsets with fewer than {C.MIN_EVAL_GROUPS} independent groups are flagged *descriptive only*. Because the fixed split places only a handful of Zenodo groups in the test set, we also report 5-fold stratified **group** cross-validation over all originals (hyper-parameters not re-tuned inside folds).

**Perturbations** (random noise, seeded, derived only from real originals; `source_type = augmented_from_real`): {', '.join(C.PERTURBATION_TYPES)}. They are applied only when applicable (e.g. Roman-Urdu spelling variation only for Roman Urdu/Mixed messages). `casing_variation` is a no-op for models whose vectoriser lower-cases, which yields exactly zero degradation by construction. **Robustness metric (our own definition, not a standard):** `degradation = original_f1 - perturbed_f1`, with macro-F1 computed on the same underlying messages (paired).

## 11. Baseline Models
TF-IDF (word 1-2-grams or character 2-5-grams) with Logistic Regression, Linear SVM, or Multinomial Naive Bayes; class-weight "balanced" for LR/SVM. Each also trained on originals + perturbed training variants ("aug-trained").

Fixed split, test originals (n = {sp['test']}):

{md(t['overall'])}

Per-language / per-source slices (fixed split):

{md(t['lang'])}

5-fold group CV (out-of-fold predictions over all {b['n_originals']} originals):

{md(t['cv'])}

## 12. Transformer Models
{tx['sec12']}

## 13. Cross-Lingual Evaluation
Language groups used: English vs Roman Urdu + Mixed (Roman Urdu alone has only 7 test messages). Experiment A (combined -> per language), B (zero-shot transfer), C (combined vs language-specific, paired cluster-bootstrap difference). Only three representative baseline configurations were run. **Caveats:** language is confounded with source and label semantics, and the Roman Urdu + Mixed test slice has 5 independent groups; results measure template/domain transfer at least as much as language transfer.

{md(t['xl'])}

Zero-shot transfer fails with TF-IDF features (macro-F1 between {zs.macro_f1.min():.2f} and {zs.macro_f1.max():.2f}): the vocabularies barely overlap and the positive-class definitions differ. This says nothing about multilingual transformers, which remain untested. Combined vs language-specific training shows differences of {cdiff.macro_f1.min():+.3f} to {cdiff.macro_f1.max():+.3f} macro-F1 (see the `C-diff` rows), i.e. no evidence that pooling languages helps or hurts on this data.

## 14. Robustness Evaluation
Degradation by perturbation (fixed split, orig-trained models; positive = worse; n_pairs and independent groups for `char_lr` shown):

{md(t['rob'], '.4f')}

Mean degradation over the 10 types (negative = perturbed F1 *higher*, which reflects instability, not a robustness gain):

{md(t['rsum'], '.4f')}

Group-CV robustness for Roman Urdu + Mixed (`char_lr`; 22 groups; out-of-fold; descriptive):

{md(t['rcv_ru'], '.3f')}

Observations: (i) for the character-level LR/SVM models, mean degradation is below 0.01 macro-F1 and most paired intervals include or touch zero (word-level SVM is the most affected, up to 0.026 for word-spacing changes); (ii) the largest effects come from character-level edits (substitution, deletion) and punctuation; (iii) all intervals for Roman-Urdu-specific perturbations on the fixed split rest on 5 groups and are uninformative; (iv) casing changes cannot matter for these lower-casing vectorisers. These perturbations are mild random noise; the results do **not** show robustness to optimised adversarial attacks.

{tx['rob']}

## 15. Explainability
Linear models: exact additive contributions (weight x TF-IDF); LIME [14] as a model-agnostic cross-check (`reports/tables/explain_examples.csv`; figures `explain_*.png`) for one true positive, true negative, false positive and false negative per model. For the word-level LR the most Scam-indicating features are call, txt, free, text, claim, mobile, www and price tokens such as 150p; Legitimate features are first-person pronouns and conversational words. For the character-level LR the top Scam features are digit n-grams (phone numbers and premium-rate/price patterns) - a **dataset artefact** worth noting: these n-grams reflect UK SMS marketing conventions in the UCI data and would not transfer to other regions. Attributions describe what the *model* used; they are not causal evidence about why a message is a scam. Transformer interpretability methods (e.g. SHAP, Integrated Gradients) were not performed.

## 16. Error Analysis
Error counts below are **per model** and count misclassifications over the test split's original messages **and** their perturbed variants. They must not be summed across models or presented as one model's FN/FP totals. `experiments/results/error_analysis.csv` lists every baseline misclassification with columns message, language, true_label, predicted_label, confidence, error_category, model; summaries in `reports/tables/error_summary.csv`.\n\n{md(t['errors'], '.0f')}\n\nFP/FN splits for mBERT and XLM-R need the raw transformer prediction files (see `results/ERROR_COUNTS.csv`).\n\n For the best character-level LR on the 935 test originals there are 12 errors (11 false negatives, 1 false positive), all on English UCI messages; the false negatives are mostly non-prototypical spam (adult-chat solicitations, a flower-delivery promotion, an "I missed your call" message containing a phone number) and several have high confidence. Its single false positive is a short message ("Now am free call me pa.", 23 characters) at confidence 0.56. For word-level LR, the 17 false positives on Roman Urdu/Mixed messages all come from **one template** ("Dear Consumer, ... ka bill ... PKR ...", legitimate utility-bill notices) - a template-level, not message-level, error, which illustrates why the fixed split's Roman Urdu numbers are fragile. The same single-template effect explains most of the apparent damage from perturbation-augmented training to word-level LR/SVM on the fixed split (for word LR + aug, 46 of 48 false positives are Zenodo legitimate messages from the 4 Zenodo test groups); the group-CV analysis does not include aug-trained models, so this effect is not separated from noise.

## 17. Efficiency Analysis
Computational benchmark on this machine ({cfg_env['cpu_count']} CPU core, sandbox environment; **not** an edge-device benchmark). Latency = wall-clock per call on test messages; parameters = learned classifier weights (vocabulary/idf vectors are additional).

{md(t['eff'], '.3f')}

All original-trained baselines are under 1 MB (largest 0.79 MB) with mean single-message latency between 0.47 and 0.97 ms on this machine. {tx['eff']}

## 18. Results
Final table (`experiments/results/final_results.csv`; unexecuted transformer rows are marked, never filled):

{md(t['final'], '.4f')}

Requested comparison table (best CPU baseline vs transformers):

{md(pd.read_csv(T / 'model_comparison.csv', keep_default_na=False)[['Model', 'Macro-F1', 'Precision', 'Recall', 'English-F1', 'Roman Urdu/Mixed-F1', 'Robustness Degradation', 'Status']] if (T / 'model_comparison.csv').exists() else pd.DataFrame())}

\\* descriptive only: the Roman Urdu + Mixed test slice has 149 messages but only 5 independent near-duplicate groups (intervals such as [0.5, 1.0]); Roman_Urdu_F1 is N/A because only 7 Roman Urdu test messages exist.

Best fixed-split model: {top.model.replace('TFIDF-', '')} (macro-F1 {fmt_ci(top)}); best in group CV: {cvo.model_id} ({cvo.macro_f1:.3f}). The differences among the best LR/SVM variants are small relative to their intervals and should not be read as a ranking.

## 19. Discussion
1. **Near-ceiling performance, mostly via template/lexical cues.** Linear (LR/SVM) TF-IDF models reach about 0.97-0.98 macro-F1 under group-aware evaluation (Naive Bayes is clearly worse, 0.88-0.89 on the fixed split), and the top features are surface cues (words like *free*, *call*; digit patterns). That is a statement about these corpora, not about general scam detection.
2. **The fixed split cannot evaluate Roman Urdu.** The test slice has 149 Roman Urdu/Mixed messages but only 5 independent groups; point estimates of 1.0 carry intervals as wide as [0.5, 1.0]. Group CV (22 groups) gives {', '.join(f"{i}: {cvru.loc[i, 'macro_f1']:.3f}" for i in cvru.index)} for Roman Urdu + Mixed, still descriptive.
3. **Word vs character features.** Character n-grams are slightly better in group CV, but the larger fixed-split gap for word LR is caused by one template group; we do not claim character n-grams are superior.
4. **Noise robustness is good for mild perturbations,** but this is a weak test; stronger tests (optimised attacks, real misspelling corpora) are future work.
5. **Zero-shot TF-IDF transfer fails;** {tx['disc5']}

## 20. Limitations
* The dataset is domain-specific (English SMS marketing spam and Pakistani bank/utility-style Roman-Urdu phishing templates); results do not generalise to other message types or regions.
* The perturbations are synthetic random noise. They are not equivalent to naturally occurring adversarial messages and do not measure real-world adversarial robustness.
* Roman Urdu/Mixed evidence is limited: the test split's Roman Urdu + Mixed slice (149 messages) comes from only **5 independent near-duplicate groups** (Roman Urdu alone: 7 messages), so every number there is descriptive.
* There is no Urdu-script benchmark: the sources contain no Urdu-script messages and Urdu is not evaluated.
* Multi-seed variance was not estimated for every model (each transformer was trained once; the baselines are deterministic given the seed).
* Hyperparameter search was not exhaustive (small grids for the baselines; fixed a priori settings for the transformers).
* Computational measurements (latency, training time) depend on the Tesla T4 / software environment and a single CPU core; they are not edge-device benchmarks.
* Transformer interpretability methods such as SHAP and Integrated Gradients were not performed.
* Language labels are heuristic; source, language and label semantics are confounded (UCI marketing spam vs Zenodo phishing).
* The Zenodo license is unverified; the UCI CC BY 4.0 statement was not re-verified offline.
* Aug-trained models were not included in group CV or the cross-lingual experiments; the augmentation effect on the fixed split is dominated by a single template.
{tx['lim_extra']}

## 21. Ethical Considerations
Messages come from public research datasets and may contain phone numbers or other identifiers; raw data are not redistributed in this repository. Dataset licenses and the Zenodo license uncertainty are documented. A detector can produce false positives (blocking legitimate bank/utility messages: see the single-template FP example) and false negatives; the demo is labelled a research prototype and must not be used for financial or security decisions. Publishing perturbation recipes could in principle help evasion, but they are generic noise operations and no attack optimisation is provided.

## 22. Conclusion
Within the limits above, character-level TF-IDF models are strong, cheap baselines on these corpora under group-aware evaluation. {tx['concl']} The more important outcome is methodological: the audit exposed an empty scaffold, a templated Roman-Urdu source and absent Urdu-script data, and the pipeline reports those limits explicitly.

## 23. Future Work
{tx['future']}add licensed real Urdu-script and larger Roman Urdu data (more than 5 independent test groups); evaluate source-balanced and cross-source protocols; add optimised adversarial attacks and real misspelling corpora; calibrate probabilities; re-tune within group-CV folds.

## 24. References
{REFS}
"""


def readme_text(d: dict, n: dict) -> str:
    from .final_tables import final_results_df, robustness_df
    from .verified import load_supplement

    b = n["b"]
    sp = b["split_counts_originals"]
    fin = final_results_df()
    fin_md = fin[["Model", "Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "Macro_F1_CI95"]].rename(
        columns={"Macro_F1": "Macro-F1", "Weighted_F1": "Weighted-F1", "Macro_F1_CI95": "Macro-F1 95% CI"}).to_markdown(index=False, floatfmt=".4f")
    rob = robustness_df()
    rob = rob[rob.Level == "mean over 10 perturbation types"][["Model", "Degradation"]]
    rob_md = rob.to_markdown(index=False, floatfmt=".4f")
    sup = load_supplement()
    qa = sup["qa_owner_verified"] if sup else {"PASS": "n/a", "FAIL": "n/a", "PENDING": "n/a", "SKIP": "n/a", "pytest_passed": "n/a"}
    g = d["groups"].set_index("source")
    zen_groups = int(g.loc["Zenodo Roman-Urdu phishing", "n_groups"])
    ov = d["base"][(d["base"].subset == "Overall") & (~d["base"].aug_trained)].sort_values("macro_f1", ascending=False).iloc[0]
    sel_id = json.load(open(C.MODELS_DIR / "best_model.json"))["model_id"]
    sel = d["base"][(d["base"].subset == "Overall") & (d["base"].model_id == sel_id)].iloc[0]
    xl = fin[fin.Model == "XLM-R"].iloc[0]
    return f"""# {TITLE}

**Author:** Muhammad Rizwan  
**Affiliation:** Independent Researcher  
**Email:** rizwan01107@gmail.com

## Project overview
MARS-NLP classifies SMS-style messages as *Scam* (scam/spam/phishing) or *Legitimate* in English, Roman Urdu and code-switched text. It compares character- and word-level TF-IDF baselines (logistic regression, linear SVM, naive Bayes) with fine-tuned mBERT and XLM-R under **group-aware evaluation** and a synthetic perturbation suite. The full manuscript is [`reports/MARS_NLP_Final_Manuscript.md`](reports/MARS_NLP_Final_Manuscript.md) (PDF: [`reports/MARS_NLP_Final_Manuscript.pdf`](reports/MARS_NLP_Final_Manuscript.pdf)); it is a research manuscript prepared for submission and has **not been peer reviewed or published**. All numbers in this README come from saved result files; [`reports/FINAL_REPRODUCIBILITY_REPORT.md`](reports/FINAL_REPRODUCIBILITY_REPORT.md) lists the source file of every value.

## Research motivation and objective
Scam messages exploit urgency and impersonation, and in South Asia they are often written in Roman Urdu (Urdu in Latin script, with no standard spelling) mixed with English. Public SMS corpora contain duplicated and templated messages, so naive splits leak and report near-ceiling scores. The objective is to measure how well TF-IDF baselines and multilingual transformers detect scam messages when evaluation respects near-duplicate/template structure, how they behave under a synthetic perturbation suite, and what the evidence does and does not support.

## Key contributions
* An audited corpus of {b['n_originals']:,} original messages with a **group-level** train/validation/test split; each of the {b['n_perturbed']:,} synthetic perturbed variants stays in its source message's split (checksum- and fingerprint-verified).
* A ten-type synthetic perturbation suite and a paired robustness measure (degradation = original macro-F1 - perturbed macro-F1).
* A comparison of six TF-IDF classifiers with mBERT and XLM-R using cluster-bootstrap confidence intervals and paired comparisons.
* Transparent reporting of limits: five independent Roman Urdu/Mixed test groups, no Urdu-script data, and transformer metrics that are owner-verified results of the original Colab run.

## Dataset statistics
| Item | Value |
|---|---|
| Original messages | {b['n_originals']:,} (UCI SMS Spam Collection + Zenodo Roman-Urdu phishing SMS) |
| Synthetic perturbed variants | {b['n_perturbed']:,} |
| Train / validation / test originals | {sp['train']:,} / {sp['validation']} / {sp['test']} |
| Test message groups | 778 (English 782 messages; Roman Urdu + Mixed 149 messages from only **5** independent groups) |
| Near-duplicate groups (all originals) | {b['n_groups']:,} (Zenodo: {zen_groups}) |
| Urdu-script messages | 0 - no Urdu-script benchmark is available |

Sources: UCI SMS Spam Collection (Almeida et al., 2011; CC BY 4.0 per UCI page) and the Roman Urdu Phishing SMS Dataset (Zenodo, DOI 10.5281/zenodo.21810885; **license unverified**). Data are not committed; see `data/README.md`.

## Methodology
### Data splitting
Exact duplicates are removed; near-duplicate/templated messages are merged into groups (character 3-5-gram TF-IDF cosine >= {C.NEAR_DUP_THRESHOLD}); whole groups are assigned to train/validation/test (70/15/15, stratified by source and label, seed {C.SEED}). Perturbed variants inherit their original's split. Verified: no group in more than one split, no exact text overlap across splits, processed-CSV SHA-256 checksum and split fingerprints (`data/processed/`) unchanged.

### Synthetic perturbation strategy
Ten seeded, synthetic noise perturbations derived only from real originals: spelling variation, character deletion, character substitution, word spacing variation, punctuation variation, casing variation, informal abbreviation, Roman Urdu spelling variation, code switching, minor noise. They are **not** equivalent to naturally occurring adversarial messages.

### Models
TF-IDF (character 2-5-grams or word 1-2-grams) with Logistic Regression, Linear SVM or Multinomial Naive Bayes (hyper-parameters chosen on validation macro-F1); mBERT (`bert-base-multilingual-cased`) and XLM-R (`xlm-roberta-base`) fine-tuned on a Google Colab NVIDIA Tesla T4 (max 4 epochs, early stopping on validation macro-F1, patience 2, batch size 32).

## Model comparison (final results)
Fixed test split, 935 original messages (Precision/Recall = Scam class; CI = 95% cluster bootstrap). Source: `results/FINAL_RESULTS.csv`.

{fin_md}

* **Char TF-IDF + Logistic Regression achieved the highest observed test macro-F1 ({ov.macro_f1:.4f}).**
* **Char TF-IDF + Linear SVM was the validation-selected CPU baseline and achieved {sel.macro_f1:.4f} test macro-F1.**
* **XLM-R was the best-performing transformer with {xl.Macro_F1:.4f} macro-F1.** Its paired difference to the selected SVM is -0.0019 (95% CI [-0.0188, 0.0133], includes zero), so no superiority is claimed; XLM-R vs mBERT is +0.0950 (95% CI [0.0017, 0.2236]); English-only XLM-R vs mBERT +0.0086 (95% CI [-0.0069, 0.0254], includes zero). See `results/STATISTICAL_COMPARISONS.csv`.

![Figure 1. Macro-F1 comparison](figures/model_macro_f1_comparison.png)

## Robustness evaluation
`degradation = original macro-F1 - perturbed macro-F1` (mean over the 10 synthetic perturbation types, paired test messages; lower is better). This is robustness to the project's synthetic perturbation suite, not real-world adversarial robustness. Source: `results/ROBUSTNESS_RESULTS.csv`.

{rob_md}

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
**Original owner-verified Colab QA (after both transformers finished): {qa['PASS']} PASS, {qa['FAIL']} FAIL, {qa['PENDING']} PENDING, {qa['SKIP']} SKIP**; {qa['pytest_passed']} pytest tests passed; no group overlap across splits; variants stay in their original split; zero exact cross-split text overlap; processed-CSV checksum verified; split fingerprints unchanged; mBERT and XLM-R trained/evaluated on a Tesla T4; metrics recomputed from raw predictions (70 values checked); robustness metrics traced; prediction smoke tests, Streamlit AppTest and headless startup passed.

{local_qa_line()} The 3 PENDING checks:

{pending_table_md()}

{PENDING_NOTE}

{REPRO_NOTE}

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
If you use this work, cite it as software/manuscript (see `CITATION.cff`): Muhammad Rizwan, *{TITLE}*, Independent Researcher (unpublished manuscript). Please also cite the data sources - Almeida, Gomez Hidalgo & Yamakami (2011), DocEng '11; Siddiq, *Roman Urdu Phishing SMS Dataset for Smishing Detection Research*, Zenodo, DOI 10.5281/zenodo.21810885 - and the core methods: Devlin et al. (2019), NAACL-HLT (BERT); Conneau et al. (2020), ACL (XLM-R); Pedregosa et al. (2011), JMLR (scikit-learn). Code license: MIT (code only; the datasets keep their own terms).
"""


def main() -> None:
    d = load()
    n = build_numbers(d)
    t = tables(d)
    (C.REPORTS_DIR / "MARS_NLP_Research_Report.md").write_text(report(d, t, n), encoding="utf-8")
    (C.ROOT_DIR / "README.md").write_text(readme_text(d, n), encoding="utf-8")
    print("Wrote README.md and reports/MARS_NLP_Research_Report.md")


if __name__ == "__main__":
    main()
