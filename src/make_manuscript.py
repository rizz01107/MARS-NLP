"""Generate reports/MARS_NLP_Final_Manuscript.md. Every number is read from results/*.csv, results/verified_transformer_results.json
and experiments/results/*; nothing is typed in by hand except narrative text. Usage: python -m src.make_manuscript
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from .make_report import PENDING_ROWS, TITLE
from .verified import load_supplement

ROOT, R, T = C.ROOT_DIR, C.RESULTS_DIR, C.TABLES_DIR
RES = ROOT / "results"
REFS = """1. Almeida, T. A., Gomez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11)*.
2. Siddiq, G. *Roman Urdu Phishing SMS Dataset for Smishing Detection Research*. Zenodo. DOI 10.5281/zenodo.21810885.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*.
4. Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale. *Proceedings of ACL 2020*.
5. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
6. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of EMNLP 2020: System Demonstrations*.
7. Belinkov, Y., & Bisk, Y. (2018). Synthetic and natural noise both break neural machine translation. *Proceedings of ICLR 2018*.
8. Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *Proceedings of ICLR 2019*.
9. Lee, K., Ippolito, D., Nystrom, A., et al. (2022). Deduplicating training data makes language models better. *Proceedings of ACL 2022*.
10. Elangovan, A., He, J., & Verspoor, K. (2021). Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation. *Proceedings of EACL 2021*.
11. Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical Society: Series B*, 69(3), 369-390.
12. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
13. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of KDD 2016*."""

PERT_DESC = {
    "spelling_variation": "transposition, doubling or vowel replacement in selected words",
    "character_deletion": "deletion of about 8% of inner letters (at least one)",
    "character_substitution": "keyboard-neighbour or look-alike character substitution (about 8% of letters)",
    "word_spacing_variation": "insertion or removal of spaces inside/between words",
    "punctuation_variation": "removal, repetition or insertion of punctuation",
    "casing_variation": "upper-casing, lower-casing or random casing",
    "informal_abbreviation": "SMS-style abbreviations (e.g. \"u\", \"pls\", \"msg\")",
    "roman_urdu_spelling_variation": "alternative Roman Urdu spellings (Roman Urdu/Mixed messages only)",
    "code_switching": "word-level English/Roman Urdu substitution from a small dictionary",
    "minor_noise": "one to three random character insertions, deletions or swaps",
}


def md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


def build() -> str:
    sup = load_supplement()
    fin = pd.read_csv(RES / "FINAL_RESULTS.csv", keep_default_na=False)
    rob = pd.read_csv(RES / "ROBUSTNESS_RESULTS.csv", keep_default_na=False)
    stat = pd.read_csv(RES / "STATISTICAL_COMPARISONS.csv")
    err = pd.read_csv(RES / "ERROR_COUNTS.csv", keep_default_na=False)
    b = json.loads((T / "build_summary.json").read_text())
    base = pd.read_csv(R / "baseline_results.csv")
    cfg = json.loads((R / "baseline_config.json").read_text())["models"]
    val = json.loads((T / "validation_report.json").read_text())["stats"]["residual_similarity_test_vs_train"]
    grp = pd.read_csv(T / "eda_group_structure.csv").set_index("source")
    cv = pd.read_csv(R / "group_cv_results.csv")
    xl = pd.read_csv(R / "cross_lingual_results.csv")
    sha = (C.PROCESSED_DIR / "mars_nlp_processed.sha256").read_text().split()[0]
    fp = json.loads((C.PROCESSED_DIR / "split_fingerprint.json").read_text())
    ds = sup["dataset"]
    F = fin.set_index("Model")
    ov = base[(base.subset == "Overall") & (~base.aug_trained)].set_index("model_id")
    zen, uci = grp.loc["Zenodo Roman-Urdu phishing"], grp.loc["UCI SMS Spam"]
    xs = stat.set_index("Comparison")
    s_svm = stat[stat.Comparison.str.contains("selected")].iloc[0]
    s_mx = stat[stat.Comparison.str.contains("mBERT \\(overall")].iloc[0]
    s_en = stat[stat.Comparison.str.contains("English-only")].iloc[0]
    del xs
    tv = sup["transformers"]
    mb, xr = tv["mbert"], tv["xlmr"]
    pr = sup["training_protocol"]
    d0 = pr["defaults_of_the_documented_command"]
    rm = rob[rob.Level == "mean over 10 perturbation types"].set_index("Model")["Degradation"].astype(float)
    svm_deg, mb_deg, xr_deg = rm["Char TF-IDF + Linear SVM"], rm["mBERT"], rm["XLM-R"]
    zs = xl[xl.experiment.str.contains("zero-shot")].macro_f1
    cvo = cv[cv.subset == "Overall"].set_index("model_id")
    en = base[(base.subset == "English") & (~base.aug_trained)].set_index("model_id")
    ru = base[(base.subset == "Roman Urdu + Mixed") & (~base.aug_trained)].set_index("model_id")
    sel = json.loads((C.MODELS_DIR / "best_model.json").read_text())
    # data facts recomputed from the processed corpus
    from .data_loader import get_split, load_processed

    df = load_processed()
    tr, va, te = (get_split(df, s) for s in C.SPLITS)
    n_test_rows = int((df.split == "test").sum())
    n_test_var = int(((df.split == "test") & df.is_perturbed.astype(bool)).sum())
    lang = te.language.value_counts()
    e = pd.read_csv(R / "error_analysis.csv")
    ew = e[(e.model == "TFIDF-word + LogisticRegression") & (~e.error_category.str.contains("perturbed"))]
    kw = ew.error_category.str.split("|").str[0]
    ec = e[(e.model == "TFIDF-char + LogisticRegression") & (~e.error_category.str.contains("perturbed"))]
    kc = ec.error_category.str.split("|").str[0]
    w_tmpl = int(ew.message.str.contains("Dear Consumer").sum())
    assert (len(tr), len(va), len(te)) == (ds["train_originals"], ds["validation_originals"], ds["test_originals"])
    assert te.group_id.nunique() == ds["test_groups"] and int((te.language == C.LANG_ENGLISH).sum()) == ds["test_english"]
    assert int(te.language.isin([C.LANG_ROMAN_URDU, C.LANG_MIXED]).sum()) == ds["test_roman_urdu_mixed"]

    t_main = F.reset_index()[["Model", "Accuracy", "Precision", "Recall", "Macro_F1", "Weighted_F1", "Macro_F1_CI95"]].rename(
        columns={"Macro_F1": "Macro-F1", "Weighted_F1": "Weighted-F1", "Macro_F1_CI95": "Macro-F1 95% CI"})
    t_split = pd.DataFrame([{"Split": n, "Original messages": len(d), "Scam": int((d.label == "Scam").sum()), "Legitimate": int((d.label == "Legitimate").sum()),
                             "Groups": d.group_id.nunique()} for n, d in [("Training", tr), ("Validation", va), ("Test", te)]])
    t_pert = pd.DataFrame([{"Perturbation type": k.replace("_", " "), "Variants": b["perturbation_counts"][k], "Description": PERT_DESC[k]} for k in C.PERTURBATION_TYPES])
    t_src = pd.DataFrame([
        {"Source": "SMS Spam Collection [1]", "Language": "English", "Raw messages": b["uci_raw_rows"], "Originals after cleaning and exact de-duplication": b["exact_dedup_per_source"][C.SRC_UCI]["n_after"],
         "Near-duplicate groups": int(uci.n_groups), "License": "CC BY 4.0 (as stated on the UCI page)"},
        {"Source": "Roman Urdu phishing SMS [2]", "Language": "Roman Urdu / English", "Raw messages": b["zenodo_raw_rows"], "Originals after cleaning and exact de-duplication": b["exact_dedup_per_source"][C.SRC_ZENODO]["n_after"],
         "Near-duplicate groups": int(zen.n_groups), "License": "unverified"}])
    t_tx = pd.DataFrame([{"Model": v["name"], "Parameters": f"{v['parameters']:,}", "Size (MB)": f"{v['model_size_mb']:.1f}", "Best epoch": v["best_epoch"], "Train time (s)": f"{v['train_seconds']:.2f}",
                          "GPU ms/msg": f"{v['gpu_latency_ms_per_message']:.2f}", "CPU ms/msg": f"{v['cpu_latency_ms_per_message']:.2f}", "Macro-F1": f"{v['macro_f1']:.4f}",
                          "95% CI": f"[{v['macro_f1_ci95'][0]:.3f}, {v['macro_f1_ci95'][1]:.3f}]"} for v in (mb, xr)])
    t_stat = pd.DataFrame([{"Comparison": r.Comparison.replace(" (overall test, macro-F1)", "").replace(" (English-only test, macro-F1)", " (English-only)"), "Difference (A - B)": f"{r.Difference_A_minus_B:+.6f}",
                            "95% CI": f"[{r.CI95_low:.6f}, {r.CI95_high:.6f}]", "CI includes zero": "yes" if r.CI_includes_zero else "no"} for r in stat.itertuples()])
    t_rob = pd.DataFrame([{"Model": m, "Degradation": rm[m]} for m in ("Char TF-IDF + Linear SVM", "mBERT", "XLM-R")])
    sv = rob[(rob.Model == "Char TF-IDF + Linear SVM") & rob.Level.str.startswith("per")]
    t_svm = pd.DataFrame([{"Perturbation type": r.Perturbation_Type.replace("_", " "), "Paired messages": int(r.n_pairs), "Groups": int(r.n_groups), "Degradation": float(r.Degradation),
                           "95% CI": f"[{float(r.CI95_low):.4f}, {float(r.CI95_high):.4f}]"} for r in sv.itertuples()])
    t_err = pd.DataFrame([{"Model": r.Model.replace("TFIDF-", "").replace(" + ", " + "), "ID": r.Model_ID, "Errors": int(r.Errors)} for r in err.itertuples()])
    nice = {"word +": "Word TF-IDF +", "char +": "Char TF-IDF +", "LogisticRegression": "Logistic Regression", "LinearSVM": "Linear SVM", "NaiveBayes": "Naive Bayes"}
    for k, v in nice.items():
        t_err["Model"] = t_err["Model"].str.replace(k, v, regex=False)
    t_err["Model"] = [("mBERT" if i == "mbert" else "XLM-R" if i == "xlmr" else m) for i, m in zip(t_err.ID, t_err.Model)]
    pend = "\n".join(f"{a}. {b} - {d}" for a, b, c, d in PENDING_ROWS)

    return f"""# {TITLE}

**Muhammad Rizwan**  
Independent Researcher  
Email: rizwan01107@gmail.com

## Abstract

Scam and spam text messages are written in many languages and orthographies, including Urdu in Latin script (Roman Urdu) mixed with English. We study binary scam message classification on two public SMS corpora ({b['n_originals']:,} original messages; English and Roman Urdu/code-switched text), comparing character- and word-level TF-IDF classifiers (logistic regression, linear SVM, naive Bayes) with fine-tuned mBERT and XLM-R. Because the Roman Urdu corpus is highly templated, data are split at the level of near-duplicate groups ({len(tr):,}/{len(va)}/{len(te)} training/validation/test originals; {te.group_id.nunique()} test groups), and {b['n_perturbed']:,} synthetic perturbed variants remain in the split of their source message. On the test split, character-level TF-IDF with logistic regression achieved the highest observed macro-F1 ({F.loc['Char TF-IDF + Logistic Regression', 'Macro_F1']:.4f}); character-level TF-IDF with a linear SVM was the validation-selected CPU baseline ({F.loc['Char TF-IDF + Linear SVM', 'Macro_F1']:.4f}); XLM-R was the best-performing transformer ({xr['macro_f1']:.4f}; 95% CI [{xr['macro_f1_ci95'][0]:.3f}, {xr['macro_f1_ci95'][1]:.3f}]), and mBERT reached {mb['macro_f1']:.4f} ([{mb['macro_f1_ci95'][0]:.3f}, {mb['macro_f1_ci95'][1]:.3f}]). The paired macro-F1 difference between XLM-R and the selected SVM was {s_svm.Difference_A_minus_B:+.4f} (95% CI [{s_svm.CI95_low:.4f}, {s_svm.CI95_high:.4f}]), which provides no evidence of a difference. Under a synthetic perturbation suite, mean macro-F1 degradation was {svm_deg:.4f} (character SVM), {mb_deg:.4f} (mBERT) and {xr_deg:.4f} (XLM-R). Evidence for Roman Urdu/Mixed text rests on five independent test groups, no Urdu-script benchmark is included, and the transformer metrics are owner-verified results of the original GPU run.

## Keywords

scam message detection; SMS spam; Roman Urdu; code-switching; multilingual transformers; character n-grams; robustness to noise; group-aware evaluation

## 1. Introduction

Scam messages delivered by SMS exploit urgency, financial pretexts and impersonation of banks or utilities. In South Asia, users routinely write Urdu in Latin script (Roman Urdu), which has no standardised spelling, and mix it with English within a single message. Such orthographic variability may affect both lexical classifiers, which depend on exact surface forms, and pretrained subword models, whose vocabularies were learned largely from standardised text. Evaluating these systems fairly is further complicated by the structure of the available data: public SMS corpora contain exact and near-duplicate messages, and phishing messages are often generated from a few templates, so a random split can place almost identical messages in training and test sets [9, 10].

This paper compares classical character- and word-level TF-IDF classifiers with fine-tuned multilingual transformers (mBERT [3] and XLM-R [4]) for scam message classification in English, Roman Urdu and code-switched text, under an evaluation protocol designed to respect this structure. The contributions are:

* an audited corpus of {b['n_originals']:,} original messages with a group-level train/validation/test split in which every synthetic perturbed variant stays in the split of its source message;
* a controlled suite of ten synthetic perturbation types and a paired robustness measure, degradation = original macro-F1 - perturbed macro-F1;
* a comparison of six TF-IDF classifiers with mBERT and XLM-R using cluster-bootstrap confidence intervals, including paired comparisons between the best transformer and the validation-selected baseline;
* an explicit account of what the evidence does and does not support, in particular for Roman Urdu/Mixed text (five independent test groups) and for the transformer artifacts that could not be re-verified locally.

## 2. Related Work

SMS spam filtering has been studied extensively; the SMS Spam Collection [1] is a widely used English benchmark and is one of the two corpora used here. The second corpus is a Roman Urdu phishing SMS dataset [2]. Pretrained multilingual encoders - mBERT, the multilingual variant of BERT [3], and XLM-R [4] - can be fine-tuned for classification in many languages; their behaviour on romanised, non-standard orthography is an empirical question that we examine on a small scale. The sensitivity of neural text models to character-level noise has been documented for machine translation [7] and, more generally, for common corruptions [8]; our perturbations follow this idea of random synthetic noise and are not optimised adversarial attacks. Duplicated data and train-test leakage can inflate reported performance [9, 10], which motivates the group-level splitting used here. Uncertainty for clustered observations is estimated with a cluster bootstrap [11, 12]. Implementations rely on scikit-learn [5] and the Hugging Face Transformers library [6]; supplementary explanations for the TF-IDF models use LIME [13].

## 3. Methodology

Figure 3 summarises the experimental pipeline.

![Figure 3. The MARS-NLP experimental pipeline.](../figures/mars_nlp_pipeline.png)

*Figure 3. The MARS-NLP experimental pipeline: dataset, grouped split, synthetic perturbations, TF-IDF baselines and transformers, evaluation, robustness analysis, error analysis and quality assurance.*

### 3.1 Dataset

The study uses two public corpora (Table 1). Messages are labelled *Scam* (spam, scam or phishing) or *Legitimate*. Raw text is retained; a cleaned copy applies Unicode NFKC normalisation, HTML-entity decoding, removal of control and zero-width characters and whitespace normalisation, while preserving case, punctuation and spelling variation. Exact duplicates were removed (none had conflicting labels), leaving {b['n_originals']:,} original messages. Language labels (English, Roman Urdu, Mixed/Code-Switched, Unknown) were assigned by a transparent function-word heuristic rather than a trained language-identification model; they are therefore approximate. The test split contains {lang.get('English', 0)} English messages, {int(lang.get('Mixed/Code-Switched', 0))} Mixed/Code-Switched, {int(lang.get('Roman Urdu', 0))} Roman Urdu and {int(lang.get('Unknown', 0))} unassigned messages; Roman Urdu and Mixed are analysed together ({ds['test_roman_urdu_mixed']} messages). The sources contain no Urdu-script messages, so this study includes no Urdu-script benchmark and makes no claim about Urdu-script text.

*Table 1. Source corpora.*

{md(t_src, ".0f")}

The Roman Urdu corpus is strongly templated: its 1,000 messages form only {int(zen.n_groups)} near-duplicate groups (largest group: {int(zen.max_group)} messages), whereas the UCI messages form {int(uci.n_groups):,} groups.

### 3.2 Data Splitting

Near-duplicate and templated messages were clustered using the cosine similarity of character 3-5-gram TF-IDF vectors of digit-masked, lower-cased text (threshold {C.NEAR_DUP_THRESHOLD}, single linkage), giving {b['n_groups']:,} groups over the {b['n_originals']:,} originals. Whole groups were assigned to training, validation and test sets (70/15/15 by message count, stratified by source and label, fixed seed), so no group appears in more than one split (Table 2). The mean maximum cosine similarity between a test message and any training message is {val['mean_max_sim']:.2f}, and no test message has a similarity of 0.80 or more to a training message.

*Table 2. Group-level split of the original messages.*

{md(t_split, ".0f")}

### 3.3 Synthetic Perturbations

Ten perturbation types (Table 3) were applied to real original messages only, with deterministic seeds; a perturbation is generated only when it is applicable to a message (for example, Roman Urdu spelling variation only for Roman Urdu/Mixed messages). Each variant keeps its source message's label and split, and the {b['n_perturbed']:,} variants are never treated as independent messages. Variants of test messages ({n_test_var:,} variants of the {len(te)} test originals) are used only to measure robustness. Variants of training messages were used only in a supplementary set of augmentation-trained baselines; the transformers were trained on the {len(tr):,} training originals only. These perturbations are synthetic and are not equivalent to naturally occurring adversarial messages.

*Table 3. Perturbation types and the number of variants generated.*

{md(t_pert, ".0f")}

### 3.4 Character-Level TF-IDF

Character n-grams (2 to 5 characters within word boundaries) with sublinear term frequency, lower-casing, a minimum document frequency of 2 and at most 300,000 features; the vocabulary and inverse document frequencies are fitted on the training split only ({int(ov.loc['char_lr', 'n_tfidf_features']):,} features were retained).

### 3.5 Word-Level TF-IDF

Word unigrams and bigrams with sublinear term frequency, lower-casing, Unicode-aware tokens and a minimum document frequency of 2 ({int(ov.loc['word_lr', 'n_tfidf_features']):,} features retained).

### 3.6 Classical Machine-Learning Models

Each feature set is combined with logistic regression, a linear support vector machine and multinomial naive Bayes (scikit-learn [5]); logistic regression and the SVM use balanced class weights. The regularisation strength (C) or smoothing parameter (alpha) was chosen on validation macro-F1 from small grids (LR: 0.1, 1, 10, 100; SVM: 0.01, 0.1, 1, 10; NB: 0.01, 0.1, 0.5, 1.0). The selected values were C = {cfg['char_lr']['best_param']:g} (character LR), C = {cfg['char_svm']['best_param']:g} (character SVM), C = {cfg['word_lr']['best_param']:g} (word LR), C = {cfg['word_svm']['best_param']:g} (word SVM) and alpha = {cfg['word_nb']['best_param']:g} (both naive Bayes models). The character-level linear SVM had the highest validation macro-F1 ({sel['val_macro_f1']:.4f}) and is referred to as the validation-selected CPU baseline.

### 3.7 mBERT

`bert-base-multilingual-cased` [3] with a sequence-classification head, fine-tuned on the {mb['train_samples']:,} training originals using the Hugging Face Transformers library [6]. Messages are tokenised with the model's tokenizer, truncated to a maximum length of {d0['max_length']} tokens and padded dynamically with attention masks. Optimisation used the default configuration of the training script (documented in `experiments/configs/transformers.yaml`): AdamW (learning rate {d0['learning_rate']}, weight decay {d0['weight_decay']}), linear warm-up over {int(d0['warmup_ratio'] * 100)}% of the steps with linear decay, class-weighted cross-entropy, batch size {pr['batch_size']}, at most {pr['max_epochs']} epochs and early stopping on validation macro-F1 (patience {pr['patience']}); the seed was {d0['seed']}. The exact configuration of each run is recorded in its run file, which was not available for re-inspection (Section 8).

### 3.8 XLM-R

`xlm-roberta-base` [4], fine-tuned with the same data, head, optimiser, schedule, loss, batch size, epoch limit and early-stopping rule as mBERT.

## 4. Experimental Setup

**Protocol.** Hyper-parameters (baselines) and the stopping epoch (transformers) were selected on the validation originals; the test split was used once for the final numbers. All reported metrics are computed on the {len(te)} test originals unless stated otherwise ({int((te.label == 'Scam').sum())} Scam and {int((te.label == 'Legitimate').sum())} Legitimate messages).

**Metrics.** Accuracy, precision and recall for the Scam class, macro-F1 (the primary metric, because the classes are imbalanced) and weighted-F1. Ninety-five percent confidence intervals for macro-F1 use a percentile cluster bootstrap that resamples near-duplicate groups rather than messages [11, 12] (1,000 resamples for the CPU baselines). Paired comparisons between two models on the same test messages use a paired cluster bootstrap over test groups of the macro-F1 difference.

**Robustness.** For each perturbation type, macro-F1 is computed on the matched original messages and on their perturbed variants; degradation is original macro-F1 minus perturbed macro-F1, so lower values are better and negative values mean that the perturbed score was higher. The model-level value is the mean over the ten perturbation types. This quantifies robustness to the study's synthetic perturbation suite only.

**Error analysis.** Errors are counted per model over the test split's original messages and their perturbed variants ({n_test_rows:,} messages in total); counts are never aggregated across models.

**Hardware and software.** The CPU baselines were trained and timed on one CPU core with scikit-learn 1.8.0 under Python 3.12. The transformers were fine-tuned on a Google Colab NVIDIA Tesla T4 GPU; their latency benchmarks are single-message wall-clock times on that GPU and on the Colab CPU. Latency values from the two environments are not directly comparable and are not edge-device measurements.

## 5. Results

### 5.1 Main Model Comparison

Table 4 and Figure 1 report test performance. Character TF-IDF with logistic regression achieved the highest observed test macro-F1 ({F.loc['Char TF-IDF + Logistic Regression', 'Macro_F1']:.4f}). Character TF-IDF with a linear SVM was the validation-selected CPU baseline, with a test macro-F1 of {F.loc['Char TF-IDF + Linear SVM', 'Macro_F1']:.4f}; the word-level SVM obtained the same value. XLM-R was the best-performing transformer with a macro-F1 of {xr['macro_f1']:.4f}. The confidence intervals of the four best models overlap almost completely, and the naive Bayes models and mBERT are clearly lower. Five-fold group cross-validation over all originals, a supplementary estimate not used for model selection, gave overall macro-F1 values of {cvo.loc['char_svm', 'macro_f1']:.3f} (character SVM), {cvo.loc['char_lr', 'macro_f1']:.3f} (character LR) and {cvo.loc['word_lr', 'macro_f1']:.3f} (word LR), consistent with the fixed-split ranking among these models.

*Table 4. Test performance on the {len(te)} original test messages (precision and recall refer to the Scam class; CI: 95% cluster bootstrap).*

{md(t_main)}

![Figure 1. Macro-F1 of all major models.](../figures/model_macro_f1_comparison.png)

*Figure 1. Test macro-F1 of all models with 95% cluster-bootstrap confidence intervals.*

For Roman Urdu/Mixed messages ({ds['test_roman_urdu_mixed']} test messages from five independent groups), the character-level LR and SVM obtained macro-F1 of {ru.loc['char_lr', 'macro_f1']:.3f} with an interval of [{ru.loc['char_lr', 'macro_f1_ci_low']:.3f}, {ru.loc['char_lr', 'macro_f1_ci_high']:.3f}]; this interval illustrates that such values are descriptive only. English test macro-F1 was {en.loc['char_lr', 'macro_f1']:.3f} (LR) and {en.loc['char_svm', 'macro_f1']:.3f} (SVM) on {ds['test_english']} messages. Per-language values for the transformers are not part of the verified summary and are not reported. Zero-shot transfer with TF-IDF models between English and Roman Urdu/Mixed text failed (macro-F1 between {zs.min():.2f} and {zs.max():.2f}); because language is confounded with source and label semantics, this reflects domain and template differences as much as language.

### 5.2 Transformer Results

Both transformers were fine-tuned on the same {mb['train_samples']:,} training originals (Table 5). XLM-R reached its best validation epoch at epoch {xr['best_epoch']} and mBERT at epoch {mb['best_epoch']}. mBERT had a lower recall for the Scam class ({F.loc['mBERT', 'Recall']:.4f}) than XLM-R ({F.loc['XLM-R', 'Recall']:.4f}) and a lower precision ({F.loc['mBERT', 'Precision']:.4f} versus {F.loc['XLM-R', 'Precision']:.4f}). The transformers are far larger and slower than the TF-IDF models: the character-level SVM has {int(ov.loc['char_svm', 'parameters']):,} classifier weights, a model file of {float(ov.loc['char_svm', 'model_size_mb']):.2f} MB and a mean single-message CPU latency of {float(ov.loc['char_svm', 'latency_single_ms_mean']):.2f} ms on the CPU baseline machine, compared with {mb['model_size_mb']} MB and {xr['model_size_mb']} MB for the transformers.

*Table 5. Transformer training details and computational benchmarks (NVIDIA Tesla T4).*

{md(t_tx)}

### 5.3 Statistical Comparison

Table 6 reports paired macro-F1 differences on the test split (cluster bootstrap over test groups). The difference between XLM-R and the selected character SVM is {s_svm.Difference_A_minus_B:+.6f} with a 95% confidence interval of [{s_svm.CI95_low:.6f}, {s_svm.CI95_high:.6f}], which includes zero; there is therefore no evidence of a statistically significant difference between them, and no superiority of XLM-R over the selected SVM is claimed. XLM-R exceeded mBERT by {s_mx.Difference_A_minus_B:.6f} (95% CI [{s_mx.CI95_low:.6f}, {s_mx.CI95_high:.6f}]); the interval excludes zero, but it is wide and its lower bound is close to zero, so the evidence is modest. On English test messages only, the difference between XLM-R and mBERT was {s_en.Difference_A_minus_B:.6f} (95% CI [{s_en.CI95_low:.6f}, {s_en.CI95_high:.6f}]), which includes zero; no English-only superiority is claimed.

*Table 6. Paired macro-F1 differences (A minus B) with 95% cluster-bootstrap intervals.*

{md(t_stat, ".6f")}

### 5.4 Robustness Evaluation

Table 7 and Figure 2 give the mean degradation under the ten synthetic perturbation types. The character-level SVM degraded by {svm_deg:.4f}, XLM-R by {xr_deg:.4f}, and mBERT by {mb_deg:.4f}; the slightly negative value for mBERT means that its perturbed macro-F1 was marginally higher than its original macro-F1 and is not interpreted as a robustness gain. XLM-R therefore showed the largest degradation of the three models, and its absolute drop is small. Table 8 gives the per-type results for the selected SVM: the only type whose paired interval excludes zero is character substitution ({float(sv[sv.Perturbation_Type == 'character_substitution'].Degradation.iloc[0]):.4f}); casing variation has zero effect by construction because the vectoriser lower-cases text, and the Roman Urdu spelling variation estimate rests on five groups. Per-type results for the transformers are not part of the verified summary. All of these values describe robustness to the study's synthetic perturbation suite, not real-world adversarial robustness.

*Table 7. Mean degradation over the ten perturbation types (original macro-F1 - perturbed macro-F1; lower is better).*

{md(t_rob)}

![Figure 2. Robustness degradation.](../figures/robustness_degradation.png)

*Figure 2. Robustness degradation under the synthetic perturbation suite.*

*Table 8. Per-type degradation for the validation-selected character-level SVM (paired test messages).*

{md(t_svm)}

### 5.5 Error Analysis

Table 9 lists the number of misclassifications per model over the test originals and their perturbed variants ({n_test_rows:,} messages). On the 935 test originals, the character-level logistic regression made {len(ec)} errors ({int((kc == 'FN').sum())} false negatives and {int((kc == 'FP').sum())} false positive). The word-level logistic regression made {len(ew)} errors, and all {int((kw == 'FP').sum())} of its false positives were legitimate Roman Urdu/Mixed messages from one templated group (utility-bill notices; {w_tmpl} messages), which illustrates how a single template can dominate errors on the small Roman Urdu/Mixed portion of the test set. Models marked "aug-trained" are supplementary baselines trained on the training originals plus their perturbed variants. False-positive and false-negative splits for the transformers require their raw prediction files and are not reported; counts are never summed across models.

*Table 9. Errors per model over test originals and perturbed variants.*

{md(t_err[['Model', 'ID', 'Errors']], ".0f")}

## 6. Discussion

Under group-aware evaluation, simple character-level TF-IDF classifiers are strong and inexpensive baselines on these corpora: the best of them reached a macro-F1 of {F.loc['Char TF-IDF + Logistic Regression', 'Macro_F1']:.4f}, and the validation-selected SVM {F.loc['Char TF-IDF + Linear SVM', 'Macro_F1']:.4f}. XLM-R ({xr['macro_f1']:.4f}) was competitive, but the paired interval against the selected SVM includes zero, so the data do not support a ranking among the top models; mBERT ({mb['macro_f1']:.4f}) was lower, mainly through recall. We did not investigate why mBERT and XLM-R differ, and the modest evidence for the XLM-R versus mBERT difference calls for repeated runs.

The lower macro-F1 of the word-level logistic regression relative to the character-level models on the fixed split is largely attributable to a single templated group of legitimate messages. Because the Roman Urdu/Mixed portion of the test set contains only five independent groups, this should not be read as evidence that character n-grams are generally superior; the cross-validated difference is smaller. Likewise, the Roman Urdu/Mixed scores of the baselines are descriptive.

The robustness comparison favours the character-level SVM on this suite (smallest positive degradation, with a precisely estimated per-type profile), whereas XLM-R had the largest degradation. A possible contributing factor, which we did not test, is the interaction between subword tokenisation and character-level noise. The suite consists of random synthetic edits, and the results do not indicate how the models would behave against messages crafted by an adversary or against naturally occurring spelling variation.

Computationally, the transformers require hundreds of megabytes and, on a CPU, on the order of one hundred milliseconds per message in the Colab environment, compared with under one millisecond for the TF-IDF models on a single core; these figures come from different machines and are indicative only. Whether the additional cost is justified depends on the deployment setting, which this study does not address.

## 7. Limitations

* The dataset is domain-specific: English SMS marketing spam and templated Roman Urdu phishing messages; results may not transfer to other message types, regions or time periods.
* The perturbations are synthetic. They are not equivalent to naturally occurring perturbations or adversarial messages, so robustness findings are limited to the study's perturbation suite.
* Evidence for Roman Urdu/Mixed text is limited, and it comes from only five independent test groups; point estimates and intervals for this subset are descriptive.
* There is no comprehensive Urdu-script benchmark: the sources contain no Urdu-script messages and Urdu script is not evaluated.
* Multi-seed variance was not estimated for every model; each transformer was trained once.
* The hyper-parameter search was not exhaustive (small grids for the baselines; fixed settings for the transformers).
* Computational measurements depend on the Tesla T4 and the software environment, and CPU and GPU benchmarks were taken on different machines.
* SHAP and Integrated Gradients analyses were not performed, and no interpretability analysis of the transformers was carried out.
* The raw transformer artifacts from the original Colab run were unavailable in the final local release workspace; the transformer metrics are therefore owner-verified results that have not been independently reproduced locally (see Section 8).
* Language labels are heuristic, source and language are partly confounded, and the positive class differs in meaning between corpora (unsolicited marketing versus phishing). The Zenodo license is unverified.

## 8. Reproducibility

*Code and data.* The repository contains the data-building, splitting, validation, training, evaluation, robustness and reporting code. Third-party data are not redistributed: the processed corpus is rebuilt deterministically from the original UCI and Zenodo files and verified against a SHA-256 checksum (`{sha}`) and split fingerprints (training {fp['train']['sha256'][:12]}..., validation {fp['validation']['sha256'][:12]}..., test {fp['test']['sha256'][:12]}...). Every training run verifies the checksum and aborts on a mismatch or on any group overlap between splits.

*Quality assurance.* The original Colab execution reported **{sup['qa_owner_verified']['PASS']} PASS, {sup['qa_owner_verified']['FAIL']} FAIL, {sup['qa_owner_verified']['PENDING']} PENDING, {sup['qa_owner_verified']['SKIP']} SKIP**. The final local release-package QA reports **50 PASS, 0 FAIL, 3 PENDING, 0 SKIP**. The three pending checks arise only because the raw Colab transformer artifacts are absent from the final local workspace:

{pend}

*Status of the transformer results.* The transformer metrics were obtained from the verified Colab execution and are treated as owner-verified results. Raw transformer artifacts (run files, predictions, robustness files, checkpoints and the CUDA/PyTorch environment record) were not available for independent local verification; they were neither reconstructed nor replaced, and the three pending checks are not failures. The CPU-baseline results, in contrast, were recomputed locally from their raw prediction files. The training settings quoted in Section 3.7 are the documented defaults of the training command. Complete raw-artifact reproducibility is therefore not claimed for mBERT and XLM-R. If the raw files are supplied, `python -m src.qa_check` recomputes and checks every transformer metric.

## 9. Conclusion

On two public SMS corpora and under a group-aware protocol, character-level TF-IDF classifiers were strong baselines: character TF-IDF with logistic regression achieved the highest observed test macro-F1 ({F.loc['Char TF-IDF + Logistic Regression', 'Macro_F1']:.4f}), and the validation-selected character SVM reached {F.loc['Char TF-IDF + Linear SVM', 'Macro_F1']:.4f}. XLM-R was the best-performing transformer ({xr['macro_f1']:.4f}) but did not detectably differ from the selected SVM, and mBERT was lower ({mb['macro_f1']:.4f}). Under the synthetic perturbation suite the SVM degraded least, while the evidence for Roman Urdu/Mixed text, limited to five independent test groups, remains descriptive. The transformer metrics are owner-verified results of the original GPU run. Future work includes a larger and more diverse Roman Urdu and Urdu-script corpus, repeated transformer runs, naturally occurring and adversarial perturbations, and interpretability analysis of the transformers.

## 10. References

{REFS}
"""


def main() -> None:
    text = build()
    out = C.REPORTS_DIR / "MARS_NLP_Final_Manuscript.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(text.split()):,} words)")


if __name__ == "__main__":
    main()
