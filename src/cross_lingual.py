"""Cross-lingual experiments (classical baselines) + group-aware cross-validation.

Experiment A  combined training -> language-specific evaluation (main split AND pooled out-of-fold predictions)
Experiment B  zero-shot transfer: train on one language, evaluate on the other
Experiment C  multilingual (combined) vs language-specific training, paired cluster-bootstrap difference

IMPORTANT CAVEATS (also stated in the report): language is confounded with source and with label semantics
(UCI "spam" = unsolicited marketing, Zenodo "phishing" = fraud), and the Zenodo data has only ~19 near-duplicate
template groups. Results therefore measure template/domain transfer at least as much as language transfer.

Usage: python -m src.cross_lingual
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from . import config as C
from .data_loader import get_split, load_processed
from .evaluation import (cluster_bootstrap_ci, compute_metrics, mcnemar_exact, paired_cluster_bootstrap_diff,
                         reliability_note, subset_status)
from .features import make_vectorizer
from .prediction import pipeline_scores
from .robustness import robustness_rows
from .train_baselines import MODEL_SPECS
from .utils import get_logger, load_json, save_json, set_seed

log = get_logger()
RU_FAMILY = [C.LANG_ROMAN_URDU, C.LANG_MIXED]
CONFIGS = [("char", "lr"), ("word", "lr"), ("char", "svm")]   # compact, informative subset


def _best_param(feat: str, key: str):
    cfg = load_json(C.RESULTS_DIR / "baseline_config.json")["models"][f"{feat}_{key}"]
    return cfg["best_param"]


def _fit(feat: str, key: str, train: pd.DataFrame) -> Pipeline:
    vec = make_vectorizer(feat)
    clf = MODEL_SPECS[key][1](_best_param(feat, key))
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    pipe.fit(train["cleaned_message"].tolist(), train["label_id"].values)
    return pipe


def _eval(pipe: Pipeline, test: pd.DataFrame, n_boot: int = 1000) -> tuple[dict, np.ndarray]:
    pred, _, _ = pipeline_scores(pipe, test["cleaned_message"].tolist())
    ok, note = subset_status(test["label_id"].values)
    m = compute_metrics(test["label_id"].values, pred)
    ci = cluster_bootstrap_ci(test["label_id"].values, pred, test["group_id"].values, "macro_f1", n_boot)
    if ok:
        note = reliability_note(ci["n_groups"])
    m.update({"macro_f1_ci_low": ci["ci_low"], "macro_f1_ci_high": ci["ci_high"], "n_groups": ci["n_groups"],
              "n_scam": int((test["label_id"] == 1).sum()), "status": note})
    return m, pred


def experiments_abc(n_boot: int = 1000) -> pd.DataFrame:
    df = load_processed()
    tr, te = get_split(df, "train"), get_split(df, "test")
    tr_en, tr_ru = tr[tr.language == C.LANG_ENGLISH], tr[tr.language.isin(RU_FAMILY)]
    te_en, te_ru = te[te.language == C.LANG_ENGLISH], te[te.language.isin(RU_FAMILY)]
    rows = []
    for feat, key in CONFIGS:
        name = f"{feat}_{key}"
        log.info("Cross-lingual (main split): %s", name)
        combined, en_only, ru_only = _fit(feat, key, tr), _fit(feat, key, tr_en), _fit(feat, key, tr_ru)
        plan = [
            ("A: combined -> English", "combined", combined, te_en),
            ("A: combined -> Roman Urdu+Mixed", "combined", combined, te_ru),
            ("B: English-only -> Roman Urdu+Mixed (zero-shot)", "English-only", en_only, te_ru),
            ("B: Roman Urdu+Mixed-only -> English (zero-shot)", "RomanUrdu+Mixed-only", ru_only, te_en),
            ("C: English-only -> English", "English-only", en_only, te_en),
            ("C: Roman Urdu+Mixed-only -> Roman Urdu+Mixed", "RomanUrdu+Mixed-only", ru_only, te_ru),
        ]
        preds = {}
        for exp, train_name, pipe, test in plan:
            m, pred = _eval(pipe, test, n_boot)
            preds[exp] = pred
            rows.append({"model_id": name, "experiment": exp, "train_set": train_name,
                         "n_train": {"combined": len(tr), "English-only": len(tr_en), "RomanUrdu+Mixed-only": len(tr_ru)}[train_name],
                         "n_test": m["n"], **{k: m[k] for k in ["n_scam", "n_groups", "accuracy", "precision_scam", "recall_scam",
                                                               "f1_scam", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "status"]}})
        for lang_name, te_sub, a_key, b_key in [("English", te_en, "A: combined -> English", "C: English-only -> English"),
                                                 ("Roman Urdu+Mixed", te_ru, "A: combined -> Roman Urdu+Mixed",
                                                  "C: Roman Urdu+Mixed-only -> Roman Urdu+Mixed")]:
            d = paired_cluster_bootstrap_diff(te_sub["label_id"].values, preds[a_key], preds[b_key], te_sub["group_id"].values,
                                              "macro_f1", n_boot)
            mc = mcnemar_exact(te_sub["label_id"].values, preds[a_key], preds[b_key])
            rows.append({"model_id": name, "experiment": f"C-diff: combined minus language-specific on {lang_name}",
                         "train_set": "paired", "n_test": int(len(te_sub)), "n_groups": d["n_groups"], "macro_f1": d["diff"],
                         "macro_f1_ci_low": d["ci_low"], "macro_f1_ci_high": d["ci_high"],
                         "status": f"CI excludes 0: {d['ci_excludes_zero']}; McNemar exact p={mc['p_value']:.3g} "
                                   f"(discordant {mc['a_only_correct']}/{mc['b_only_correct']}; p optimistic: clustered data)"})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "cross_lingual_results.csv", index=False)
    return out


def group_cv(n_splits: int = 5, n_boot: int = 1000) -> pd.DataFrame:
    """Stratified group k-fold over ALL originals: every near-duplicate group is held out exactly once."""
    set_seed(C.SEED)
    df = load_processed()
    orig = df[~df["is_perturbed"].astype(bool)].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    strat = (orig["source_dataset"].str[:3] + "|" + orig["label"]).values
    folds = list(sgkf.split(orig, strat, groups=orig["group_id"].values))
    var = df[df["is_perturbed"].astype(bool)].reset_index(drop=True)
    rows, fold_rows, rob_rows = [], [], []
    for feat, key in CONFIGS:
        name = f"{feat}_{key}"
        log.info("Group CV: %s", name)
        oof = np.full(len(orig), -1)
        var_pred = pd.Series(-1, index=var["message_id"].values)
        for k, (tri, tei) in enumerate(folds):
            pipe = _fit(feat, key, orig.iloc[tri])
            oof[tei], _, _ = pipeline_scores(pipe, orig.iloc[tei]["cleaned_message"].tolist())
            held = set(orig.iloc[tei]["message_id"])
            vf = var[var["original_message_id"].isin(held)]
            var_pred.loc[vf["message_id"].values] = pipeline_scores(pipe, vf["cleaned_message"].tolist())[0]
            fm = compute_metrics(orig.iloc[tei]["label_id"].values, oof[tei])
            fold_rows.append({"model_id": name, "fold": k, "n_test": fm["n"], "macro_f1": fm["macro_f1"], "accuracy": fm["accuracy"]})
        assert (oof >= 0).all() and (var_pred.values >= 0).all()
        rob_rows += robustness_rows(orig, var.assign(pred=var_pred.loc[var["message_id"]].values),
                                    pd.Series(oof, index=orig["message_id"].values), name, n_boot)
        slices = {"Overall": None, "English": [C.LANG_ENGLISH], "Roman Urdu": [C.LANG_ROMAN_URDU], "Mixed/Code-Switched": [C.LANG_MIXED],
                  "Roman Urdu + Mixed": RU_FAMILY, "source:Zenodo": "zen", "source:UCI": "uci"}
        for sname, sel in slices.items():
            if sel is None:
                mask = np.ones(len(orig), bool)
            elif sel == "zen":
                mask = (orig.source_dataset == C.SRC_ZENODO).values
            elif sel == "uci":
                mask = (orig.source_dataset == C.SRC_UCI).values
            else:
                mask = orig.language.isin(sel).values
            sub = orig[mask]
            ok, note = subset_status(sub["label_id"].values) if len(sub) else (False, "no samples")
            m = compute_metrics(sub["label_id"].values, oof[mask]) if len(sub) else {"n": 0}
            if ok:
                ci = cluster_bootstrap_ci(sub["label_id"].values, oof[mask], sub["group_id"].values, "macro_f1", n_boot)
                m.update({"macro_f1_ci_low": ci["ci_low"], "macro_f1_ci_high": ci["ci_high"], "n_groups": ci["n_groups"]})
                note = reliability_note(ci["n_groups"])
            rows.append({"model_id": name, "subset": sname, "status": note, "n_scam": int((sub["label_id"] == 1).sum()),
                         **{k: m.get(k, np.nan) for k in ["n", "n_groups", "accuracy", "precision_scam", "recall_scam", "f1_scam",
                                                         "macro_f1", "weighted_f1", "macro_f1_ci_low", "macro_f1_ci_high"]}})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "group_cv_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(C.RESULTS_DIR / "group_cv_folds.csv", index=False)
    pd.DataFrame(rob_rows).to_csv(C.RESULTS_DIR / "group_cv_robustness.csv", index=False)
    save_json({"n_splits": n_splits, "splitter": "StratifiedGroupKFold(shuffle=True)", "seed": C.SEED,
               "hyperparameters": "best_param from main-split validation selection (not re-tuned inside folds)"},
              C.RESULTS_DIR / "group_cv_config.json")
    return out


if __name__ == "__main__":
    print(experiments_abc().to_string(index=False))
    print(group_cv().to_string(index=False))
