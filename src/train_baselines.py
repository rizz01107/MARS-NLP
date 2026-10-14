"""TF-IDF baselines: Logistic Regression, Linear SVM, Multinomial Naive Bayes.

Usage:  python -m src.train_baselines [--no-aug] [--quick]
Outputs: experiments/results/baseline_results.csv, baseline_predictions.csv, baseline_config.json,
         models/baseline_<id>.joblib, models/best_model.json, reports/figures/baseline_confusion_matrices.png

Protocol: hyper-parameters are chosen on the VALIDATION originals (macro-F1); the TEST split is used once for the
final numbers. Models are fit on training originals only ("orig"), or on training originals + their perturbed
variants ("aug": noise-augmented training). Validation/test variants are never used for fitting or selection.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import config as C
from .data_loader import get_split, load_processed
from .evaluation import benchmark_latency, compute_metrics, file_size_mb, metrics_with_ci, reliability_note, subset_status
from .features import FEATURE_KINDS, make_vectorizer
from .prediction import pipeline_scores
from .utils import ensure_dir, environment_info, get_logger, save_json, set_seed, timer

log = get_logger()

MODEL_SPECS = {
    "lr": ("LogisticRegression", lambda p: LogisticRegression(C=p, max_iter=2000, solver="liblinear",
                                                              class_weight="balanced", random_state=C.SEED),
           [0.1, 1.0, 10.0, 100.0], "C"),
    "svm": ("LinearSVM", lambda p: LinearSVC(C=p, class_weight="balanced", random_state=C.SEED, max_iter=5000),
            [0.01, 0.1, 1.0, 10.0], "C"),
    "nb": ("NaiveBayes", lambda p: MultinomialNB(alpha=p), [0.01, 0.1, 0.5, 1.0], "alpha"),
}
EVAL_SUBSETS = [("Overall", None), (C.LANG_ENGLISH, [C.LANG_ENGLISH]), (C.LANG_URDU, [C.LANG_URDU]),
                (C.LANG_ROMAN_URDU, [C.LANG_ROMAN_URDU]), (C.LANG_MIXED, [C.LANG_MIXED]),
                ("Roman Urdu + Mixed", [C.LANG_ROMAN_URDU, C.LANG_MIXED])]


def model_id(feat: str, key: str, aug: bool) -> str:
    return f"{feat}_{key}" + ("_aug" if aug else "")


def display_name(feat: str, key: str, aug: bool) -> str:
    return f"TFIDF-{feat} + {MODEL_SPECS[key][0]}" + (" (aug-trained)" if aug else "")


def select_and_fit(feat: str, key: str, train: pd.DataFrame, val: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Fit vectoriser once, choose the hyper-parameter on validation macro-F1, return the final pipeline."""
    _, factory, grid, pname = MODEL_SPECS[key]
    vec = make_vectorizer(feat)
    with timer() as t_vec:
        Xtr = vec.fit_transform(train["cleaned_message"].tolist())
    Xva = vec.transform(val["cleaned_message"].tolist())
    best, rows = None, []
    with timer() as t_grid:
        for p in grid:
            clf = factory(p)
            with timer() as t_fit:
                clf.fit(Xtr, train["label_id"].values)
            f1 = compute_metrics(val["label_id"].values, clf.predict(Xva))["macro_f1"]
            rows.append({pname: p, "val_macro_f1": f1, "fit_seconds": t_fit["seconds"]})
            if best is None or f1 > best[0] + 1e-12:
                best = (f1, p, clf, t_fit["seconds"])
    f1, p, clf, fit_s = best
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    info = {"param_name": pname, "best_param": p, "val_macro_f1": f1, "grid": rows,
            "train_seconds": t_vec["seconds"] + fit_s, "grid_search_seconds_total": t_grid["seconds"] + t_vec["seconds"],
            "n_features": int(Xtr.shape[1]), "n_train": int(len(train))}
    return pipe, info


def n_parameters(pipe: Pipeline) -> int:
    """Number of learned classifier weights (+ intercept); vocabulary/idf vectors are reported separately."""
    clf = pipe.steps[-1][1]
    if hasattr(clf, "coef_"):
        return int(np.size(clf.coef_) + np.size(clf.intercept_))
    return int(np.size(clf.feature_log_prob_) + np.size(clf.class_log_prior_))


def evaluate_subsets(test: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    rows = []
    for name, langs in EVAL_SUBSETS:
        mask = np.ones(len(test), bool) if langs is None else test["language"].isin(langs).values
        sub = test[mask]
        ok, note = subset_status(sub["label_id"].values) if len(sub) else (False, "no samples")
        if name == "Overall":
            ok, note = True, "ok"
        if ok:
            m = metrics_with_ci(sub, pred[mask])
            note = reliability_note(m["n_groups"])
        else:
            m = {k: np.nan for k in ["accuracy", "precision_scam", "recall_scam", "f1_scam", "macro_precision",
                                     "macro_recall", "macro_f1", "weighted_f1", "tn", "fp", "fn", "tp",
                                     "macro_f1_ci_low", "macro_f1_ci_high"]}
            m.update({"n": int(len(sub)), "n_groups": int(sub["group_id"].nunique()) if len(sub) else 0})
        rows.append({"subset": name, "status": note, "n_scam": int((sub["label_id"] == 1).sum()), **m})
    for src, label in [("UCI", C.SRC_UCI), ("Zenodo", C.SRC_ZENODO)]:
        mask = (test["source_dataset"] == label).values
        sub = test[mask]
        if len(sub):
            ok, note = subset_status(sub["label_id"].values)
            m = metrics_with_ci(sub, pred[mask]) if ok else {"n": int(len(sub))}
            if ok:
                note = reliability_note(m["n_groups"])
            rows.append({"subset": f"source:{src}", "status": note, "n_scam": int((sub["label_id"] == 1).sum()), **m})
    return rows


def run(include_aug: bool = True, quick: bool = False) -> pd.DataFrame:
    set_seed(C.SEED)
    df = load_processed()
    train_o, val, test = (get_split(df, s) for s in C.SPLITS)
    train_aug = get_split(df, "train", include_perturbed=True)
    log.info("train originals=%d (aug=%d) | val=%d | test=%d", len(train_o), len(train_aug), len(val), len(test))
    ensure_dir(C.MODELS_DIR), ensure_dir(C.RESULTS_DIR), ensure_dir(C.FIGURES_DIR)

    result_rows, pred_rows, configs = [], [], {}
    modes = [False] + ([True] if include_aug else [])
    feats = FEATURE_KINDS[:1] if quick else FEATURE_KINDS
    for aug in modes:
        train = train_aug if aug else train_o
        for feat in feats:
            for key in MODEL_SPECS:
                mid, name = model_id(feat, key, aug), display_name(feat, key, aug)
                log.info("Training %s ...", name)
                pipe, info = select_and_fit(feat, key, train, val)
                ids, p, kind = pipeline_scores(pipe, test["cleaned_message"].tolist())
                path = C.MODELS_DIR / f"baseline_{mid}.joblib"
                meta = {"name": name, "model_id": mid, "features": feat, "model": MODEL_SPECS[key][0], "augmented_training": aug,
                        "best_param": info["best_param"], "label2id": C.LABEL2ID, "seed": C.SEED,
                        "confidence_type": kind}
                joblib.dump({"pipeline": pipe, "meta": meta}, path, compress=3)
                lat = benchmark_latency(lambda t, pp=pipe: pipeline_scores(pp, t), test["cleaned_message"].tolist())
                eff = {"parameters": n_parameters(pipe), "n_tfidf_features": info["n_features"],
                       "model_size_mb": file_size_mb(path), "train_seconds": info["train_seconds"], **lat}
                for r in evaluate_subsets(test, ids):
                    result_rows.append({"model_id": mid, "model": name, "features": feat, "classifier": MODEL_SPECS[key][0],
                                        "aug_trained": aug, "split": "test", "best_param": info["best_param"],
                                        "val_macro_f1": info["val_macro_f1"], **r, **(eff if r["subset"] == "Overall" else {})})
                pred_rows.append(pd.DataFrame({"model_id": mid, "message_id": test["message_id"], "y_true": test["label_id"],
                                               "y_pred": ids, "p_scam": p, "confidence_type": kind}))
                configs[mid] = {**meta, "grid": info["grid"], "n_train": info["n_train"]}
                log.info("  val macro-F1=%.4f  test macro-F1=%.4f", info["val_macro_f1"],
                         compute_metrics(test["label_id"].values, ids)["macro_f1"])
    res = pd.DataFrame(result_rows)
    res.to_csv(C.RESULTS_DIR / "baseline_results.csv", index=False)
    pd.concat(pred_rows).to_csv(C.RESULTS_DIR / "baseline_predictions.csv", index=False)
    save_json({"environment": environment_info(), "models": configs, "split_sizes": {
        "train_originals": len(train_o), "train_with_variants": len(train_aug), "validation": len(val), "test": len(test)}},
        C.RESULTS_DIR / "baseline_config.json")
    # best model (by validation macro-F1, originals-trained only) for the demo app
    overall = res[(res.subset == "Overall") & (~res.aug_trained)].sort_values("val_macro_f1", ascending=False)
    top = overall.iloc[0]
    save_json({"model_id": top.model_id, "name": top.model, "selected_by": "validation macro-F1 (originals-trained baselines)",
               "val_macro_f1": float(top.val_macro_f1)}, C.MODELS_DIR / "best_model.json")
    _plot_confusions(res)
    return res


def _plot_confusions(res: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ov = res[res.subset == "Overall"].reset_index(drop=True)
    n = len(ov)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, ov.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Legit", "Scam"], fontsize=10)
        ax.set_yticks([0, 1], ["Legit", "Scam"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(r.model.replace(" + ", "\n+ "), fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Baseline confusion matrices (test originals)", fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "baseline_confusion_matrices.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-aug", action="store_true", help="skip noise-augmented training variants")
    ap.add_argument("--quick", action="store_true", help="word features only (fast smoke run)")
    a = ap.parse_args()
    run(include_aug=not a.no_aug, quick=a.quick)


if __name__ == "__main__":
    main()
