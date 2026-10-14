"""Explainability for the classical baselines and a model-agnostic LIME wrapper (usable with transformers).

* Linear models (LogReg / LinearSVM / NaiveBayes): exact additive per-feature contributions,
  ``contribution_j = w_j * x_j`` (x = TF-IDF value) - the decomposition of the decision score.
* LIME (Ribeiro et al., 2016): local surrogate for any ``predict_proba`` function; the same wrapper is used for the
  fine-tuned transformers (the wrapper is model-agnostic; transformer interpretability runs were not performed in this project).

IMPORTANT: attributions describe what the MODEL used, not causal reasons why a message is a scam. TF-IDF
weights also reflect dataset artefacts (e.g. source-specific vocabulary and templates).

Usage: python -m src.explainability
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import get_split, load_processed
from .prediction import BaselinePredictor, pipeline_scores
from .utils import ensure_dir, get_logger

log = get_logger()


def _class_weights(clf, n_features: int) -> np.ndarray:
    """Per-feature weight towards the Scam class (positive = pushes towards Scam)."""
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel()
    return np.asarray(clf.feature_log_prob_[1] - clf.feature_log_prob_[0])


def linear_feature_contributions(pipe, text: str, top_k: int = 8) -> list[tuple[str, float]]:
    """Top-|k| additive contributions (feature, signed contribution towards Scam) for one message."""
    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    x = vec.transform([text])
    w = _class_weights(clf, x.shape[1])
    idx = x.indices
    contrib = w[idx] * x.data
    names = vec.get_feature_names_out()
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [(str(names[idx[i]]), float(contrib[i])) for i in order]


def global_top_features(pipe, k: int = 15) -> pd.DataFrame:
    """Globally most Scam-/Legitimate-indicating features (largest signed weights)."""
    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    w = _class_weights(clf, len(vec.vocabulary_))
    names = vec.get_feature_names_out()
    top_s = np.argsort(-w)[:k]
    top_l = np.argsort(w)[:k]
    return pd.concat([
        pd.DataFrame({"direction": "Scam", "feature": names[top_s], "weight": w[top_s]}),
        pd.DataFrame({"direction": "Legitimate", "feature": names[top_l], "weight": w[top_l]})], ignore_index=True)


def lime_explain(predict_scam_proba: Callable[[Sequence[str]], np.ndarray], text: str, num_features: int = 8,
                 num_samples: int = 500, seed: int = C.SEED) -> list[tuple[str, float]]:
    """LIME explanation for the Scam class from a function ``texts -> p(scam)`` (works for any model)."""
    from lime.lime_text import LimeTextExplainer

    def fn(texts):
        p = np.asarray(predict_scam_proba(list(texts)), dtype=float)
        return np.column_stack([1 - p, p])

    expl = LimeTextExplainer(class_names=list(C.LABELS), random_state=seed, split_expression=r"\s+", bow=True)
    e = expl.explain_instance(text, fn, num_features=num_features, num_samples=num_samples, labels=(1,))
    return [(w, float(s)) for w, s in e.as_list(label=1)]


def pick_examples(preds: pd.DataFrame, test: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """One example per outcome (TP, TN, FP, FN) - the most confident instance of each, if it exists."""
    d = test.merge(preds, on="message_id")
    d["outcome"] = np.select([(d.y_true == 1) & (d.y_pred == 1), (d.y_true == 0) & (d.y_pred == 0),
                              (d.y_true == 0) & (d.y_pred == 1), (d.y_true == 1) & (d.y_pred == 0)],
                             ["TP", "TN", "FP", "FN"], default="NA")
    d["conf"] = np.where(d.y_pred == 1, d.p_scam, 1 - d.p_scam)
    picks = [d[d.outcome == o].sort_values(["conf", "message_id"], ascending=[False, True]).head(1) for o in ["TP", "TN", "FP", "FN"]]
    return pd.concat(picks, ignore_index=True)


def run() -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(C.FIGURES_DIR), ensure_dir(C.TABLES_DIR)
    df = load_processed()
    test = get_split(df, "test")
    rows = []
    for mid in ["word_lr", "char_lr"]:
        pred_df = pd.read_csv(C.RESULTS_DIR / "baseline_predictions.csv")
        preds = pred_df[pred_df.model_id == mid].drop(columns=["model_id"])
        predictor = BaselinePredictor(C.MODELS_DIR / f"baseline_{mid}.joblib", mid)
        pipe = predictor.pipeline
        g = global_top_features(pipe, 15)
        g.insert(0, "model_id", mid)
        g.to_csv(C.TABLES_DIR / f"global_features_{mid}.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, (direction, color) in zip(axes, [("Scam", "#c0392b"), ("Legitimate", "#2874a6")]):
            sub = g[g.direction == direction].iloc[::-1]
            ax.barh(sub.feature, sub.weight, color=color)
            ax.set_title(f"Top features -> {direction}", fontsize=12)
            ax.set_xlabel("Logistic-regression weight (towards Scam)", fontsize=10)
            ax.set_ylabel("TF-IDF feature", fontsize=10)
            ax.tick_params(labelsize=9)
        fig.suptitle(f"Global feature weights: {pipe.steps[-1][1].__class__.__name__} on TF-IDF-{mid.split('_')[0]}", fontsize=13)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / f"explain_global_{mid}.png", dpi=150)
        plt.close(fig)

        ex = pick_examples(preds, test)
        proba = lambda t, pp=pipe: pipeline_scores(pp, t)[1]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, (_, r) in zip(axes.ravel(), ex.iterrows()):
            lin = linear_feature_contributions(pipe, r.cleaned_message, 8)
            lime = lime_explain(proba, r.cleaned_message, 8)
            rows.append({"model_id": mid, "outcome": r.outcome, "message_id": r.message_id, "language": r.language,
                         "true_label": C.ID2LABEL[int(r.y_true)], "predicted_label": C.ID2LABEL[int(r.y_pred)],
                         "p_scam": float(r.p_scam), "message": r.cleaned_message[:200],
                         "linear_top_features": "; ".join(f"{f}:{c:+.3f}" for f, c in lin),
                         "lime_top_features": "; ".join(f"{f}:{c:+.3f}" for f, c in lime)})
            names, vals = zip(*lin[::-1]) if lin else ((), ())
            ax.barh(names, vals, color=["#c0392b" if v > 0 else "#2874a6" for v in vals])
            ax.set_title(f"{r.outcome}: true={C.ID2LABEL[int(r.y_true)]}, pred={C.ID2LABEL[int(r.y_pred)]} (p_scam={r.p_scam:.2f})",
                         fontsize=10)
            ax.set_xlabel("Contribution to Scam score (w*tfidf)", fontsize=9)
            ax.set_ylabel("Feature", fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"Per-message additive contributions ({mid}); red = towards Scam. Model attributions, not causal evidence.",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / f"explain_examples_{mid}.png", dpi=150)
        plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES_DIR / "explain_examples.csv", index=False)
    return out


if __name__ == "__main__":
    print(run().to_string(index=False))
