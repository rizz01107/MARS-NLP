"""Metrics, cluster-bootstrap confidence intervals, paired tests and latency benchmarking."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import config as C


# --------------------------------------------------------------------------- point metrics
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Accuracy, scam-class precision/recall/F1, macro & weighted F1, confusion counts.

    ``precision``/``recall``/``f1_scam`` refer to the positive class (Scam = 1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {k: np.nan for k in ["n", "accuracy", "precision_scam", "recall_scam", "f1_scam", "macro_precision",
                                    "macro_recall", "macro_f1", "weighted_f1", "tn", "fp", "fn", "tp"]} | {"n": 0}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_scam": float(p[1]), "recall_scam": float(r[1]), "f1_scam": float(f[1]),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1], zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def subset_status(y_true: Sequence[int]) -> tuple[bool, str]:
    """Is a subset large enough to report? Needs MIN_EVAL_N samples and MIN_EVAL_PER_CLASS of each class."""
    y = np.asarray(y_true)
    n, n1 = len(y), int((y == 1).sum())
    n0 = n - n1
    if n < C.MIN_EVAL_N:
        return False, f"insufficient (n={n} < {C.MIN_EVAL_N})"
    if min(n0, n1) < C.MIN_EVAL_PER_CLASS:
        return False, f"insufficient (class counts {n0}/{n1}; need >= {C.MIN_EVAL_PER_CLASS} each)"
    return True, "ok"


def reliability_note(n_groups: int) -> str:
    """Flag subsets with too few independent (near-duplicate) groups: n rows overstates the effective sample size."""
    if n_groups < C.MIN_EVAL_GROUPS:
        return f"descriptive only: {n_groups} independent groups (< {C.MIN_EVAL_GROUPS}); CI/point estimates unreliable"
    return "ok"


# --------------------------------------------------------------------------- confusion-count based metrics
def _f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def metric_from_counts(c: np.ndarray, metric: str) -> np.ndarray:
    """``c`` has shape (..., 4) = (tn, fp, fn, tp). Supports macro_f1, accuracy, f1_scam, recall_scam, precision_scam."""
    tn, fp, fn, tp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    if metric == "macro_f1":
        return 0.5 * (_f1_from_counts(tp, fp, fn) + _f1_from_counts(tn, fn, fp))
    if metric == "accuracy":
        return (tn + tp) / np.maximum(tn + fp + fn + tp, 1)
    if metric == "f1_scam":
        return _f1_from_counts(tp, fp, fn)
    if metric == "recall_scam":
        return tp / np.maximum(tp + fn, 1)
    if metric == "precision_scam":
        return tp / np.maximum(tp + fp, 1)
    raise ValueError(f"unsupported metric {metric}")


def group_confusions(y_true, y_pred, groups, index: Optional[pd.Index] = None) -> pd.DataFrame:
    """Per-group (tn, fp, fn, tp) table indexed by group id."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    g = pd.Series(np.asarray(groups))
    codes, uniques = pd.factorize(g)
    k = len(uniques)
    cells = [((y_true == 0) & (y_pred == 0)), ((y_true == 0) & (y_pred == 1)),
             ((y_true == 1) & (y_pred == 0)), ((y_true == 1) & (y_pred == 1))]
    arr = np.stack([np.bincount(codes, weights=c.astype(float), minlength=k) for c in cells], axis=1)
    df = pd.DataFrame(arr, index=uniques, columns=["tn", "fp", "fn", "tp"])
    return df.reindex(index, fill_value=0.0) if index is not None else df


def cluster_bootstrap_ci(y_true, y_pred, groups, metric: str = "macro_f1", n_boot: int = 1000,
                         alpha: float = 0.05, seed: int = C.SEED) -> dict:
    """Percentile CI from a *cluster* bootstrap (resampling groups, not rows).

    Rows within a near-duplicate group are not independent, so a naive row bootstrap would give overly narrow
    intervals (cf. Field & Welsh, 2007). With few groups the interval is wide - and that is the honest answer.
    """
    gc = group_confusions(y_true, y_pred, groups)
    arr = gc.to_numpy()
    k = len(arr)
    point = float(metric_from_counts(arr.sum(0), metric))
    if k < 2:
        return {"point": point, "ci_low": np.nan, "ci_high": np.nan, "n_groups": k}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = metric_from_counts(arr[idx].sum(1), metric)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "n_groups": int(k)}


def paired_cluster_bootstrap_diff(y_true, pred_a, pred_b, groups, metric: str = "macro_f1", n_boot: int = 1000,
                                  alpha: float = 0.05, seed: int = C.SEED) -> dict:
    """CI for metric(A) - metric(B) on the same samples via paired cluster bootstrap."""
    ga = group_confusions(y_true, pred_a, groups)
    gb = group_confusions(y_true, pred_b, groups, index=ga.index)
    A, B = ga.to_numpy(), gb.to_numpy()
    k = len(A)
    diff = float(metric_from_counts(A.sum(0), metric) - metric_from_counts(B.sum(0), metric))
    if k < 2:
        return {"diff": diff, "ci_low": np.nan, "ci_high": np.nan, "n_groups": k, "ci_excludes_zero": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = metric_from_counts(A[idx].sum(1), metric) - metric_from_counts(B[idx].sum(1), metric)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": diff, "ci_low": float(lo), "ci_high": float(hi), "n_groups": int(k),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test on discordant pairs. NOTE: assumes independent samples; with clustered
    (templated) data the p-value is optimistic, so the cluster bootstrap is the primary inference tool."""
    from scipy.stats import binomtest

    y = np.asarray(y_true)
    a_ok, b_ok = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b = int((a_ok & ~b_ok).sum())   # A right, B wrong
    c = int((~a_ok & b_ok).sum())   # A wrong, B right
    n = b + c
    p = float(binomtest(min(b, c), n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def metrics_with_ci(df: pd.DataFrame, y_pred, group_col: str = "group_id", n_boot: int = 1000) -> dict:
    """Overall metrics + cluster-bootstrap CI for macro-F1 on a dataframe with y_true column ``label_id``."""
    m = compute_metrics(df["label_id"].values, y_pred)
    ci = cluster_bootstrap_ci(df["label_id"].values, y_pred, df[group_col].values, "macro_f1", n_boot)
    m.update({"macro_f1_ci_low": ci["ci_low"], "macro_f1_ci_high": ci["ci_high"], "n_groups": ci["n_groups"]})
    return m


# --------------------------------------------------------------------------- latency / size
def benchmark_latency(predict_fn: Callable[[list[str]], object], texts: Sequence[str], n_single: int = 200,
                      n_batch: int = 1000, warmup: int = 5) -> dict:
    """Computational benchmark (wall-clock) of ``predict_fn`` on the *current machine*.

    This is NOT an edge-device benchmark. Single-message latency calls ``predict_fn([text])`` repeatedly.
    """
    texts = list(texts)
    for t in texts[:warmup]:
        predict_fn([t])
    sample = texts[:n_single] if len(texts) >= n_single else (texts * (n_single // max(len(texts), 1) + 1))[:n_single]
    times = []
    for t in sample:
        s = time.perf_counter()
        predict_fn([t])
        times.append((time.perf_counter() - s) * 1000)
    batch = (texts * (n_batch // max(len(texts), 1) + 1))[:n_batch]
    s = time.perf_counter()
    predict_fn(batch)
    batch_ms = (time.perf_counter() - s) * 1000 / len(batch)
    return {"latency_single_ms_mean": float(np.mean(times)), "latency_single_ms_p50": float(np.percentile(times, 50)),
            "latency_single_ms_p95": float(np.percentile(times, 95)), "latency_batch_ms_per_msg": float(batch_ms),
            "n_single": len(sample), "n_batch": len(batch)}


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)
