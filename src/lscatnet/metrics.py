"""Evaluation metrics for multi-class smoke-scene classification."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_far_frr(cm: np.ndarray):
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - fp - fn - tp
    far = fp / np.maximum(fp + tn, 1)
    frr = fn / np.maximum(fn + tp, 1)
    return far, frr, tp, fp, tn, fn


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict:
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    far, frr, tp, fp, tn, fn = compute_far_frr(cm)

    per_acc = tp / np.maximum(cm.sum(axis=1), 1)
    per_prec = tp / np.maximum(tp + fp, 1)
    per_rec = tp / np.maximum(tp + fn, 1)
    per_f1 = 2 * per_prec * per_rec / np.maximum(per_prec + per_rec, 1e-12)
    per_spec = tn / np.maximum(tn + fp, 1)

    result = {
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "far_macro": float(far.mean()),
            "frr_macro": float(frr.mean()),
        },
        "per_class": [],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=list(class_names), digits=4, zero_division=0),
    }

    if y_prob is not None:
        try:
            result["overall"]["auc_macro_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(num_classes)))
            )
        except Exception:
            result["overall"]["auc_macro_ovr"] = None

    for idx, name in enumerate(class_names):
        result["per_class"].append(
            {
                "class": name,
                "samples": int(cm.sum(axis=1)[idx]),
                "accuracy": float(per_acc[idx]),
                "precision": float(per_prec[idx]),
                "recall": float(per_rec[idx]),
                "f1": float(per_f1[idx]),
                "specificity": float(per_spec[idx]),
                "far": float(far[idx]),
                "frr": float(frr[idx]),
            }
        )
    return result


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap CIs for headline metrics.

    This is useful for reporting confidence intervals requested by reviewers.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = {"accuracy": [], "f1_macro": [], "far_macro": []}
    has_prob = y_prob is not None
    if has_prob:
        values["auc_macro_ovr"] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Skip resamples missing classes for AUC.
        m = compute_multiclass_metrics(
            y_true[idx],
            y_pred[idx],
            y_prob[idx] if has_prob else None,
            class_names,
        )
        values["accuracy"].append(m["overall"]["accuracy"])
        values["f1_macro"].append(m["overall"]["f1_macro"])
        values["far_macro"].append(m["overall"]["far_macro"])
        if has_prob and m["overall"].get("auc_macro_ovr") is not None:
            values["auc_macro_ovr"].append(m["overall"]["auc_macro_ovr"])

    ci = {}
    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)
    for key, vals in values.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            continue
        ci[key] = {
            "mean": float(arr.mean()),
            "lower": float(np.percentile(arr, lo_q)),
            "upper": float(np.percentile(arr, hi_q)),
        }
    return ci


def roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]):
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    data = {}
    fprs = []
    mean_tpr = None
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        data[name] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
        fprs.append(fpr)
    all_fpr = np.unique(np.concatenate(fprs))
    mean_tpr = np.zeros_like(all_fpr)
    for name in class_names:
        mean_tpr += np.interp(all_fpr, data[name]["fpr"], data[name]["tpr"])
    mean_tpr /= num_classes
    data["macro"] = {"fpr": all_fpr, "tpr": mean_tpr, "auc": float(auc(all_fpr, mean_tpr))}
    return data
