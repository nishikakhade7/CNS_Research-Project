"""Shared metric computation — used by both internal (Stage 12) and external (Stage 15)
evaluation so the two are guaranteed to compute things identically."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)


def full_classification_report(y_true, y_pred, labels: list) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc),
        "per_class": {
            str(lbl): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lbl, p, r, f, s in zip(labels, prec, rec, f1, support)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [str(l) for l in labels],
    }
