"""QDA adapter — strongly reduced, dense, scaled input.

QDA fits a full covariance matrix per class; MalDroid's smallest class (Adware, 1,253
samples) cannot support a stable covariance estimate over hundreds of raw dimensions, so
this adapter caps the feature count at `models.qda.max_features` (config) and always uses
the scaled representation.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def build_model(cfg: dict):
    q = cfg["models"]["qda"]
    return QuadraticDiscriminantAnalysis(reg_param=q["reg_param"])


def select_view(feature_names: list[str], top_ranked: list[str], cfg: dict) -> list[str]:
    """Cap to max_features using the already-computed ranking (train-only) — never
    re-derives ranking from val/test/external data."""
    max_f = cfg["models"]["qda"]["max_features"]
    capped = top_ranked[:max_f]
    return [f"{c}__scaled" for c in capped]
