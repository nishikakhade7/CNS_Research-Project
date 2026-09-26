"""LightGBM adapter — raw/selected numeric features, no scaling.

Tree-based splits are scale-invariant (Ke et al., NeurIPS 2017), so this adapter always
uses the raw (log1p-transformed where applicable) representation, never the scaled columns.
GPU usage is only ever requested if `capture_environment()` verified it actually works on
this install — never assumed from the package merely being present.
"""

from __future__ import annotations

import lightgbm as lgb


def build_model(cfg: dict, gpu_verified: bool):
    m = cfg["models"]["lightgbm"]
    device = "gpu" if gpu_verified else "cpu"
    return lgb.LGBMClassifier(
        n_estimators=m["n_estimators"],
        num_leaves=m["num_leaves"],
        learning_rate=m["learning_rate"],
        device_type=device,
        random_state=cfg["seed"],
        verbose=-1,
    ), device


def select_view(top_ranked: list[str], k: int) -> list[str]:
    return top_ranked[:k]
