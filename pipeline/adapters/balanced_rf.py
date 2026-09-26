"""Balanced Random Forest adapter — raw/selected numeric features, no scaling.

Rebalances classes internally via bootstrap sampling (Chen, Liaw & Breiman, UC Berkeley
Tech Report 666, 2004) — this is why it is run WITHOUT any Stage 09 resampling applied on
top; combining both would double-correct the imbalance.
"""

from __future__ import annotations

from imblearn.ensemble import BalancedRandomForestClassifier


def build_model(cfg: dict):
    m = cfg["models"]["balanced_rf"]
    return BalancedRandomForestClassifier(
        n_estimators=m["n_estimators"], random_state=cfg["seed"],
        sampling_strategy="all", replacement=True, bootstrap=False,
    )


def select_view(top_ranked: list[str], k: int) -> list[str]:
    return top_ranked[:k]
