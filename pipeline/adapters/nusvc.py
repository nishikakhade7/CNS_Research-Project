"""NuSVC adapter — compact, scaled, count-transformed input.

Kernel methods are scale-sensitive and don't handle very high dimensionality well
(Schölkopf et al., Neural Computation, 2000), so this adapter caps features and always
uses the log1p+scaled representation already produced by Stages 05/06.
"""

from __future__ import annotations

from sklearn.svm import NuSVC


def build_model(cfg: dict):
    s = cfg["models"]["nusvc"]
    return NuSVC(nu=s["nu"], kernel=s["kernel"], probability=True, random_state=cfg["seed"])


def select_view(feature_names: list[str], top_ranked: list[str], cfg: dict) -> list[str]:
    max_f = cfg["models"]["nusvc"]["max_features"]
    capped = top_ranked[:max_f]
    return [f"{c}__scaled" for c in capped]
