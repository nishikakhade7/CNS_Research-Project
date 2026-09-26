"""Hardware / environment capture. Never claims GPU execution that didn't actually happen."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import psutil


def capture_environment() -> dict[str, Any]:
    """Snapshot the runtime environment: Python, packages, CPU/RAM, and honest GPU/CUDA status."""
    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
    }

    packages = {}
    for mod_name in ["numpy", "pandas", "sklearn", "lightgbm", "imblearn", "torch", "shap", "pyarrow"]:
        try:
            mod = __import__(mod_name)
            packages[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[mod_name] = None
    env["packages"] = packages

    # GPU / CUDA — verified, not assumed
    cuda_info: dict[str, Any] = {"torch_cuda_available": False, "reason_if_unavailable": None}
    try:
        import torch

        cuda_info["torch_version"] = torch.__version__
        cuda_info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cuda_info["gpu_name"] = torch.cuda.get_device_name(0)
            cuda_info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            cuda_info["cuda_runtime_version"] = torch.version.cuda
        else:
            cuda_info["reason_if_unavailable"] = "torch.cuda.is_available() returned False on this install/machine"
    except ImportError as e:
        cuda_info["reason_if_unavailable"] = f"torch not importable: {e}"
    env["cuda"] = cuda_info

    # LightGBM GPU support — verified by attempting a trivial GPU fit, not assumed from the package existing
    lgb_gpu = {"supported": False, "reason_if_unavailable": None}
    try:
        import lightgbm as lgb
        import numpy as np

        try:
            m = lgb.LGBMClassifier(device_type="gpu", n_estimators=1, verbose=-1)
            m.fit(np.random.rand(20, 3), np.array([0, 1] * 10))
            lgb_gpu["supported"] = True
        except Exception as e:
            lgb_gpu["reason_if_unavailable"] = (
                f"device_type='gpu' fit failed ({e.__class__.__name__}: {e}); "
                "the installed lightgbm wheel is the standard CPU build, not a GPU/CUDA build"
            )
    except ImportError as e:
        lgb_gpu["reason_if_unavailable"] = f"lightgbm not importable: {e}"
    env["lightgbm_gpu"] = lgb_gpu

    # nvidia-smi, best-effort, informational only
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        env["nvidia_smi"] = out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        env["nvidia_smi"] = None

    return env
