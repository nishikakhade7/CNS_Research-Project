"""Sanitize machine-specific absolute paths in runs/real JSON files."""
import json
import pathlib

PROJECT_ROOT_MARKER = "<project_root>"

def sanitize_value(v, prefix: str):
    if isinstance(v, str):
        return v.replace(prefix, PROJECT_ROOT_MARKER + "/").replace("\\", "/")
    if isinstance(v, dict):
        return {k: sanitize_value(vv, prefix) for k, vv in v.items()}
    if isinstance(v, list):
        return [sanitize_value(vv, prefix) for vv in v]
    return v

MACHINE_PREFIX = r"C:\Users\Deep\OneDrive\Desktop\Coding\App-Malware-CNS-Project"

targets = [
    pathlib.Path("runs/real/status.json"),
    pathlib.Path("runs/real/artifacts/stage14_lock_manifest.json"),
]

for target in targets:
    with open(target, encoding="utf-8") as f:
        data = json.load(f)
    sanitized = sanitize_value(data, MACHINE_PREFIX)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)
    # Quick verify: no machine prefix remains
    text = target.read_text(encoding="utf-8")
    if "C:\\Users\\Deep" in text or "C:/Users/Deep" in text:
        raise RuntimeError(f"Machine path still present in {target}")
    print(f"OK: {target}")
