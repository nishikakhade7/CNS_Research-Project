"""verify_artifacts.py — one-shot verification script, runs from project root."""
import json, os, collections
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
arts = os.path.join(ROOT, "runs", "real", "artifacts")

# ---------- artifact schemas ----------
for name, fname in [
    ("stage05_static", "stage05_static_features.parquet"),
    ("stage06_dyn139", "stage06_dynamic_139_features.parquet"),
    ("stage06_dyn470", "stage06_dynamic_470_features.parquet"),
    ("stage07_fused",  "stage07_fused_features.parquet"),
]:
    path = os.path.join(arts, fname)
    pf   = pq.ParquetFile(path)
    sch  = pf.schema_arrow
    cols = [sch.field(i).name for i in range(len(sch))]
    meta = [c for c in cols if c in ("sample_id","split","Class","label","label_canonical","dup_group")]
    scl  = [c for c in cols if c.endswith("__scaled")]
    feat = [c for c in cols if c not in set(meta) and not c.endswith("__scaled")]
    print(f"{name}: total={len(cols)}  meta={len(meta)}  feat={len(feat)}  scaled={len(scl)}")
    print(f"  meta_cols  = {meta}")
    print(f"  feat[:3]   = {feat[:3]}")

# ---------- selection JSON ----------
sel = json.load(open(os.path.join(arts, "stage08_feature_selection.json")))
print("\n--- feature_selection.json ---")
for mod in ("static","dynamic_139","dynamic_470","fused"):
    entry = sel[mod]
    nr = entry.get("n_raw_features", "?")
    meths = entry.get("methods", {})
    ks = {}
    for mname, mval in meths.items():
        ks[mname] = sorted(mval.keys(), key=int)
    print(f"  {mod}: n_raw={nr}  K_values={ks.get('lightgbm_importance')}")

# ---------- split counts from readable artifact ----------
pf2 = pq.ParquetFile(os.path.join(arts, "stage06_dynamic_139_features.parquet"))
split_counts = collections.Counter()
for batch in pf2.iter_batches(batch_size=2000, columns=["split"]):
    for s in batch.column("split").to_pylist():
        split_counts[s] += 1
print("\n--- Split counts (from dyn139) ---")
for k, v in sorted(split_counts.items()):
    print(f"  {k}: {v}")

# ---------- lock manifest ----------
lock = json.load(open(os.path.join(arts, "stage14_lock_manifest.json")))
vm   = lock.get("verified_test_metrics", {})
print("\n--- stage14_lock_manifest ---")
print(f"  n_features: {lock.get('n_features')}")
print(f"  modality:   {lock.get('selected_modality')}")
print(f"  accuracy:   {vm.get('accuracy')}")
print(f"  macro_f1:   {vm.get('macro_f1')}")
print(f"  weighted_f1: {vm.get('weighted_f1')}")
print(f"  balanced_acc:{vm.get('balanced_accuracy')}")
print(f"  retrain_sec: {lock.get('retrain_time_sec')}")
print(f"  sha256:      {lock.get('frozen_artifact_sha256_prefix')}")

# ---------- stage09 files ----------
stage09 = [f for f in os.listdir(arts) if "stage09" in f]
print("\n--- stage09 artifacts ---", stage09)

# ---------- manifest schema ----------
mf_path = os.path.join(ROOT, "runs", "real", "manifests", "maldroid_split_manifest.parquet")
pf3 = pq.ParquetFile(mf_path)
sch3 = pf3.schema_arrow
print("\n--- manifest schema ---")
print(f"  cols: {[sch3.field(i).name for i in range(len(sch3))]}")
print(f"  rows: {pf3.metadata.num_rows}")
