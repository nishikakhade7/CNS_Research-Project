"""
generate_all_figures.py
=======================
Generates ALL publication-quality figures for Group 3 final documentation.
Source: runs/real artifacts only. No invented data.

Run from project root:
    python docs/final/assets/generate_all_figures.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_REAL     = os.path.join(PROJECT_ROOT, "runs", "real")
ARTIFACTS     = os.path.join(RUNS_REAL, "artifacts")
METRICS       = os.path.join(RUNS_REAL, "metrics")
REPORTS       = os.path.join(RUNS_REAL, "reports")
OUT_DIR       = os.path.join(PROJECT_ROOT, "docs", "final", "assets", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 200

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
with open(os.path.join(RUNS_REAL, "status.json")) as f:
    status = json.load(f)

with open(os.path.join(METRICS, "stage12_internal_evaluation.json")) as f:
    eval12 = json.load(f)

with open(os.path.join(METRICS, "stage13_ablation_results.json")) as f:
    ablation = json.load(f)

with open(os.path.join(ARTIFACTS, "stage14_lock_manifest.json")) as f:
    lock = json.load(f)

# Class weights
class_weights = {
    "Adware":       1.85249500998004,
    "Banking":      1.1048809523809524,
    "Benign":       1.2890277777777779,
    "Riskware":     0.9116895874263261,
    "SMS Malware":  0.5943643932116555,
}

# Split distributions (from stage09 imbalance report)
train_dist  = {"SMS Malware": 3123, "Riskware": 2036, "Banking": 1680, "Benign": 1440, "Adware": 1002}
val_dist    = {"SMS Malware": 390,  "Riskware": 255,  "Banking": 210,  "Benign": 176,  "Adware": 125}
test_dist   = {"SMS Malware": 391,  "Riskware": 255,  "Banking": 210,  "Benign": 179,  "Adware": 126}

CLASSES      = ["Adware", "Banking", "Benign", "Riskware", "SMS Malware"]
modalities   = ["static", "dynamic_139", "dynamic_470", "fused"]
mod_labels   = ["Static\n(270 feat)", "Dynamic-139\n(139 feat)", "Dynamic-470\n(270 feat)", "Fused\n(270 feat)"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path

# ===========================================================================
# FIG 01  Confusion matrix -- Final locked model (fused LightGBM, TEST split)
# ===========================================================================
cm = np.array(lock["verified_test_metrics"]["confusion_matrix"])
labels = lock["verified_test_metrics"]["confusion_matrix_labels"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Figure 1 -- Confusion Matrix\nFinal LightGBM (Fused, TEST split)", fontsize=13)
plt.tight_layout()
savefig("fig01_confusion_matrix_fused_test.png")

# ===========================================================================
# FIG 02  Per-class precision / recall / F1 -- final model
# ===========================================================================
pc = lock["verified_test_metrics"]["per_class"]
x = np.arange(len(CLASSES))
w = 0.25
prec = [pc[c]["precision"] for c in CLASSES]
rec  = [pc[c]["recall"]    for c in CLASSES]
f1s  = [pc[c]["f1"]        for c in CLASSES]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w,  prec, w, label="Precision", color="#4472C4")
ax.bar(x,      rec,  w, label="Recall",    color="#ED7D31")
ax.bar(x + w,  f1s,  w, label="F1",        color="#A9D18E")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylim(0.85, 1.01)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Figure 2 -- Per-Class Metrics\nFinal LightGBM (Fused, TEST split)", fontsize=13)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig02_per_class_metrics_fused_test.png")

# ===========================================================================
# FIG 03  Modality comparison (accuracy + macro-F1)
# ===========================================================================
accs = [ablation[m]["lightgbm"]["accuracy"]  for m in modalities]
mf1s = [ablation[m]["lightgbm"]["macro_f1"]  for m in modalities]

x = np.arange(len(modalities))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, accs, w, label="Accuracy",  color="#4472C4")
ax.bar(x + w/2, mf1s, w, label="Macro-F1", color="#ED7D31")
ax.set_xticks(x)
ax.set_xticklabels(mod_labels, fontsize=10)
ax.set_ylim(0.90, 1.00)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Figure 3 -- Modality Comparison\nLightGBM (TEST split, all modalities)", fontsize=13)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for i, (a, m) in enumerate(zip(accs, mf1s)):
    ax.text(i - w/2, a + 0.001, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, m + 0.001, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
savefig("fig03_modality_comparison.png")

# ===========================================================================
# FIG 04  Class distribution -- train / val / test
# ===========================================================================
classes_ordered = ["Adware", "Banking", "Benign", "Riskware", "SMS Malware"]
train_vals = [train_dist[c] for c in classes_ordered]
val_vals   = [val_dist[c]   for c in classes_ordered]
test_vals  = [test_dist[c]  for c in classes_ordered]

x = np.arange(len(classes_ordered))
w = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w,  train_vals, w, label="Train",      color="#4472C4")
ax.bar(x,      val_vals,   w, label="Validation", color="#ED7D31")
ax.bar(x + w,  test_vals,  w, label="Test",       color="#A9D18E")
ax.set_xticks(x)
ax.set_xticklabels(classes_ordered, fontsize=10)
ax.set_ylabel("Sample Count", fontsize=12)
ax.set_title("Figure 4 -- Class Distribution\nTrain / Validation / Test Splits (MalDroid2020)", fontsize=13)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig04_class_distribution_splits.png")

# ===========================================================================
# FIG 05  Feature type distribution (static columns)
# ===========================================================================
type_labels = [
    "Binary\nNumeric",
    "Count/\nContinuous",
    "All-Missing\n(dropped)",
    "Zero-Variance\n(dropped)",
    "High-Card.\nString (dropped)",
    "Categorical\nString (dropped)",
    "Identity\n(dropped)",
]
type_counts = [31317, 335, 18931, 22, 4, 11, 1]
usable      = [True,  True, False, False, False, False, False]
colors_bar  = ["#4472C4" if u else "#C0504D" for u in usable]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(type_labels)), type_counts, color=colors_bar, edgecolor="white")
ax.set_xticks(range(len(type_labels)))
ax.set_xticklabels(type_labels, fontsize=9)
ax.set_ylabel("Column Count", fontsize=12)
ax.set_title("Figure 5 -- Static Feature Type Distribution\n(50,620 total columns, MalDroid2020 static)", fontsize=13)
for bar, cnt in zip(bars, type_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{cnt:,}", ha="center", va="bottom", fontsize=9)
legend_patches = [
    mpatches.Patch(color="#4472C4", label="Usable (31,652 total)"),
    mpatches.Patch(color="#C0504D", label="Dropped (18,968 total)"),
]
ax.legend(handles=legend_patches, fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig05_feature_type_distribution.png")

# ===========================================================================
# FIG 06  Feature selection reduction
# ===========================================================================
stages6 = ["Raw Static\n(31,652 cols)", "After Variance\nFilter (31,652)", "Selected K=270\n(Fused model)"]
vals6   = [31652, 31652, 270]
col6    = ["#4472C4", "#4472C4", "#70AD47"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(stages6, vals6, color=col6)
ax.set_xlabel("Feature Count", fontsize=12)
ax.set_title("Figure 6 -- Feature Count Reduction\nFrom raw static columns to final 270 selected", fontsize=13)
for i, v in enumerate(vals6):
    ax.text(v + 150, i, f"{v:,}", va="center", fontsize=10)
ax.set_xlim(0, 36000)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig06_feature_selection_reduction.png")

# ===========================================================================
# FIG 07  Stage execution times
# ===========================================================================
stage_names = ["S01 Audit", "S02 Manifest", "S03 Typing", "S04 Split",
               "S05 Static", "S06 Dynamic", "S07 Fusion", "S08 Selection",
               "S09 Imbalance", "S11 Train", "S12 Eval", "S13 Ablation",
               "S14 Lock", "S15 External", "S16 Report"]
stage_times = [361.73, 28.48, 125.24, 1.28,
               175.33, 1.6, 168.4, 2464.7,
               288.76, 5.5, 0.31, 323.76,
               247.43, 0.1, 1.34]
colors_stage = ["#C0504D" if t > 500 else "#4472C4" for t in stage_times]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(stage_names, stage_times, color=colors_stage, edgecolor="white")
ax.set_ylabel("Duration (seconds)", fontsize=12)
ax.set_title("Figure 7 -- Pipeline Stage Execution Times (runs/real)", fontsize=13)
ax.set_xticklabels(stage_names, rotation=35, ha="right", fontsize=9)
for bar, t in zip(bars, stage_times):
    label = f"{t:.0f}s" if t >= 1 else f"{t:.2f}s"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            label, ha="center", va="bottom", fontsize=7)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig07_stage_execution_times.png")

# ===========================================================================
# FIG 08  Class weights
# ===========================================================================
cw_classes = list(class_weights.keys())
cw_values  = [class_weights[c] for c in cw_classes]
bar_colors = ["#4472C4" if v > 1.0 else "#ED7D31" for v in cw_values]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(cw_classes, cw_values, color=bar_colors, edgecolor="white")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Neutral weight = 1.0")
ax.set_ylabel("Class Weight", fontsize=12)
ax.set_title("Figure 8 -- Class Weights Applied in LightGBM Training\n(Inversely proportional to training set frequency)", fontsize=13)
ax.legend(fontsize=10)
for i, (c, v) in enumerate(zip(cw_classes, cw_values)):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig08_class_weights.png")

# ===========================================================================
# FIG 09  Three-metric comparison across modalities
# ===========================================================================
metrics_to_plot = [
    ("Accuracy",          [ablation[m]["lightgbm"]["accuracy"]          for m in modalities]),
    ("Macro-F1",          [ablation[m]["lightgbm"]["macro_f1"]          for m in modalities]),
    ("Balanced Accuracy", [ablation[m]["lightgbm"]["balanced_accuracy"] for m in modalities]),
]
x = np.arange(len(modalities))
w = 0.25
offsets = [-w, 0, w]
met_colors = ["#4472C4", "#ED7D31", "#A9D18E"]

fig, ax = plt.subplots(figsize=(9, 5))
for idx, (met_name, vals) in enumerate(metrics_to_plot):
    ax.bar(x + offsets[idx], vals, w, label=met_name, color=met_colors[idx])
ax.set_xticks(x)
ax.set_xticklabels(mod_labels, fontsize=10)
ax.set_ylim(0.90, 1.00)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Figure 9 -- Test Metrics Across Modalities\n(Accuracy, Macro-F1, Balanced Accuracy)", fontsize=13)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig09_modality_metrics_comparison.png")

# ===========================================================================
# FIG 10  Static vs Dynamic in selected 270 features
# ===========================================================================
feat_cols = lock["feature_columns"]
n_static  = sum(1 for f in feat_cols if f.startswith("static__"))
n_dynamic = sum(1 for f in feat_cols if f.startswith("dyn470__"))
n_other   = len(feat_cols) - n_static - n_dynamic

labels_pie = ["Static features", "Dynamic-470 features"]
sizes_pie  = [n_static, n_dynamic]
if n_other > 0:
    labels_pie.append("Other")
    sizes_pie.append(n_other)

fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    sizes_pie, labels=labels_pie, autopct="%1.1f%%",
    colors=["#4472C4", "#ED7D31"],
    startangle=140, pctdistance=0.75,
)
for t in autotexts:
    t.set_fontsize(11)
ax.set_title(f"Figure 10 -- Feature Source Contribution\n(270 selected features, locked fused model)\n"
             f"Static: {n_static}  |  Dynamic-470: {n_dynamic}", fontsize=11)
plt.tight_layout()
savefig("fig10_static_dynamic_feature_split.png")

# ===========================================================================
# FIG 11  MalDroid class imbalance in full dataset
# ===========================================================================
maldroid_full = {"Adware": 1253, "Banking": 2100, "Benign": 1795, "Riskware": 2546, "SMS Malware": 3904}
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(list(maldroid_full.keys()), list(maldroid_full.values()), color="#4472C4", edgecolor="white")
ax.set_ylabel("Sample Count", fontsize=12)
ax.set_title("Figure 11 -- MalDroid2020 Full Dataset\nClass Distribution (11,598 samples total)", fontsize=13)
for i, (c, v) in enumerate(maldroid_full.items()):
    ax.text(i, v + 20, str(v), ha="center", va="bottom", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig11_maldroid_class_distribution.png")

# ===========================================================================
# FIG 12  Pipeline stage durations (horizontal bar)
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 7))
y = range(len(stage_names))
col12 = ["#C0504D" if t > 500 else "#4472C4" for t in stage_times]
ax.barh(list(y), stage_times, color=col12, edgecolor="white")
ax.set_yticks(list(y))
ax.set_yticklabels(stage_names, fontsize=10)
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_title("Figure 12 -- Pipeline Stage Durations (Horizontal)\n(runs/real full chronology)", fontsize=13)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for i, t in enumerate(stage_times):
    label = f"{t:.1f}s" if t >= 1 else f"{t:.2f}s"
    ax.text(t + 5, i, label, va="center", fontsize=8)
plt.tight_layout()
savefig("fig12_pipeline_durations_horizontal.png")

# ===========================================================================
# FIG 13  Top 20 selected feature names labeled by origin
# ===========================================================================
top20 = feat_cols[:20]
top20_short = []
for f in top20:
    s = f.replace("static__", "[S] ").replace("dyn470__", "[D] ")
    if len(s) > 50:
        s = s[:47] + "..."
    top20_short.append(s)

col13 = ["#4472C4" if f.startswith("static__") else "#ED7D31" for f in top20]
fig, ax = plt.subplots(figsize=(11, 7))
y_pos = list(range(len(top20_short)))
ax.barh(y_pos, [1]*20, color=col13, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(top20_short, fontsize=8)
ax.set_xticks([])
ax.set_title("Figure 13 -- Top 20 Selected Features (Fused Model)\n[S] = Static feature   [D] = Dynamic-470 feature", fontsize=12)
legend_patches2 = [
    mpatches.Patch(color="#4472C4", label="Static feature"),
    mpatches.Patch(color="#ED7D31", label="Dynamic-470 feature"),
]
ax.legend(handles=legend_patches2, fontsize=9, loc="lower right")
plt.tight_layout()
savefig("fig13_top20_selected_features.png")

print("\nAll 13 figures generated successfully.")
print(f"Output directory: {OUT_DIR}")
