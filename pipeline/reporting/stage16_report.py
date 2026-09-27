"""Stage 16 — Final Reporting.

Generates the required plots (saved as PNG, matplotlib source kept reproducible via this
script) and one consolidated markdown report pulling together every prior stage's real
output — nothing here is invented; every number is read back from a prior stage's saved
JSON/metrics file.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.core.runlog import get_stage_logger
from pipeline.core.stage import Stage, StageContext

log = get_stage_logger("stage16_report")


def _bar_plot(labels, values, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#2f6b4f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(values) * 1.15) if values else 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _confusion_matrix_plot(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=8,
                     color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class FinalReportStage(Stage):
    name = "stage16_report"
    depends_on = ["stage12_eval", "stage13_ablation", "stage15_external"]

    def config_section(self, cfg: dict) -> dict:
        return {}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        plots_dir = ctx.run_dir / "plots"
        reports_dir = ctx.run_dir / "reports"

        with open(ctx.run_dir / "metrics" / "stage12_internal_evaluation.json", encoding="utf-8") as f:
            internal = json.load(f)
        with open(ctx.run_dir / "metrics" / "stage13_ablation_results.json", encoding="utf-8") as f:
            ablation = json.load(f)
        with open(ctx.run_dir / "metrics" / "stage15_external_validation.json", encoding="utf-8") as f:
            external = json.load(f)
        with open(ctx.run_dir / "artifacts" / "stage14_lock_manifest.json", encoding="utf-8") as f:
            lock = json.load(f)

        conflicts_path = ctx.run_dir / "reports" / "stage04_cross_class_duplicate_conflicts.json"
        cross_class_conflicts = None
        if conflicts_path.exists():
            with open(conflicts_path, encoding="utf-8") as f:
                cross_class_conflicts = json.load(f)

        # Plot 1: internal macro-F1 across the 5 models (primary modality)
        clf_names = [n for n in internal if n != "autoencoder"]
        macro_f1s = [internal[n]["macro_f1"] for n in clf_names]
        _bar_plot(clf_names, macro_f1s, "Internal Macro-F1 by Model (dynamic_470, TEST split)",
                  "Macro-F1", plots_dir / "internal_macro_f1_by_model.png")

        # Plot 2: modality comparison (LightGBM, since it ran on every modality)
        modalities = list(ablation.keys())
        lgb_f1 = [ablation[m]["lightgbm"]["macro_f1"] for m in modalities]
        _bar_plot(modalities, lgb_f1, "LightGBM Macro-F1 Across Modalities (TEST split)",
                  "Macro-F1", plots_dir / "modality_comparison_lightgbm.png")

        # Plot 3: training time comparison (pull from stage11 — stage12 eval doesn't record
        # train_time itself; only models that appear in both reports are included)
        with open(ctx.run_dir / "metrics" / "stage11_training_results.json", encoding="utf-8") as f:
            training = json.load(f)
        timed_names = [n for n in clf_names if n in training]
        train_times = [training[n]["train_time_sec"] for n in timed_names]
        _bar_plot(timed_names, train_times, "Training Time by Model (dynamic_470)",
                  "Seconds", plots_dir / "training_time_by_model.png")

        # Confusion matrices for each classifier (primary modality)
        for name in clf_names:
            cm = np.array(internal[name]["confusion_matrix"])
            labels = internal[name]["confusion_matrix_labels"]
            _confusion_matrix_plot(cm, labels, f"Confusion Matrix — {name} (dynamic_470, TEST)",
                                    plots_dir / f"confusion_matrix_{name}.png")

        # Final markdown report
        match_note = "MATCHES Stage 13" if lock.get("metrics_match_stage13") else "**DISCREPANCY vs Stage 13 — investigate**"
        vtm = lock.get("verified_test_metrics")
        md_lines = [
            "# Final Reproducibility & Results Report",
            "",
            f"**Locked configuration:** model=`{lock['selected_model']}`, modality=`{lock['selected_modality']}`, "
            f"features={lock['n_features']}, class_weight_applied={lock.get('class_weight_applied')}, "
            f"internal macro-F1 at selection={lock['internal_macro_f1_at_selection_time']:.4f}",
        ]
        if vtm:
            md_lines.append(
                f"**Independently re-verified on TEST after retrain:** macro-F1={vtm['macro_f1']:.4f}, "
                f"accuracy={vtm['accuracy']:.4f} — {match_note}"
            )
        md_lines += [
            "",
            "## Internal Evaluation (MalDroid TEST split, dynamic_470)",
            "",
            "| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |",
            "|---|---|---|---|---|",
        ]
        for n in clf_names:
            r = internal[n]
            md_lines.append(f"| {n} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | {r['balanced_accuracy']:.4f} |")

        md_lines += ["", "## Modality Ablation (LightGBM, TEST split)", "",
                     "| Modality | Macro-F1 | Accuracy | Feature Count | Class Weight Applied |",
                     "|---|---|---|---|---|"]
        for m in modalities:
            r = ablation[m]["lightgbm"]
            md_lines.append(f"| {m} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['feature_count']} | {r.get('class_weight_applied')} |")

        md_lines += ["", "## Known Data-Quality Disclosure", ""]
        if cross_class_conflicts and cross_class_conflicts["count"] > 0:
            md_lines.append(
                f"**{cross_class_conflicts['count']} group(s)** of samples share an identical 470-feature "
                f"dynamic vector but carry *different* ground-truth labels (e.g. same behavioral fingerprint "
                f"labeled both Adware and Riskware). Each group is kept entirely within one split (no "
                f"train/test leakage), but this represents a genuine label-ambiguity ceiling in this feature "
                f"representation — disclose alongside any accuracy/macro-F1 figure derived from it."
            )
            for c in cross_class_conflicts["conflicts"]:
                md_lines.append(f"- dup_group={c['dup_group']}: classes={c['classes']}, resolved to split=`{c['resolved_split']}`")
        else:
            md_lines.append("No cross-class duplicate-vector conflicts found in this run.")

        md_lines += ["", "## External (AndMal2020) Validation", ""]
        if external.get("external_validation_possible"):
            em = external["metrics"]
            md_lines.append(f"Macro-F1: {em['macro_f1']:.4f}, Accuracy: {em['accuracy']:.4f}, on {external['n_samples']} AndMal samples.")
        else:
            md_lines.append(f"**Not possible for the locked configuration.** Reason: {external.get('reason', 'see external validation report for details.')}")

        md_lines += ["", "## Plots", ""]
        for p in sorted(plots_dir.glob("*.png")):
            md_lines.append(f"- `{p.relative_to(ctx.run_dir)}`")

        report_path = reports_dir / "FINAL_REPORT.md"
        report_path.write_text("\n".join(md_lines), encoding="utf-8")
        log.info(f"Final report written: {report_path}")

        return {"final_report": str(report_path), "plots_dir": str(plots_dir)}
