# Presentation Architecture
## Android Malware Classification — Group 3

> **Slide-ready content.** Each `---` section = one slide or slide group.

---

## Slide 1 — Problem Statement

**Android Malware Detection as 5-Class Classification**

- Android malware is a growing threat; automated, scalable detection is required
- Given: static + dynamic features extracted from Android APKs
- Predict: one of 5 categories — Adware · Banking · SMS Malware · Riskware · Benign
- Primary metric: **Macro-F1** (equal weighting across all classes)
- Supporting metrics: Accuracy · Balanced Accuracy

---

## Slide 2 — Dataset Choice

```mermaid
flowchart LR
    subgraph Primary["🗂️ MalDroid2020 — Primary"]
        M1["11,598 applications\n5 classes"]
        M2["Static: 50,620 columns\n(APK analysis)"]
        M3["Dynamic-139: 139 syscall features\nDynamic-470: 470 syscall+Binder features"]
    end

    subgraph External["🔬 AndMal2020 — External Validation"]
        A1["357,805 static records\n53,439 dynamic records"]
        A2["9,503 static features\n144 dynamic features"]
        A3["Broader taxonomy\n(17+ categories → mapped to 5)"]
    end

    Primary -- "Train / Val / Test" --> Model["LightGBM Model"]
    External -- "External Evaluation\n(attempted)" --> Model

    style Primary fill:#2b4c7e,color:#fff,stroke:#1a365d
    style External fill:#276749,color:#fff,stroke:#276749
    style Model fill:#553c9a,color:#fff
```

**Why MalDroid as primary?** It provides matched static + dynamic representations for the **same** application population — enabling valid fusion. AndMal2020 is used only for external generalization analysis.

---

## Slide 3 — Why MalDroid Primary

| Property | MalDroid2020 | AndMal2020 |
|---|---|---|
| Samples | 11,598 | 357,805 |
| Static features | 50,620 | 9,503 |
| Dynamic features | 139 / 470 | 144 (malicious only) |
| Static + Dynamic matched? | ✅ Yes (same population) | ⚠️ Not pre-established |
| Per-sample hash in static? | ❌ No | ✅ Yes |
| Dynamic instrumentation | Syscall/Binder traces | Memory/API/battery metrics |
| Role in this project | Primary | External validation |

---

## Slide 4 — Why AndMal External

- Different dynamic instrumentation → feature-space incompatible with MalDroid model
- No benign dynamic records in the released dataset
- Static/dynamic populations not pre-aligned for fusion
- Broader taxonomy needs mapping → some categories unmappable

**Used for:** post-hoc generalization check after model is finalized on MalDroid.  
**NOT used for:** tuning, training, preprocessing decisions, or feature selection.

---

## Slide 5 — Pipeline Overview

```mermaid
flowchart TD
    Raw["Raw Data\nMalDroid + AndMal"] --> Audit["Stage 01\nFile Audit\n361.7 s"]
    Audit --> Manifest["Stage 02\nCanonical Manifest\n28.5 s"]
    Manifest --> Typing["Stage 03\nFeature Typing\n125.2 s\n31,652 usable / 18,968 dropped"]
    Typing --> Split["Stage 04\nLeakage-Safe Split\nSeed 42 · 80/10/10"]

    Split --> StaticPrep["Stage 05\nStatic Preprocessing\n175.3 s"]
    Split --> DynPrep["Stage 06\nDynamic Preprocessing\n1.6 s"]
    StaticPrep --> Fusion["Stage 07\nFusion\n168.4 s"]
    DynPrep --> Fusion

    Fusion --> Selection["Stage 08\nFeature Selection\n2,464.7 s\n270 features selected"]
    Selection --> Imbalance["Stage 09\nImbalance Strategy\nClass weights computed"]

    Imbalance --> Train["Stage 11\nLightGBM Training"]
    Train --> Eval["Stage 12\nInternal Evaluation"]
    Eval --> Ablation["Stage 13\nModality Ablation\n4 modalities compared"]
    Ablation --> Lock["Stage 14\nModel Lock\nFused · K=270 · Frozen"]
    Lock --> ExtVal["Stage 15\nExternal Validation\n0/270 features matched"]
    Lock --> Report["Stage 16\nFinal Report"]

    style Split fill:#2b6cb0,color:#fff
    style Selection fill:#744210,color:#fff
    style Lock fill:#276749,color:#fff
    style ExtVal fill:#742a2a,color:#fff
```

---

## Slide 6 — Leakage Prevention

```mermaid
flowchart TD
    S["Stage 04: Create Split Manifest\nTrain · Val · Test labels assigned"]
    S --> F1["Stages 05/06: Fit scalers on TRAIN ONLY\nPersist .joblib artifacts"]
    F1 --> F2["Stage 08: Fit feature selectors on TRAIN ONLY\nPersist selection JSON"]
    F2 --> T["Stages 11/13/14: Train model on TRAIN ONLY"]
    T --> E["Stages 12/13/14: Evaluate on TEST ONLY\nVal used for config selection only"]
    E --> EXT["Stage 15: Transform external data\nusing locked feature contract\nNo refitting on external data"]

    style S fill:#2b6cb0,color:#fff,stroke:#2c5282
    style T fill:#276749,color:#fff
    style E fill:#276749,color:#fff
    style EXT fill:#553c9a,color:#fff
```

**Key guarantee:** No test-set information informed any fitting step (preprocessing, selection, or training).

---

## Slide 7 — Feature Engineering

### Static Features (50,620 → 31,652 usable → 152 selected)

![Figure 5 — Feature Type Distribution](assets/figures/fig05_feature_type_distribution.png)

### Feature Selection: 3 Methods × 3 K-Values × 4 Modalities

| Method | Description |
|---|---|
| Mutual Information | Information-theoretic dependence with label |
| Chi-Squared | Statistical feature-label dependence |
| LightGBM Importance | Tree split-gain scores (train-only) |

---

## Slide 8 — Feature Selection Result

![Figure 6 — Feature Count Reduction](assets/figures/fig06_feature_selection_reduction.png)

![Figure 10 — Static vs Dynamic Contribution in 270 Features](assets/figures/fig10_static_dynamic_feature_split.png)

**Final contract:** 270 features = 152 static + 118 dynamic-470

---

## Slide 9 — Fusion

```mermaid
flowchart LR
    S["Static Branch\n31,652 usable cols\nStandardScaler (train-only)"]
    D["Dynamic-470 Branch\n470 features\nStandardScaler (train-only)"]

    S --> FS["Feature Selection\n→ 152 static features"]
    D --> FD["Feature Selection\n→ 118 dynamic features"]

    FS --> FUS["Row-wise Concatenation\n(same 11,598 apps · same row order)"]
    FD --> FUS

    FUS --> FC["Fused Feature Contract\n270 columns\nInput to LightGBM"]

    style FUS fill:#553c9a,color:#fff
    style FC fill:#276749,color:#fff
```

**Validity:** Fusion is valid because MalDroid static and dynamic files share the same 11,598 applications in the same row order.

---

## Slide 10 — LightGBM Model

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosted Trees (histogram-based) |
| `n_estimators` | 300 |
| `num_leaves` | 31 |
| `learning_rate` | 0.05 |
| `device_type` | CPU |
| `class_weight` | Inverse-frequency (computed from training set) |
| Seed | 42 |
| Training time | 11.49 seconds |

**Why LightGBM?** Handles sparse binary features natively · Fast on high-dimensional data · Built-in class weighting · No normalization required · Multi-class softmax support

---

## Slide 11 — Results

```mermaid
%%{init: {"themeVariables": {"primaryColor": "#276749"}}}%%
quadrantChart
    title Modality Performance (TEST split)
    x-axis "Lower Accuracy" --> "Higher Accuracy"
    y-axis "Lower Macro-F1" --> "Higher Macro-F1"
    quadrant-1 Best Zone
    Dynamic-139: [0.15, 0.10]
    Dynamic-470: [0.43, 0.47]
    Static-only: [0.77, 0.75]
    Fused: [0.98, 0.97]
```

| Modality | Accuracy | Macro-F1 |
|---|:---:|:---:|
| Dynamic-139 | 0.9440 | 0.9308 |
| Dynamic-470 | 0.9543 | 0.9457 |
| Static-only | 0.9776 | 0.9750 |
| **Fused ⭐** | **0.9819** | **0.9795** |

![Figure 3 — Modality Comparison](assets/figures/fig03_modality_comparison.png)

---

## Slide 12 — Final Test Results

![Figure 1 — Confusion Matrix](assets/figures/fig01_confusion_matrix_fused_test.png)

![Figure 2 — Per-Class Precision / Recall / F1](assets/figures/fig02_per_class_metrics_fused_test.png)

**1,140 / 1,161 correctly classified on held-out test set**

---

## Slide 13 — External Validation Limitation

```mermaid
flowchart LR
    Model["Locked Fused Model\n270 features required\n118 × dyn470__ names"]
    AndMal["AndMal2020 Dynamic\n144 columns\nmemory/API/battery metrics"]

    Model -- "Feature name check\nStage 15 (0.1 s)" --> Check{"0 / 270\nnames matched"}
    Check -- "❌ Incompatible" --> Result["Validation NOT possible\nZero-fill rejected\n(semantically invalid)"]

    style Check fill:#742a2a,color:#fff
    style Result fill:#742a2a,color:#fff
    style Model fill:#276749,color:#fff
```

**Root cause:** MalDroid dynamic = syscall/Binder traces · AndMal dynamic = memory/API/battery monitoring — completely different instrumentation frameworks.

**Correct interpretation:** Feature-space incompatibility for THIS locked fused model. Not a model failure. Not a dataset failure.

---

## Slide 14 — Key Findings

| Finding | Detail |
|---|---|
| ✅ Fused modality wins | Macro-F1 = 0.9795 vs. Static-only 0.9750 |
| ✅ Static > Dynamic alone | Static-only outperforms both dynamic-only variants |
| ✅ Fast training | 11.49 s retrain on CPU (11,598 samples, 270 features) |
| ✅ Strong per-class F1 | Worst class: Adware F1 = 0.9688 |
| ⚠️ Cross-class duplicates | 3 groups — genuine label ambiguity in dynamic space |
| ❌ External validation | Feature-space incompatible; not performed |

---

## Slide 15 — Limitations

1. **Static alignment:** Empirical row-order matching (no per-sample hash in MalDroid static file)
2. **Cross-class dynamic duplicates:** 3 groups with identical feature vectors but different labels — performance ceiling
3. **LightGBM CPU only:** GPU Tree Learner unavailable in the installed wheel
4. **External validation:** Not possible for locked fused model (feature-space incompatibility)
5. **Temporal scope:** Results bounded to MalDroid2020 collection period

---

## Rendered Architecture Diagrams

All Mermaid diagrams above render natively in:
- GitHub Markdown previews
- VS Code with Mermaid extension
- Any Mermaid-compatible documentation system

Source `.mmd` files are in [`diagrams/`](diagrams/).

---

*Content sourced entirely from `runs/real` execution artifacts. All metric values are exact.*
