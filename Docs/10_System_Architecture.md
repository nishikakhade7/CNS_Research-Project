# 10 — Complete System Architecture

The single "whole picture" document. Every pipeline step, where each of the 5 algorithms sits, and one end-to-end diagram tying it all together. Pulls together [Doc 04 — Preprocessing](04_Proposed_Data_Preprocessing_Pipeline.md) and [Doc 05 — Model Architecture](05_Proposed_Model_Architecture.md) into one place so the full system can be read without jumping between files.

**Legend:** `CONFIRMED` = directly verified from our files · `RESEARCH-BACKED` = from official dataset paper/docs · `PROPOSED` = our design choice, not yet run · `OPEN` = must be tested, not assumed.

---

## 1. The Pipeline, Step by Step

| # | Stage | What happens to the data | Why this step exists |
|---|---|---|---|
| 1 | **Raw data** | MalDroid2020 (static 50,621 feat + dynamic 139/470 feat, 11,598 apps) and AndMal2020 (static 9,503 feat, dynamic 141 feat, 357,805 rows) sit untouched | Nothing is edited in place — every fix produces a new file, so we can always trace back to the original `CONFIRMED` |
| 2 | **Schema validation** | Check column counts/types match what's expected | Catches a broken file immediately — this is exactly how we caught MalDroid's static file being 6 columns short `CONFIRMED` |
| 3 | **Label validation** | MalDroid: confirm `Class` ∈ {1..5}. AndMal: derive label from filename (no native column) | Prevents training on a row with no real label |
| 4 | **Identity resolution** | AndMal: record hash. MalDroid: no hash exists anywhere — row index is the only identity | Needed to prevent the same sample appearing in both train and test |
| 5 | **Duplicate audit** | Flag MalDroid's 72–77 exact-duplicate rows/file; flag AndMal's 1,503 hash-overlapping benign rows | So duplicates are grouped correctly at split time, not scattered `CONFIRMED` |
| 6 | **Taxonomy mapping** | Map AndMal's 14 categories → MalDroid's 5 (only 5 map cleanly — Doc 03) | AndMal can only be compared to MalDroid on categories that actually correspond |
| 7 | **Train / Validation / Test split** | MalDroid split 3 ways, stratified by class, duplicate rows kept together | Validation tunes the model; test gives one honest final number; AndMal is separate from all three |
| 8 | **Train-only preprocessing fit** | Feature typing → filtering → selection → scaling — calculated using the TRAIN split only | If we calculate this using val/test/AndMal data, we leak information and the results become fake-optimistic `PROPOSED, critical rule` |
| 9 | **Static branch** | 50,621 raw static features → typed (numeric vs. text) → filtered → selected | Static file mixes real numbers with text (e.g. package names) — must be typed before selection or it crashes |
| 10 | **Dynamic branch** | 139-feature and 470-feature views both processed, in parallel | Tests whether the extra 331 binder/behavior features actually help, instead of assuming so |
| 11 | **Fusion branch** | Selected static + selected dynamic concatenated (MalDroid only) | AndMal can't do this — its static and dynamic files are different, non-overlapping apps `CONFIRMED` |
| 12 | **Feature selection** | Train-only ranking; candidate counts benchmarked against literature (80-feature ReliefF, 270-feature LightGBM importance) | Reduces dimensionality to something each model can actually handle |
| 13 | **Model-specific views** | Same selected features reshaped per model (scaled+reduced for QDA, scaled for NuSVC/Autoencoder, raw for LightGBM/BRF) | One matrix doesn't fit all 5 models' math (§2 below) |
| 14 | **Model training** | QDA, NuSVC, LightGBM, Balanced RF, Autoencoder each trained on their own view | See §2 |
| 15 | **Internal evaluation** | MalDroid held-out val/test — Macro-F1, per-class metrics, confusion matrix | Honest, in-distribution performance check |
| 16 | **External evaluation** | Frozen model applied once to AndMal's mapped subset | Tests generalization to an independently collected dataset — the actual point of using AndMal at all |
| 17 | **Explainability** | SHAP + LightGBM feature importance + manual error review | So results aren't just "94% accuracy, trust us" |
| 18 | **Report** | Internal and external results shown separately, never averaged | Combining them would hide whether the model actually generalizes |

---

## 2. Where Each Algorithm Sits, and Why

| Model | Sits at pipeline step | Input it receives | Why this input, not the shared raw matrix |
|---|---|---|---|
| **QDA** | 13→14, static/dynamic/fused view | Strongly reduced, dense, **scaled** features | QDA estimates a covariance matrix per class. MalDroid's smallest class (Adware) has only 1,253 samples — feeding it hundreds of raw features would make that covariance estimate unstable `RESEARCH-BACKED — standard statistical learning theory` |
| **NuSVC** | 13→14, static/dynamic/fused view | Compact, **scaled**, count-transformed features | Kernel methods are sensitive to feature scale and don't handle very high dimensionality well `RESEARCH-BACKED — Schölkopf et al., Neural Computation, 2000` |
| **LightGBM** | 13→14, static/dynamic/fused view | Raw/selected numeric features, **no scaling** | Tree-based splits are scale-invariant; LightGBM's histogram method is built to handle thousands of features efficiently `RESEARCH-BACKED — Ke et al., NeurIPS 2017`. This is Group 3's own primary model. |
| **Balanced Random Forest** | 13→14, static/dynamic/fused view | Raw/selected numeric features, **no scaling** | Rebalances classes internally via bootstrap sampling, so it doesn't need external class-weighting `RESEARCH-BACKED — Chen, Liaw & Breiman, UC Berkeley Tech Report 666, 2004` |
| **Autoencoder** | 13→14, static/dynamic/fused view | **Scaled** input | Learns its own compressed (latent) representation via reconstruction error — needs scaled input for that to be meaningful. Its latent output can optionally feed other models `PROPOSED` |

All five are evaluated on the same internal MalDroid metrics (step 15) and the same external AndMal metrics (step 16), so results are directly comparable across algorithms.

---

## 3. Complete End-to-End Architecture Diagram

```mermaid
flowchart TD
    subgraph RAW["1. Raw Data"]
      M["MalDroid2020<br/>11,598 apps · 5 classes<br/>Static 50,621 · Dynamic 139/470"]
      A["AndMal2020<br/>357,805 static rows · 14 categories<br/>Dynamic 141 feat, malicious only"]
    end

    M --> V["2-3. Schema + Label Validation"]
    A --> V
    V --> ID["4. Identity Resolution<br/>AndMal: hash available · MalDroid: none"]
    ID --> DUP["5. Duplicate Audit<br/>MalDroid: 72-77 exact dupes/file<br/>AndMal: 1,503 benign hash-overlaps"]
    DUP --> TAX["6. Taxonomy Mapping<br/>AndMal 14-cat -> MalDroid 5-cat subset"]
    TAX --> SPLIT["7. Leakage-safe Train/Val/Test Split<br/>(MalDroid only)"]

    SPLIT --> TRAIN[TRAIN split]
    SPLIT --> VAL[VALIDATION split]
    SPLIT --> TEST[TEST split]

    TRAIN --> FIT["8. Train-only Preprocessing Fit<br/>typing -> filtering -> selection -> scaling"]
    FIT --> ART[Saved, versioned artifacts]
    ART -->|apply only| VAL
    ART -->|apply only| TEST
    ART -->|apply only| EXT[AndMal external subset]

    FIT --> STA["9. Static Branch<br/>typed -> filtered -> selected"]
    FIT --> DYN139["10. Dynamic Branch: 139 feat<br/>(syscall-only, ablation)"]
    FIT --> DYN470["10. Dynamic Branch: 470 feat<br/>(syscall+binder, primary)"]
    STA --> FUS["11. Fusion Branch<br/>static + dynamic (MalDroid only)"]
    DYN470 --> FUS

    STA --> SEL["12. Feature Selection<br/>train-only, benchmarked vs. literature"]
    DYN139 --> SEL
    DYN470 --> SEL
    FUS --> SEL

    SEL --> MV["13. Model-Specific Views"]
    MV --> QDA_v["Reduced + scaled"]
    MV --> SVC_v["Compact + scaled"]
    MV --> LGB_v["Raw/selected, no scaling"]
    MV --> BRF_v["Raw/selected, no scaling"]
    MV --> AE_v["Scaled"]

    QDA_v --> QDA[14. QDA]
    SVC_v --> SVC[14. NuSVC]
    LGB_v --> LGB[14. LightGBM]
    BRF_v --> BRF[14. Balanced Random Forest]
    AE_v --> AE[14. Autoencoder]
    AE --> Latent[Latent representation]

    QDA --> IE["15. Internal Evaluation<br/>MalDroid val/test<br/>Macro-F1, per-class, confusion matrix"]
    SVC --> IE
    LGB --> IE
    BRF --> IE
    AE --> IE

    IE --> FREEZE{Model finalized?}
    FREEZE -- "No, keep tuning" --> FIT
    FREEZE -- "Yes, frozen" --> EE["16. External Evaluation<br/>AndMal mapped subset, applied ONCE"]
    EXT --> EE

    LGB --> XAI["17. Explainability<br/>SHAP + Feature Importance + Error Analysis"]

    IE --> REP["18. Final Report<br/>Internal + External shown SEPARATELY"]
    EE --> REP
    XAI --> REP

    style FIT fill:#d4edda,stroke:#2d6a4f,color:#1a3a1a
    style ART fill:#d4edda,stroke:#2d6a4f,color:#1a3a1a
    style EE fill:#d0e8f5,stroke:#1a6fa8,color:#0a2a40
    style EXT fill:#d0e8f5,stroke:#1a6fa8,color:#0a2a40
    style REP fill:#fde8d8,stroke:#c0392b,color:#3a1a00
```

---

## 4. Reading the Diagram

- **Green boxes** = the train-only-fitting rule in action — this is the one step that, if done wrong, invalidates every downstream number.
- **Blue boxes** = anything touching AndMal2020 — always downstream of a frozen model, never feeding back into training.
- **Orange box** = the final report, which deliberately keeps internal and external results separate rather than blending them into one number.
- Steps 9–12 run once per branch (static, dynamic-139, dynamic-470, fused) — four parallel passes through the same logic, not four different pipelines.
- Steps 13–14 run once per model per branch — the diagram shows one pass for readability; in practice this is a grid of (4 branches) × (5 models) = 20 training runs per experiment configuration.

## 5. What's Confirmed vs. Still Open

`CONFIRMED`: all dataset numbers, the static-file corruption/repair, the AndMal duplicate findings, the taxonomy mapping gaps, the model-specific input requirements (scaling/dimensionality).

`OPEN — requires experiment` (Doc 06 has the full list): exact feature-selection method/count per branch, QDA's target dimensionality, autoencoder architecture/latent size, whether the optional full-dataset binary AndMal check is run alongside the 5-class comparison.

---

## 6. Related Documents

| Document | What it adds beyond this doc |
|---|---|
| [01 — Dataset Selection & Problem Statement](01_Dataset_Selection_and_Problem_Statement.md) | Why MalDroid2020 + AndMal2020 were chosen; original dataset problem |
| [02 — Deep Dataset Forensic Findings](02_Deep_Dataset_Forensic_Findings.md) | Verified row counts, duplicate details, header/column issues |
| [03 — Dataset Comparison & Label Mapping](03_Dataset_Comparison_and_Label_Mapping.md) | Full 14→5 category mapping diagram; excluded categories explained |
| [04 — Proposed Data Preprocessing Pipeline](04_Proposed_Data_Preprocessing_Pipeline.md) | Deep-dive into steps 1–12: static branch, dynamic branch, fusion branch with separate diagrams |
| [05 — Proposed Model Architecture](05_Proposed_Model_Architecture.md) | Deep-dive into steps 13–14: model-specific pipeline adapters |
| [06 — Experimental & Evaluation Plan](06_Experimental_and_Evaluation_Plan.md) | 4 feature experiments × 5 models grid; metrics; literature benchmarks |
| [07 — Leakage & Reproducibility Protocol](07_Leakage_Reproducibility_Protocol.md) | Full anti-leakage rules, seed policy, artifact versioning |
| [08 — Limitations & Open Questions](08_Limitations_and_Open_Questions.md) | Confirmed gaps and assumptions that cannot be resolved with current data |
| [09 — Professor Approval Summary](09_Professor_Approval_Summary.md) | One-page summary + open questions for supervisor |
