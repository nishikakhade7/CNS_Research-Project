# 05 — Proposed Model Architecture

## 1. Overall End-to-End Architecture

```mermaid
flowchart TD
    A[MalDroid2020<br/>Static + Dynamic] --> B[Integrity, Label & Identity Checks]
    B --> C[Leakage-safe Train/Val/Test Split]
    C --> D[Train-only Preprocessing]
    D --> E1[Static View]
    D --> E2[Dynamic View]
    E1 --> E3[Fused View]
    E2 --> E3
    D --> E4[Feature Selection per View]
    E1 --> E4
    E2 --> E4
    E3 --> E4

    E4 --> F[Model-Specific Feature Views]
    F --> G1[QDA]
    F --> G2[NuSVC]
    F --> G3[LightGBM]
    F --> G4[Balanced Random Forest]
    F --> G5[Autoencoder]

    G1 --> H[Internal Evaluation<br/>MalDroid val/test]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H

    H --> I["Frozen models applied to<br/>AndMal2020 (external, mapped subset)"]
    I --> J[Generalization Report]

    style D fill:#d4edda,stroke:#2d6a4f,color:#1a3a1a
    style I fill:#d0e8f5,stroke:#1a6fa8,color:#0a2a40
```

## 2. Why One Shared Matrix Doesn't Work for All 5 Models

Each model has different mathematical needs:

| Model                  | Needs scaling? | Handles many features well?       | Key limitation                                                                                           |
| ---------------------- | -------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| QDA                    | Yes            | No                                | Needs far more samples-per-class than features, or its internal math (covariance estimation) breaks down |
| NuSVC                  | Yes            | No                                | Kernel methods work best on compact, well-scaled features                                                |
| LightGBM               | No             | Yes                               | Tree splits don't care about feature scale                                                               |
| Balanced Random Forest | No             | Yes                               | Same as above; rebalances classes internally                                                             |
| Autoencoder            | Yes            | Learns its own compressed version | Needs scaled input to learn a meaningful compressed representation                                       |

Giving QDA the same 470-raw-feature matrix that LightGBM is happy with would break QDA's own assumptions (its smallest class, Adware, only has 1,253 samples). Giving LightGBM a heavily-reduced matrix built for QDA would throw away signal trees could have used. **So: one shared foundation, then a model-specific adapter per model.**

## 3. Model-Specific Pipelines

```mermaid
flowchart TD
    Shared[Shared, Train-only-fitted<br/>Selected Feature Set]

    Shared --> QDA_in["Strongly reduced,<br/>dense, scaled features"]
    QDA_in --> QDA[QDA]

    Shared --> SVC_in["Compact, scaled features<br/>+ log/count transform"]
    SVC_in --> SVC[NuSVC]

    Shared --> LGB_in["Raw/selected numeric features<br/>NO scaling"]
    LGB_in --> LGB[LightGBM]

    Shared --> BRF_in["Raw/selected numeric features<br/>NO scaling"]
    BRF_in --> BRF[Balanced Random Forest]

    Shared --> AE_in["Scaled features"]
    AE_in --> AE[Autoencoder]
    AE --> Latent["Learned latent representation<br/>(train-only, may feed other models)"]

    QDA --> Eval[Compare all 5 on same metrics]
    SVC --> Eval
    LGB --> Eval
    BRF --> Eval
    AE --> Eval
```

**Nothing here is final.** Exact feature counts for each view, QDA's target dimensionality, and the autoencoder's architecture are all open questions to be settled by experiment (Doc 06), not fixed values.

## 4. Quick Reference — Where Each Algorithm's Design Comes From

- LightGBM: histogram-based tree splitting, doesn't need scaling (Ke et al., NeurIPS 2017).
- Balanced Random Forest: rebalances classes via bootstrap sampling (Chen, Liaw & Breiman, UC Berkeley Tech Report 666, 2004).
- NuSVC: ν parameter controls margin/error trade-off (Schölkopf et al., Neural Computation, 2000).
- QDA: needs samples-per-class ≫ feature count for a stable model (standard statistical learning theory).
- SHAP for explainability: Lundberg & Lee, NeurIPS 2017.
