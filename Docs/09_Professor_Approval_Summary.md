# 09 — Professor Approval Summary

*Short version — the other 8 documents contain the full detail.*

## 1. Problem
Classify Android apps by malware category using static (declared) and dynamic (behavioral) features, and check whether the model still works on a separately collected dataset.

## 2. Dataset Issue Found
The originally assigned CIC-AndMal2017 data, as available to us, only had network-flow features — no manifest-based static data and no syscall-based dynamic data for the same apps, which our pipeline needs.

## 3. Why We Changed Datasets
Switched to **MalDroid2020** (matched static+dynamic data, same apps) and **AndMal2020** (large, independently collected, for generalization testing).

## 4. Why MalDroid2020
Same 11,598 apps have both static and dynamic features — the only dataset we found that supports true static+dynamic fusion. Manageable size and imbalance (3.1×) for comparing 5 different algorithm types. Verified against the official paper — our numbers match exactly.

## 5. Why AndMal2020
Independently collected (different apps, different labeling method, much larger, different category system) — the right kind of dataset to test whether our model generalizes, rather than just memorizing MalDroid2020's quirks.

## 6. Proposed Architecture
Shared preprocessing (checks → leakage-safe split → train-only feature selection/scaling) → model-specific views → QDA / NuSVC / LightGBM / Balanced RF / Autoencoder → internal evaluation → external AndMal2020 check. Full diagrams in Documents 04–06.

## 7. What Will Be Trained
All 5 models, across 4 feature experiments (static-only, dynamic syscall-only, dynamic syscall+binder, static+dynamic fusion) — MalDroid2020 only.

## 8. What Will Be Evaluated
Internal performance (MalDroid held-out data — Macro-F1, per-class metrics, confusion matrices) and, separately, external generalization (AndMal2020, mapped 5-category subset) — reported as two distinct results.

## 9. How We Prevent Leakage
All preprocessing fit on training data only; AndMal2020 used exactly once, after everything is finalized, never for tuning (full protocol in Document 07).

## 10. What We Need Approval For
- The MalDroid2020 = train / AndMal2020 = external-validation-only strategy.
- Excluding 9 of AndMal2020's 14 categories from the main comparison (no MalDroid equivalent exists).
- Whether to also run the optional full-dataset binary AndMal check.

## 11. Questions for Ma'am
1. Is excluding those 9 unmapped AndMal categories an acceptable scope, or should we attempt a broader relabeling?
2. Should the full-dataset binary AndMal check be required, or a bonus experiment?
3. Is it acceptable that MalDroid2020's static labels rest on an empirically validated (92.8% vs 33.7%) but not independently-proven assumption, given the dataset provides no other option?
4. Should we try to obtain raw-APK hashes for MalDroid2020 (if available elsewhere) purely to enable a MalDroid↔AndMal overlap check, or accept this as a documented limitation?
