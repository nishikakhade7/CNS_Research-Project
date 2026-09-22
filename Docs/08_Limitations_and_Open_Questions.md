# 08 — Limitations & Open Questions

## CONFIRMED LIMITATIONS (facts, already verified)

1. MalDroid2020 has no sample hash/ID anywhere — identity-based splitting isn't fully possible for it.
2. MalDroid2020's static labels rely on row-order matching, validated empirically (92.8% vs 33.7% baseline) but not proven with an independent key.
3. AndMal2020's static and dynamic files describe different apps — no fusion possible for AndMal2020.
4. AndMal2020's dynamic data is malicious-categories-only — no benign dynamic data exists in what we downloaded.
5. Only 5 of AndMal2020's 14 categories map to MalDroid2020's taxonomy — the rest are excluded from the main comparison.
6. 1,503 AndMal2020 benign rows share a hash with another row but aren't exact duplicates (only 1 of 1,502 checked pairs matched exactly).
7. We cannot check for overlap between MalDroid2020 and AndMal2020 (same app in both datasets) because MalDroid2020 has no hash column.
8. MalDroid2020's static file mixes numeric and text columns — naive "treat everything as a number" code will crash (we hit this directly).
9. AndMal2020's static files have no header row — reading them without `header=None` will silently corrupt the first sample.

## PROPOSED ASSUMPTIONS (our choices, not proven facts)

1. MalDroid2020 is the right primary dataset because of the static+dynamic same-app advantage — this is a project-scope judgment.
2. The 470-feature dynamic view is proposed as primary, 139-feature as a comparison — not yet tested against each other in our own experiments.
3. AndMal's mapped categories (e.g., "Riskware") are assumed behaviorally similar to MalDroid's same-named category — this is a labeling assumption, not a verified behavioral match.

## OPEN QUESTIONS (must be tested, not assumed)

1. Best feature-selection method and count for each branch.
2. QDA's target dimensionality.
3. Autoencoder architecture and latent size.
4. Whether to run the optional full-dataset AndMal binary check alongside the main 5-class comparison.
5. Full breakdown of how much the 1,502 near-duplicate AndMal benign pairs actually differ (we checked one example — 3 of 9,503 columns differed — but didn't finish characterizing all 1,502).
6. Why our downloaded AndMal2020 benign total (162,181) is short of the commonly cited 200,000 figure — worth checking the official site for an additional file.
