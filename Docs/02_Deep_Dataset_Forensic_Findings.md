# 02 — Deep Dataset Forensic Findings

Everything here was directly measured by running scripts against our own downloaded files. `CONFIRMED` unless marked otherwise.

## Part A — MalDroid2020

### A.1 Size, Classes, Imbalance

| Class | Category | Count | % |
|---|---|---|---|
| 1 | Adware | 1,253 | 10.8% |
| 2 | Banking | 2,100 | 18.1% |
| 3 | SMS Malware | 3,904 | 33.7% |
| 4 | Riskware | 2,546 | 22.0% |
| 5 | Benign | 1,795 | 15.5% |

**Total: 11,598 apps. Imbalance ratio: 3.1× (largest ÷ smallest).** These exact numbers match the official dataset paper — strong confirmation our files are genuine and unmodified.

### A.2 Feature Types

| Feature family | Count | Type | Notes |
|---|---|---|---|
| Static | 50,621 | Mixed: mostly binary presence flags, some genuine text (e.g. package names) | Manifest-derived (permissions/intents/APIs) |
| Dynamic — syscalls | 139 | Numeric frequency counts | System calls only |
| Dynamic — syscalls+binders | 470 | Numeric frequency counts | System calls + Android binder/behavior categories (richer) |

No missing values, no zero-variance columns in either dynamic file. Static file has a mix of numeric and text columns (important for later steps — see Doc 04).

### A.3 The Problem We Found in the Static File

The static file was **corrupted**, in two specific, verified ways:
1. **Wrong text encoding** (UTF-7 instead of UTF-8) — scrambled column names into unreadable tokens.
2. **Header shorter than the data** — header had 50,615 columns, but every single one of the 11,598 data rows had 50,621 columns. A fixed 6-column gap, present in 100% of rows.

The file also has **no label column at all** — the label lives only in the separate dynamic files.

### A.4 What We Repaired, and How We Validated It

**Repair:** fixed the text encoding, named the 6 orphan columns, and attached the correct label to each row by matching row position with the dynamic file (same 11,598 apps, same order).

**This label-matching is an assumption, not a proven fact** — the static file has no ID/hash column to check against directly. To test it, we trained a simple classifier on the repaired file using only real feature columns and the attached labels:

| Metric | Result |
|---|---|
| Random-guess baseline | 33.7% |
| Classifier accuracy after repair | **92.8%** |
| Classifier macro-F1 | **92.1%** |

If the label-matching were wrong, the model could not have scored anywhere near this — it would sit near 33.7%. This is strong evidence the repair is correct, though not formal proof. `PROPOSED-verified via experiment`

### A.5 Duplicates

72–77 exact duplicate rows found in each dynamic file (~0.6–0.7% of the data) — flagged for removal at the data-split stage, not yet removed from the source files.

## Part B — AndMal2020

### B.1 Size, Classes, Imbalance

| | Rows |
|---|---|
| Benign (5 files) | 162,181 |
| Malicious (14 categories) | 195,624 |
| **Total** | **357,805** |

Malicious category range: **Riskware 97,349** down to **FileInfector 669** — an **imbalance ratio of 145×**, far more extreme than MalDroid2020.

### B.2 Feature Types

| Feature family | Count | Notes |
|---|---|---|
| Static | 9,503 | Manifest-derived; **no header row, no label column** — label comes only from which file the row was in |
| Dynamic (before/after reboot) | 141 | Memory / API / battery / network / logcat / process activity; has real `Hash`, `Category`, `Family` columns |

Static data is extremely sparse — only ~0.25% of values are non-zero.

### B.3 Important Structural Fact: Static and Dynamic Are NOT the Same Apps

Dynamic data only exists for the 14 malicious categories (**no benign dynamic data at all**), and only ~53,000 rows total — a much smaller, different set of apps than the 357,805-row static set. **Static+dynamic fusion, which works for MalDroid2020, is not possible for AndMal2020.**

### B.4 Duplicate Investigation

Every file's hashes were checked against every other file:

| Finding | Count |
|---|---|
| Hashes appearing in more than one file | 1,503 |
| — of which are Benign↔Benign | 1,503 (100%) |
| — of which are Benign↔Malicious (label conflict) | **0** |
| Of the 1,502 `Ben0`/`Ben1` overlaps, exact byte-identical | 1 |

**Conclusion:** no app is ever labeled both benign and malicious — no integrity problem. But 1,501 of 1,502 overlapping pairs are *not* exact duplicates (they differ in a handful of feature values despite sharing a hash) — meaning these are the same real app observed twice with minor extraction differences, not simple copy-paste duplicates. This affects **0.42% of the whole dataset** and must be handled by keeping both copies of a hash together in the same split (never splitting them across train/test), not by deleting rows.

## Part C — KronoDroid

Source: Guerra-Manzanares, Bahsi, Nõmm, "KronoDroid: Time-based Hybrid-featured Dataset for Effective Android Malware Detection and Characterization," *Computers & Security*, Vol 110, 102399, 2021 (`github.com/aleguma/kronodroid`). Four CSVs extracted from `kronodroid-main.zip`: `real_malware_v1`, `real_legitimate_v1`, `emu_malware_v1`, `emu_legitimate_v1`.

### C.1 Row-Count Discrepancy — RESOLVED, Not a Duplicate Problem

A naive line count (`wc -l`) on `real_malware_v1.csv` / `real_legitimate_v1.csv` reports 82,764 / 73,510 data rows — roughly double the paper's published 41,382 / 36,755. This was investigated directly, not assumed:

| Check | real_malware | real_legitimate |
|---|---|---|
| `wc -l` (physical lines, minus header) | 82,764 | 73,510 |
| `pandas.read_csv` rows (quote-aware parse) | **41,382** | **36,755** |
| Duplicated `sha256` values | 0 | 0 |
| Fully-identical duplicate rows | 0 | 0 |
| Rows with an embedded `\n` inside the `Package` field | 41,382 (100%) | not re-checked, same pattern confirmed on malware file |

**Root cause:** every data row's quoted `Package` field contains an embedded newline character, so each logical CSV record spans two physical lines. A tool that counts physical lines instead of parsing CSV quoting (e.g. `wc -l`) double-counts every row. This is a text-encoding/export quirk of the source file, not duplicate or corrupted data.

**Conclusion:** the correctly-parsed row counts (41,382 malware / 36,755 legitimate, `real_device`) match the paper's published numbers **exactly**. No v2 drift, no silent dataset growth — the dataset on disk is the one in the 2021 paper. `CONFIRMED`

### C.2 Size, Classes, Imbalance

| Split | File | Rows | Columns | Label |
|---|---|---|---|---|
| real_device | malware | 41,382 | 484 | `Malware`=1 |
| real_device | legitimate | 36,755 | 484 | `Malware`=0 |
| emulator | malware | 28,745 | 483 | `Malware`=1 |
| emulator | legitimate | 35,246 | 484 | `Malware`=0 |

`Malware` is a clean per-file label (verified: every row in each malware file is 1, every row in each legitimate file is 0 — no mixed labels within a file). Imbalance ratio real_device: 1.13× (mild); emulator: 1.23× (mild) — far less skewed than MalDroid2020 (3.1×) or AndMal2020 (145×).

Column counts are **not identical across the four files** (484/484/483/484) — this must be reconciled (intersection vs. union of columns) before building a single combined schema; flagged for the column-manifest step, not yet resolved.

### C.3 Feature Types and Label Columns

Each file mixes feature columns with metadata/identifier columns: `Package`, `sha256`, `Malware`, `MalFamily`, `CFileSize`, `Detection_Ratio`, `Scanners`, and (real_device only) `EarliestModDate`/`HighestModDate`. Excluding these ~7-9 metadata columns leaves ~475-477 genuine feature columns per file, all numeric — consistent with the paper's claimed 200 static (permissions) + 289 dynamic (syscalls) split; exact static/dynamic column manifest still needs to be built by name-matching against Android permission strings vs. syscall names (next step, not yet done).

### C.4 Missing Values and MalFamily

- `MalFamily` is populated only for malware rows (legitimate rows are benign by definition, so it's NaN there — expected, not a defect) — accounts for the bulk of missing values in the legitimate files.
- `real_malware`: 240 unique `MalFamily` values, 165 rows with missing family (malware present but family unrecorded) — the "240 families" claim in secondary sources is **CONFIRMED** directly from the file.
- Top families by row count: Airpush/StopSMS (7,775), SMSreg (5,019), Malap (4,055), Boxer (3,597), Agent (2,934) — heavily long-tailed, relevant for any multi-class framing.

### C.5 Duplicate Investigation (sha256) — RESOLVED

No duplicate `sha256` within any of the four files individually (0 in all cases). The cross-file check (all four files concatenated, same method as the AndMal cross-file hash check in Doc 02 Part B.4) was run directly:

| Check | Result |
|---|---|
| Total rows, 4 files concatenated | 142,128 |
| Unique `sha256` across all 4 files | 78,800 |
| `sha256` values appearing more than once anywhere | 63,320 |
| **real_device ∩ emulator** (same app analyzed by both methods) | **63,320 apps** |
| real_malware ∩ emu_malware | 28,343 |
| real_legitimate ∩ emu_legitimate | 34,981 |
| real_malware ∩ real_legitimate (within real_device) | 4 |
| emu_malware ∩ emu_legitimate (within emulator) | 4 |
| malware↔legitimate label conflicts (same hash, opposite label, **anywhere**) | **4** |

**Finding 1 — real_device and emulator are the same population, not independent datasets.** 63,320 of 78,800 unique apps (80%) were captured by *both* dynamic-analysis methods. Any pipeline that concatenates real_device + emulator rows is training on two different dynamic-behavior views of mostly the same APKs, not on 2× the independent samples — this governs the C.6 decision below.

**Finding 2 — 4 confirmed label conflicts, same 4 hashes in both views.** `home.nicole.app.image` (sha256 `65ada6c1...`), `com.omesoft.baccalc` (`8ba8b904...`), `com.ikags.ikagsrandomring` (`a71f168d...`), `com.rocky.almanac` (`cf5b1cab...`) each appear as `Malware=1` in **both** `real_malware` and `emu_malware`, and as `Malware=0` in **both** `real_legitimate` and `emu_legitimate` — the conflict is consistent across both capture methods, ruling out a one-off measurement glitch. This is an upstream labeling defect in the source dataset (same APK filed under both the malware and legitimate CSVs). `CONFIRMED` — **these 4 apps must be dropped from training/eval entirely** (label cannot be trusted), not resolved by picking one label.

### C.6 real_device vs. emulator — RESOLVED: real_device is the primary track

**Decision: use `real_device` only as the primary KronoDroid track. Document `emulator` but exclude it from training.**

Reasoning, in the same style as Doc 03's real-vs-synthetic reasoning for AndMal:

1. **Not independent data.** C.5 showed 80% of apps appear in both views. Concatenating real_device + emulator rows would not add independent samples — it would duplicate most apps with a second, dynamically-different feature vector, and even with strict hash-grouping at split time (never splitting one `sha256` across train/val/test) the model would still train on the *same app* twice, biasing feature importance toward whatever is more learnable from repeated apps.
2. **Matches the cited benchmark paper exactly.** Waheed & Qadir (2024) used the real_device split (41,382 malware / 36,755 legitimate — independently re-confirmed by us in C.1) as their experimental dataset. Staying on real_device keeps our benchmark comparison (Stage 7 of this pipeline) apples-to-apples on the *same rows*, not a superset the cited paper never evaluated.
3. **Schema is cleaner.** real_device has a consistent 484-column schema across both its malware/legitimate files; emulator's malware file has only 483 columns (one fewer — see C.2), meaning emulator itself needs schema reconciliation before use, on top of the overlap problem.
4. **Emulator is not discarded, just deferred.** It remains on disk, documented, and available for a future "does the model generalize to emulator-captured dynamic behavior" experiment — but that is a distinct research question (sandbox-evasion-style generalization) from building the primary detection pipeline, and is out of scope here. `PROPOSED — decision, not a measured result`

## Trusted Files Going Forward

| Dataset | Use this file |
|---|---|
| MalDroid2020 static | `feature_vectors_static_repaired.parquet` (repaired) — **not** the original |
| MalDroid2020 dynamic | Both syscalls files — clean, no repair needed |
| AndMal2020 static/dynamic | All 19 + 28 files — clean at source, duplicates handled at split time |
| KronoDroid real_device | `real_malware_v1.csv` + `real_legitimate_v1.csv` — **primary track** (see C.6). Parse with a proper CSV reader (never `wc -l`/line-count tools). Drop the 4 label-conflict hashes in C.5 before any split. |
| KronoDroid emulator | `emu_malware_v1.csv` + `emu_legitimate_v1.csv` — documented, not used in the primary pipeline (80% app overlap with real_device, see C.5/C.6) |
