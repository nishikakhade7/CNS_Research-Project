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

## Trusted Files Going Forward

| Dataset | Use this file |
|---|---|
| MalDroid2020 static | `feature_vectors_static_repaired.parquet` (repaired) — **not** the original |
| MalDroid2020 dynamic | Both syscalls files — clean, no repair needed |
| AndMal2020 static/dynamic | All 19 + 28 files — clean at source, duplicates handled at split time |
