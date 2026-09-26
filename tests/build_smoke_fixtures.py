"""Builds tiny synthetic fixtures that reproduce the REAL schema/quirks of both datasets,
so smoke tests exercise the actual stage code paths (typing edge cases, duplicate rows,
cross-category hash conflicts, disjoint feature vocabularies) without touching the real
multi-GB files. Run once: `python tests/build_smoke_fixtures.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "smoke"

N_PER_CLASS = 20
CLASSES = [1, 2, 3, 4, 5]  # Adware, Banking, SMS Malware, Riskware, Benign
N_TOTAL = N_PER_CLASS * len(CLASSES)

rng = np.random.RandomState(7)


def build_maldroid_dynamic(n_feat: int, tag: str, make_duplicates: bool) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        for _ in range(N_PER_CLASS):
            rows.append({f"feat_{tag}_{i}": int(rng.randint(0, 50)) for i in range(n_feat)} | {"Class": cls})
    df = pd.DataFrame(rows)
    if make_duplicates:
        # duplicate rows 0 and 1 exactly (mirrors the real ~0.6% exact-duplicate finding)
        df.iloc[1] = df.iloc[0]
        df.iloc[21] = df.iloc[20]
    return df


def build_maldroid_static() -> pd.DataFrame:
    """Mixed-type static matrix, string-dtype throughout (mirrors the real repaired
    parquet's dtype=str storage, INCLUDING the '' == blank quirk that caused the real
    misclassification bug)."""
    n = N_TOTAL
    data = {"idx": [str(i) for i in range(n)]}
    # binary flags stored as '' (absent) / '1.0' (present) — the exact real-world quirk
    for b in range(5):
        col = []
        for i in range(n):
            col.append("1.0" if rng.rand() < 0.3 else "")
        data[f"binary_flag_{b}"] = col
    # count-like columns stored as numeric strings, with some blanks too
    for c in range(5):
        col = []
        for i in range(n):
            col.append(str(int(rng.randint(0, 200))) if rng.rand() < 0.9 else "")
        data[f"count_feat_{c}"] = col
    # genuine categorical string columns (package-name-style) — must NOT be treated as numeric
    packages = ["com.example.alpha", "com.example.beta", "com.example.gamma"]
    data["package_name_like"] = [rng.choice(packages) for _ in range(n)]
    data["another_string_col"] = [rng.choice(["com.a.b", "com.c.d"]) for _ in range(n)]
    data["third_string_col"] = [rng.choice(["x.y.z", "p.q.r"]) for _ in range(n)]
    # high-cardinality string column (near-unique per row)
    data["high_cardinality_id"] = [f"uuid-{i}-{rng.randint(0,999999)}" for i in range(n)]
    # a zero-variance numeric column
    data["always_zero"] = ["" for _ in range(n)]

    df = pd.DataFrame(data)
    return df


def write_static_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df.astype(str), preserve_index=False)
    pq.write_table(table, path)


def build_andmal_static(fix_dir: Path) -> None:
    benign_dir = fix_dir / "AndMal" / "Static" / "Benign"
    mal_dir = fix_dir / "AndMal" / "Static" / "Malicious"
    benign_dir.mkdir(parents=True, exist_ok=True)
    mal_dir.mkdir(parents=True, exist_ok=True)

    def rows_for(label_seed: str, n: int):
        out = []
        for i in range(n):
            h = f"HASH_{label_seed}_{i:03d}"
            feats = [rng.randint(0, 100) for _ in range(6)]
            out.append([h] + feats)
        return out

    # Ben0.csv, Ben1.csv — includes ONE internal duplicate hash between them (mirrors the
    # real Ben0/Ben1 finding), headerless, first column = hash
    ben0 = rows_for("BEN0", 6)
    ben1 = rows_for("BEN1", 6)
    ben1[0][0] = ben0[0][0]  # duplicate hash, benign<->benign (expected, non-conflicting)
    pd.DataFrame(ben0).to_csv(benign_dir / "Ben0.csv", header=False, index=False)
    pd.DataFrame(ben1).to_csv(benign_dir / "Ben1.csv", header=False, index=False)

    # Malicious categories that map to the 5-class taxonomy
    adware = rows_for("ADW", 6)
    banker = rows_for("BANK", 5)
    sms = rows_for("SMS", 5)
    riskware = rows_for("RISK", 5)

    # deliberately inject ONE cross-category conflict: same hash in Ben0 (benign) AND
    # Adware (malicious) — this MUST be caught by the fixed cross-directory overlap check.
    adware[0][0] = ben0[1][0]

    pd.DataFrame(adware).to_csv(mal_dir / "Adware.csv", header=False, index=False)
    pd.DataFrame(banker).to_csv(mal_dir / "Banker.csv", header=False, index=False)
    pd.DataFrame(sms).to_csv(mal_dir / "SMS.csv", header=False, index=False)
    pd.DataFrame(riskware).to_csv(mal_dir / "Riskware.csv", header=False, index=False)


def build_andmal_dynamic(fix_dir: Path) -> None:
    """Deliberately uses a DIFFERENT feature-name vocabulary than the MalDroid dynamic
    fixture (Memory_/API_-style vs feat_470_-style) — this mirrors the real, confirmed
    finding that the two datasets share no common dynamic feature vocabulary, and is what
    Stage 15 must correctly report as 'external validation not possible', not silently
    fake around."""
    d = fix_dir / "AndMal" / "Dynamic" / "BeforeAndAfterReboot"
    d.mkdir(parents=True, exist_ok=True)
    categories = {"Adware": 6, "Banker": 5, "SMS": 5, "Riskware": 5}
    for cat, n in categories.items():
        for phase in ("before", "after"):
            rows = []
            for i in range(n):
                rows.append({
                    "Memory_PssTotal": int(rng.randint(0, 100000)),
                    "Memory_HeapSize": int(rng.randint(0, 50000)),
                    "API_Process_start": int(rng.randint(0, 20)),
                    "Logcat_error": int(rng.randint(0, 10)),
                    "Hash": f"ANDMAL_DYN_{cat}_{phase}_{i}",
                    "Category": cat,
                    "Family": f"{cat.lower()}fam{i%3}",
                })
            pd.DataFrame(rows).to_csv(d / f"{cat}_{phase}_reboot_Cat.csv", index=False)


def main():
    if FIX.exists():
        shutil.rmtree(FIX)
    maldroid_dir = FIX / "MalDroid"
    (maldroid_dir / "Dynamic").mkdir(parents=True, exist_ok=True)
    (maldroid_dir / "Static").mkdir(parents=True, exist_ok=True)

    dyn139 = build_maldroid_dynamic(8, "139", make_duplicates=True)
    dyn470 = build_maldroid_dynamic(10, "470", make_duplicates=True)
    dyn139.to_csv(maldroid_dir / "Dynamic" / "dynamic_139.csv", index=False)
    dyn470.to_csv(maldroid_dir / "Dynamic" / "dynamic_470.csv", index=False)

    static_df = build_maldroid_static()
    write_static_parquet(static_df, maldroid_dir / "Static" / "static_repaired.parquet")

    build_andmal_static(FIX)
    build_andmal_dynamic(FIX)

    print(f"Smoke fixtures written under: {FIX}")
    print(f"MalDroid: {N_TOTAL} rows, {len(CLASSES)} classes, dynamic_139 (8 feat), dynamic_470 (10 feat), static (mixed types)")
    print("AndMal: Benign(Ben0,Ben1 with 1 internal dup) + Malicious(Adware,Banker,SMS,Riskware) "
          "with 1 deliberate cross-category (benign<->malicious) hash conflict injected")


if __name__ == "__main__":
    main()
