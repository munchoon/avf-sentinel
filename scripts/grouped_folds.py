#!/usr/bin/env python3
"""Assign patient-grouped, stratified folds to a real AVF manifest.

The demo generator already writes patient-grouped train/val/test splits. This
helper is for a real manifest that has no splits yet, or for cross-validation.

Key rule: every row for a patient goes to the same fold. We stratify on the
finding pattern (e.g. ``stenosis+aneurysm``) so rare combinations are spread
across folds, collapsing very rare patterns to the scenario label.

Usage:
    python scripts/grouped_folds.py --manifest data/real/manifest.csv \
        --out data/real/manifest_folds.csv --folds 5 --write-splits

    # Then point training at a directory containing the fold manifest as
    # manifest.csv, or use the fold column for your own loop.

``--write-splits`` maps fold 0 -> test, fold 1 -> val, folds 2.. -> train.
For a small cohort, use ``--folds 5``; for a quick run, ``--folds 3``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from avfsentinel.constants import CLASSES
except Exception:  # allow running without PYTHONPATH=src
    CLASSES = ("stenosis", "thrombosis", "aneurysm", "infection", "hematoma")


def pattern(row: pd.Series) -> str:
    active = [c for c in CLASSES if int(row.get(c, 0)) == 1]
    return "+".join(active) if active else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-grouped stratified folds for an AVF manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write-splits", action="store_true", help="fold 0=test, fold 1=val, rest=train")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "patient_id" not in df.columns:
        raise SystemExit("manifest needs a patient_id column")
    df["patient_id"] = df["patient_id"].astype(str)

    # One row per patient for stratification.
    pat = df.drop_duplicates("patient_id").copy()
    pat["_pattern"] = pat.apply(pattern, axis=1)
    counts = pat["_pattern"].value_counts()
    rare = set(counts[counts < args.folds].index)
    pat["_stratum"] = pat["_pattern"].where(~pat["_pattern"].isin(rare), "rare")
    if pat["_stratum"].value_counts().min() < args.folds:
        # Still too granular; fall back to scenario or a single stratum.
        if "scenario" in pat.columns:
            pat["_stratum"] = pat["scenario"].astype(str)
        else:
            pat["_stratum"] = "all"

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_of: dict[str, int] = {}
    for fold, (_, val_idx) in enumerate(skf.split(pat, pat["_stratum"])):
        for pid in pat.iloc[val_idx]["patient_id"]:
            fold_of[pid] = fold

    df["fold"] = df["patient_id"].map(fold_of).astype(int)
    if args.write_splits:
        split = df["fold"].map(lambda f: "test" if f == 0 else ("val" if f == 1 else "train"))
        df["split"] = split

    out = Path(args.out) if args.out else Path(args.manifest).with_name("manifest_folds.csv")
    df.to_csv(out, index=False)

    print(f"Wrote {out} with {args.folds} patient-grouped folds")
    print("Patients per fold:", df.groupby("fold")["patient_id"].nunique().to_dict())
    if args.write_splits:
        print("Rows per split:", df["split"].value_counts().to_dict())
        # Leakage check.
        leaking = df.groupby("patient_id")["split"].nunique().max()
        print("Max splits per patient (must be 1):", int(leaking))
    print("\nClass prevalence by fold:")
    for c in CLASSES:
        if c in df.columns:
            print(f"  {c:<10}", df.groupby("fold")[c].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
