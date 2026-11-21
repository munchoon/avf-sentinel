"""Cache preprocessing, paired-photo features, and audio spectrograms.

Run once after generating or importing a dataset:
    PYTHONPATH=src python -m avfsentinel.prepare --data data/synthetic

Outputs:
    preprocessed/*.jpg       ROI-cropped, white-balanced, CLAHE images
    audio_mels/*.npy         log-Mel dB arrays (128 x 224)
    features.csv             manifest + model features + quality flags
    feature_stats.json       train-set standardisation statistics
    quality_summary.json     guardrail audit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .audio import acoustic_features, log_mel, read_wav
from .constants import CLASSES
from .preprocessing import PAIR_FEATURE_NAMES, Preprocessor, pair_features
from .utils import ensure_dir, save_json


def _quality_columns(q, prefix: str) -> dict:
    if q is None:
        return {
            f"{prefix}_blur_var": np.nan,
            f"{prefix}_luma": np.nan,
            f"{prefix}_angle": np.nan,
            f"{prefix}_skin_frac": np.nan,
            f"{prefix}_issues": "",
        }
    return {
        f"{prefix}_blur_var": round(q.blur_var, 3),
        f"{prefix}_luma": round(q.mean_luma, 2),
        f"{prefix}_angle": round(q.angle_deg, 2),
        f"{prefix}_skin_frac": round(q.skin_frac, 4),
        f"{prefix}_issues": " | ".join(q.issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an AVF dataset and cache features.")
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}. Generate the demo set or import a real manifest.")
    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit).copy()
    df["patient_id"] = df["patient_id"].astype(str)

    img_out = ensure_dir(data_dir / "preprocessed" / "images")
    mel_out = ensure_dir(data_dir / "audio_mels")
    pre = Preprocessor(size=args.size)

    rows: list[dict] = []
    t0 = time.time()
    skipped_audio = 0
    for i, row in df.iterrows():
        pid = str(row["patient_id"])
        d_path = data_dir / str(row["image_down"])
        u_path = data_dir / str(row["image_up"])
        rgb_d = pre.read(d_path)
        rgb_u = pre.read(u_path)
        pair = pre.process_pair(rgb_d, rgb_u)
        pd_ = pair["down"]
        pu_ = pair["up"]
        wb_method = pair["wb_method"]

        Image.fromarray(pd_["image"]).save(img_out / f"{pid}_down.jpg", quality=92)
        Image.fromarray(pu_["image"]).save(img_out / f"{pid}_up.jpg", quality=92)
        pf = pair_features(pd_["stats"], pu_["stats"])
        quality_pass = int(not (pd_["quality"].hard_fail or pu_["quality"].hard_fail))

        af = np.zeros(16, dtype=np.float32)
        mel_rel = f"audio_mels/{pid}.npy"
        if not args.skip_audio:
            try:
                y = read_wav(data_dir / str(row["audio"]))
                db = log_mel(y)
                np.save(mel_out / f"{pid}.npy", db.astype(np.float32))
                af = acoustic_features(db, y)
            except Exception as exc:  # keep the row, flag it
                skipped_audio += 1
                mel_rel = ""
                print(f"  ! audio failed for {pid}: {exc}")

        out = row.to_dict()
        out.update(
            {
                "raw_image_down": str(row["image_down"]),
                "raw_image_up": str(row["image_up"]),
                "raw_audio": str(row["audio"]),
                "image_down": f"preprocessed/images/{pid}_down.jpg",
                "image_up": f"preprocessed/images/{pid}_up.jpg",
                "audio_mel": mel_rel,
                "quality_pass": quality_pass,
                "wb_down": wb_method,
                "wb_up": wb_method,
            }
        )
        out.update(_quality_columns(pd_["quality"], "down"))
        out.update(_quality_columns(pu_["quality"], "up"))
        for j, name in enumerate(PAIR_FEATURE_NAMES):
            out[f"pair_{j}"] = float(pf[j])
        for j, val in enumerate(af):
            out[f"acoustic_{j}"] = float(val)
        rows.append(out)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  prepared {i + 1}/{len(df)} ({time.time() - t0:.1f}s)", flush=True)

    feat = pd.DataFrame(rows)
    feat.to_csv(data_dir / "features.csv", index=False)

    # Standardisation statistics: training split, quality-passing rows only.
    train = feat[(feat["split"] == "train") & (feat["quality_pass"] == 1)]
    if len(train) < 5:
        train = feat[feat["quality_pass"] == 1]
    if len(train) < 2:
        train = feat
    pair_cols = [f"pair_{i}" for i in range(12)]
    ac_cols = [f"acoustic_{i}" for i in range(16)]
    pair_arr = np.clip(train[pair_cols].to_numpy(np.float32), -8, 8)
    ac_arr = np.clip(train[ac_cols].to_numpy(np.float32), -12, 12)
    stats = {
        "pair_mean": np.nanmean(pair_arr, axis=0).round(6).tolist(),
        "pair_std": np.maximum(np.nanstd(pair_arr, axis=0), 1e-3).round(6).tolist(),
        "acoustic_mean": np.nanmean(ac_arr, axis=0).round(6).tolist(),
        "acoustic_std": np.maximum(np.nanstd(ac_arr, axis=0), 1e-3).round(6).tolist(),
        "n_train_rows": int(len(train)),
    }
    save_json(stats, data_dir / "feature_stats.json")

    # Guardrail audit.
    q = feat["quality_issue"].astype(str) if "quality_issue" in feat else pd.Series(["none"] * len(feat))
    expected_fail = q.isin(["blur", "dark", "bright", "angle"])
    actual_fail = feat["quality_pass"] == 0
    quality = {
        "n": int(len(feat)),
        "pass": int((~actual_fail).sum()),
        "fail": int(actual_fail.sum()),
        "pass_rate": round(float((~actual_fail).mean()), 4),
        "expected_issue_counts": q.value_counts().to_dict(),
        "caught_by_issue": {
            issue: {
                "n": int((q == issue).sum()),
                "caught": int(((q == issue) & actual_fail).sum()),
            }
            for issue in ["blur", "dark", "bright", "angle"]
        },
        "false_rejections_on_clean": int(((q == "none") & actual_fail).sum()),
        "wb_methods": {
            "down": feat["wb_down"].value_counts().to_dict(),
            "up": feat["wb_up"].value_counts().to_dict(),
        },
        "skipped_audio": skipped_audio,
    }
    save_json(quality, data_dir / "quality_summary.json")
    print(f"Wrote {data_dir / 'features.csv'} and caches in {time.time() - t0:.1f}s")
    print(f"Guardrail: {quality['fail']}/{quality['n']} pairs rejected; false rejections on clean: {quality['false_rejections_on_clean']}")
    print(f"Train rows for model: {int(((feat['split'] == 'train') & (feat['quality_pass'] == 1)).sum())}")


if __name__ == "__main__":
    main()
