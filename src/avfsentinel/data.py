"""Datasets for paired arm photos + spectrogram audio."""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, functional as TF

from .constants import CLASSES, IMAGENET_MEAN, IMAGENET_STD
from .utils import load_json

ACOUSTIC_FEATURE_COUNT = 16


def load_features(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m avfsentinel.prepare --data {data_dir}` first."
        )
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


class AVFPairDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        image_size: int = 224,
        include_rejected: bool = False,
        feature_stats: dict | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.df = load_features(self.data_dir)
        if "quality_pass" in self.df.columns and not include_rejected:
            self.df = self.df[self.df["quality_pass"] == 1]
        if split != "all":
            self.df = self.df[self.df["split"] == split]
        self.df = self.df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split!r} in {self.data_dir / 'features.csv'}")
        self.split = split
        self.augment = augment
        self.image_size = image_size
        self.feature_stats = feature_stats or self._load_stats()

    def _load_stats(self) -> dict:
        path = self.data_dir / "feature_stats.json"
        if path.exists():
            return load_json(path)
        return {
            "pair_mean": [0.0] * 12,
            "pair_std": [1.0] * 12,
            "acoustic_mean": [0.0] * ACOUSTIC_FEATURE_COUNT,
            "acoustic_std": [1.0] * ACOUSTIC_FEATURE_COUNT,
        }

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel: str) -> np.ndarray:
        img = cv2.imread(str(self.data_dir / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.data_dir / rel)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _augment_pair(self, down: np.ndarray, up: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared geometry: the collapse ratio must remain valid across the pair.
        down_chw = torch.from_numpy(down).permute(2, 0, 1)
        i, j, h, w = RandomResizedCrop.get_params(
            down_chw, scale=(0.72, 1.0), ratio=(0.85, 1.18)
        )
        down_p = TF.resized_crop(down_chw, i, j, h, w, (self.image_size, self.image_size))
        up_p = TF.resized_crop(torch.from_numpy(up).permute(2, 0, 1), i, j, h, w, (self.image_size, self.image_size))
        if random.random() < 0.5:
            down_p, up_p = TF.hflip(down_p), TF.hflip(up_p)
        angle = random.uniform(-7, 7)
        down_p = TF.rotate(down_p, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=120)
        up_p = TF.rotate(up_p, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=120)
        down_p = down_p.float() / 255.0
        up_p = up_p.float() / 255.0
        b = random.uniform(0.90, 1.10)
        c = random.uniform(0.90, 1.10)
        s = random.uniform(0.90, 1.10)
        hshift = random.uniform(-0.02, 0.02)
        down_p = TF.adjust_brightness(down_p, b)
        up_p = TF.adjust_brightness(up_p, b)
        down_p = TF.adjust_contrast(down_p, c)
        up_p = TF.adjust_contrast(up_p, c)
        down_p = TF.adjust_saturation(down_p, s)
        up_p = TF.adjust_saturation(up_p, s)
        down_p = TF.adjust_hue(down_p, hshift)
        up_p = TF.adjust_hue(up_p, hshift)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        down_t = (down_p - mean) / std
        up_t = (up_p - mean) / std
        return down_t, up_t

    def _eval_pair(self, down: np.ndarray, up: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        down_p = TF.resize(torch.from_numpy(down).permute(2, 0, 1), [self.image_size, self.image_size], antialias=True)
        up_p = TF.resize(torch.from_numpy(up).permute(2, 0, 1), [self.image_size, self.image_size], antialias=True)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (down_p.float() / 255.0 - mean) / std, (up_p.float() / 255.0 - mean) / std

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (2, mels, frames)
        n_mels, n_frames = spec.shape[1], spec.shape[2]
        for _ in range(2):
            f = random.randint(0, max(1, n_mels // 12))
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, f0 : f0 + f, :] = 0.0
        for _ in range(2):
            t = random.randint(0, max(1, n_frames // 10))
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, :, t0 : t0 + t] = 0.0
        return spec

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        pid = str(row["patient_id"])
        down = self._load_image(str(row["image_down"]))
        up = self._load_image(str(row["image_up"]))
        if self.augment:
            down_t, up_t = self._augment_pair(down, up)
            # Small independent photometric jitter simulates two separate captures.
            if random.random() < 0.5:
                down_t = down_t + torch.randn_like(down_t) * 0.01
            if random.random() < 0.5:
                up_t = up_t + torch.randn_like(up_t) * 0.01
        else:
            down_t, up_t = self._eval_pair(down, up)

        db = np.load(self.data_dir / row["audio_mel"]).astype(np.float32)
        # Fixed dB mapping, plus local contrast channel (see audio.normalize_spectrogram).
        db = np.clip(db, -90.0, 0.0)
        base = (db + 90.0) / 90.0 * 2.0 - 1.0
        smooth = cv2.GaussianBlur(db, (0, 0), sigmaX=2.5, sigmaY=2.5)
        contrast = np.clip((db - smooth) / 8.0, -1.0, 1.0)
        spec = torch.from_numpy(np.stack([base, contrast], axis=0)).float()
        if self.augment:
            spec = self._spec_augment(spec)

        pair = np.array([float(row[f"pair_{i}"]) for i in range(12)], dtype=np.float32)
        pair = np.clip(pair, -8.0, 8.0)
        pair = (pair - np.array(self.feature_stats["pair_mean"], np.float32)) / np.array(self.feature_stats["pair_std"], np.float32)
        ac = np.array([float(row[f"acoustic_{i}"]) for i in range(ACOUSTIC_FEATURE_COUNT)], dtype=np.float32)
        ac = np.clip(ac, -12.0, 12.0)
        ac = (ac - np.array(self.feature_stats["acoustic_mean"], np.float32)) / np.array(self.feature_stats["acoustic_std"], np.float32)

        labels = torch.tensor([float(row[c]) for c in CLASSES], dtype=torch.float32)
        meta = {
            "patient_id": pid,
            "scenario": str(row.get("scenario", "")),
            "skin_type": int(row.get("skin_type", 0)),
            "split": str(row.get("split", self.split)),
            "quality_pass": int(row.get("quality_pass", 1)),
        }
        return {
            "image_down": down_t,
            "image_up": up_t,
            "spectrogram": spec,
            "pair_features": torch.from_numpy(pair),
            "acoustic_features": torch.from_numpy(ac),
            "labels": labels,
            "meta": meta,
        }


def build_datasets(
    data_dir: str | Path,
    image_size: int = 224,
    include_rejected: bool = False,
    feature_stats: dict | None = None,
) -> tuple[AVFPairDataset, AVFPairDataset, AVFPairDataset]:
    train = AVFPairDataset(data_dir, "train", augment=True, image_size=image_size, include_rejected=include_rejected, feature_stats=feature_stats)
    val = AVFPairDataset(data_dir, "val", augment=False, image_size=image_size, include_rejected=include_rejected, feature_stats=feature_stats)
    test = AVFPairDataset(data_dir, "test", augment=False, image_size=image_size, include_rejected=include_rejected, feature_stats=feature_stats)
    return train, val, test


def class_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out = {}
    for c in CLASSES:
        out[c] = {
            "positive": int(df[c].sum()),
            "negative": int(len(df) - df[c].sum()),
        }
    return out
