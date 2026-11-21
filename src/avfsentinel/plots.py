"""Artifact plots for training, evaluation, and the hackathon deck."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import CLASSES, CLASS_SHORT, SCENARIOS
from .metrics import roc_points

COLORS = ["#d1495b", "#00798c", "#edae49", "#30638e", "#6a4c93"]


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(history: list[dict], path: str | Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("focal loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [h["val_macro_auroc"] for h in history], marker="o", color="#00798c")
    axes[1].set_ylim(0.4, 1.01)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro AUROC")
    axes[1].set_title("Validation discrimination (patient-grouped)")
    axes[1].grid(alpha=0.25)
    _save(fig, path)


def plot_roc_pr(y: np.ndarray, p: np.ndarray, path: str | Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    for i, c in enumerate(CLASSES):
        yc, pc = y[:, i], p[:, i]
        fpr, tpr = roc_points(yc, pc)
        axes[i].plot(fpr, tpr, color=COLORS[i], lw=2, label=f"AUROC={_auc(yc, pc):.3f}")
        axes[i].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[i].set_title(f"ROC — {CLASS_SHORT[c]}")
        axes[i].set_xlabel("1 − specificity")
        axes[i].set_ylabel("sensitivity")
        axes[i].legend(loc="lower right", fontsize=8)
        axes[i].grid(alpha=0.2)

    # PR curves on the last panel (all classes), with prevalence baselines.
    ax = axes[5]
    for i, c in enumerate(CLASSES):
        yc, pc = y[:, i], p[:, i]
        order = np.argsort(-pc)
        tp = np.cumsum(yc[order])
        fp = np.cumsum(1 - yc[order])
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(yc.sum(), 1)
        ax.plot(recall, precision, color=COLORS[i], lw=2, label=f"{CLASS_SHORT[c]} (AP={_ap(yc, pc):.3f})")
    ax.set_title("Precision–recall (all findings)")
    ax.set_xlabel("recall / sensitivity")
    ax.set_ylabel("precision / PPV")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    _save(fig, path)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def plot_reliability(y: np.ndarray, p: np.ndarray, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    for i, c in enumerate(CLASSES):
        ax = axes[i]
        yc, pc = y[:, i], p[:, i]
        bins = np.linspace(0, 1, 9)
        centres, frac = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (pc >= lo) & (pc < hi) if hi < 1 else (pc >= lo) & (pc <= hi)
            if sel.sum() > 0:
                centres.append((lo + hi) / 2)
                frac.append(yc[sel].mean())
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(centres, frac, marker="o", color=COLORS[i], lw=2)
        ax.set_title(CLASS_SHORT[c], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("predicted probability")
        if i == 0:
            ax.set_ylabel("observed frequency")
        ax.grid(alpha=0.2)
    _save(fig, path)


def plot_confusion(y: np.ndarray, p: np.ndarray, thresholds: dict[str, float], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for i, c in enumerate(CLASSES):
        yc, pc = y[:, i], p[:, i]
        pred = pc >= thresholds[c]
        tp = int(np.sum(pred & (yc == 1)))
        fp = int(np.sum(pred & (yc == 0)))
        tn = int(np.sum(~pred & (yc == 0)))
        fn = int(np.sum(~pred & (yc == 1)))
        mat = np.array([[tn, fp], [fn, tp]])
        ax = axes[i]
        ax.imshow(mat, cmap="Blues", vmin=0)
        for r in range(2):
            for col in range(2):
                ax.text(col, r, str(mat[r, col]), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["no", "yes"])
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(CLASS_SHORT[c], fontsize=10)
    _save(fig, path)


def plot_preprocessing_grid(df: pd.DataFrame, data_dir: str | Path, path: str | Path, n: int = 3) -> None:
    data_dir = Path(data_dir)
    df = df.head(n)
    if len(df) == 0:
        return
    fig, axes = plt.subplots(len(df), 4, figsize=(12, 3.1 * len(df)))
    if len(df) == 1:
        axes = axes[None, :]
    for r, (_, row) in enumerate(df.iterrows()):
        raw_d = cv2.cvtColor(cv2.imread(str(data_dir / row["raw_image_down"])), cv2.COLOR_BGR2RGB)
        raw_u = cv2.cvtColor(cv2.imread(str(data_dir / row["raw_image_up"])), cv2.COLOR_BGR2RGB)
        pre_d = cv2.cvtColor(cv2.imread(str(data_dir / row["image_down"])), cv2.COLOR_BGR2RGB)
        pre_u = cv2.cvtColor(cv2.imread(str(data_dir / row["image_up"])), cv2.COLOR_BGR2RGB)
        for c, (img, title) in enumerate(
            [(raw_d, "raw — arm down"), (pre_d, "ROI + WB + CLAHE"), (raw_u, "raw — arm raised"), (pre_u, "ROI + WB + CLAHE")]
        ):
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(f"{row['patient_id']}\n{row.get('scenario', '')}", fontsize=8)
    fig.suptitle("Capture guardrail + normalisation (synthetic demo cases)", y=1.01)
    _save(fig, path)


def plot_spectrogram_grid(df: pd.DataFrame, data_dir: str | Path, path: str | Path) -> None:
    data_dir = Path(data_dir)
    chosen = []
    for s in SCENARIOS:
        sub = df[df["scenario"] == s]
        if len(sub):
            chosen.append(sub.iloc[0])
    if not chosen:
        return
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.0 * len(chosen), 3.6))
    if len(chosen) == 1:
        axes = [axes]
    for ax, row in zip(axes, chosen):
        db = np.load(data_dir / row["audio_mel"])
        ax.imshow(db, aspect="auto", origin="lower", cmap="magma", vmin=np.percentile(db, 5), vmax=np.percentile(db, 99.5))
        ax.set_title(f"{row['scenario']} / {row['bruit_type']}", fontsize=9)
        ax.set_xlabel("time frames")
        ax.set_ylabel("Mel bands")
    fig.suptitle("Phone-mic bruit → log-Mel 'photo' (synthetic)", y=1.02)
    _save(fig, path)


def plot_triage_distribution(triage_counts: dict[str, int], path: str | Path) -> None:
    labels = ["red", "amber", "green"]
    values = [triage_counts.get(k, 0) for k in labels]
    colors = ["#d1495b", "#edae49", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", fontsize=10)
    ax.set_ylabel("patients")
    ax.set_title("Held-out test triage (synthetic cohort)")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)
