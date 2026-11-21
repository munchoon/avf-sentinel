"""Train the two-stream AVF Sentinel model.

Example:
    PYTHONPATH=src python -m avfsentinel.train \
        --data data/synthetic --out artifacts/avf-demo \
        --epochs 14 --batch-size 12 --device auto
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .constants import CLASS_LABELS, CLASSES, DISCLAIMER
from .data import build_datasets, class_counts
from .metrics import (
    bootstrap_ci,
    choose_sensitivity_threshold,
    compute_metrics,
    fit_temperature,
    format_metrics,
    triage,
)
from .models import FocalLoss, build_model
from .plots import (
    plot_confusion,
    plot_preprocessing_grid,
    plot_reliability,
    plot_roc_pr,
    plot_spectrogram_grid,
    plot_training,
    plot_triage_distribution,
)
from .utils import count_parameters, ensure_dir, fmt_seconds, get_device, load_json, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AVF Sentinel multimodal model.")
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="artifacts/avf-demo")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--arm-backbone", default="convnext_tiny")
    parser.add_argument("--audio-backbone", default="resnet18")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-sens", type=float, default=0.90)
    parser.add_argument(
        "--sens-margin",
        type=float,
        default=0.05,
        help="Extra validation sensitivity requested as safety margin for threshold selection.",
    )
    parser.add_argument(
        "--min-val-spec",
        type=float,
        default=0.80,
        help="Lowest validation specificity allowed while maximising sensitivity. Falls back to best specificity if unreachable.",
    )
    parser.add_argument(
        "--from-checkpoint",
        default="",
        help="Skip training and recalibrate/re-evaluate an existing checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def make_loader(ds, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=workers > 0,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "image_down": batch["image_down"].to(device),
        "image_up": batch["image_up"].to(device),
        "spectrogram": batch["spectrogram"].to(device),
        "pair_features": batch["pair_features"].to(device),
        "acoustic_features": batch["acoustic_features"].to(device),
        "labels": batch["labels"].to(device),
        "meta": batch["meta"],
    }


def forward_batch(model, batch: dict) -> torch.Tensor:
    return model(
        batch["image_down"],
        batch["image_up"],
        batch["spectrogram"],
        batch["pair_features"],
        batch["acoustic_features"],
        has_audio=True,
    )


def evaluate_loss(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray, dict]:
    model.eval()
    logits_all, labels_all = [], []
    pids, skin, quality = [], [], []
    total, n = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = forward_batch(model, batch)
            loss = criterion(logits, batch["labels"])
            total += float(loss.detach()) * len(batch["labels"])
            n += len(batch["labels"])
            logits_all.append(logits.float().cpu())
            labels_all.append(batch["labels"].float().cpu())
            pids.extend(list(batch["meta"]["patient_id"]))
            st = batch["meta"].get("skin_type")
            quality.extend(
                batch["meta"].get("quality_pass", torch.ones(len(batch["labels"]))).tolist()
            )
            if isinstance(st, torch.Tensor):
                skin.extend(st.tolist())
            else:
                skin.extend(list(st))
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return total / max(n, 1), logits, labels, {"patient_id": pids, "skin_type": skin, "quality_pass": quality}


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for i in range(y.shape[1]):
        if len(np.unique(y[:, i])) > 1:
            vals.append(roc_auc_score(y[:, i], p[:, i]))
    return float(np.mean(vals)) if vals else float("nan")


def train_epoch(model, loader, optimizer, criterion, device, frozen: bool) -> float:
    model.train()
    if frozen:
        model.arm.encoder.eval()
        model.audio.encoder.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_batch(model, batch)
        loss = criterion(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(batch["labels"])
        n += len(batch["labels"])
    return total / max(n, 1)


def choose_operating_points(
    y_val: np.ndarray,
    p_val: np.ndarray,
    target_sens: float,
    sens_margin: float = 0.05,
    min_val_spec: float = 0.80,
) -> tuple[dict, dict, dict]:
    """Pick red/amber thresholds with a safety margin for the small val split."""
    red, amber, meta = {}, {}, {}
    for i, c in enumerate(CLASSES):
        nominal = 0.95 if c == "thrombosis" else target_sens
        target = min(1.0, nominal + sens_margin)
        amber_target = min(0.99, target + 0.04)
        r = choose_sensitivity_threshold(y_val[:, i], p_val[:, i], target, min_val_spec)
        a = choose_sensitivity_threshold(y_val[:, i], p_val[:, i], amber_target, min_val_spec * 0.9)
        amber_thr = min(a["threshold"], r["threshold"] * 0.9)
        red[c] = r["threshold"]
        amber[c] = amber_thr
        meta[c] = {
            "target_sensitivity": nominal,
            "selection_target_sensitivity": target,
            "sens_margin": sens_margin,
            "min_val_spec": min_val_spec,
            "red": r,
            "amber_target_sensitivity": amber_target,
            "amber": a,
            "amber_threshold": amber_thr,
        }
    return red, amber, meta


def subgroup_metrics(y: np.ndarray, p: np.ndarray, meta: dict, thresholds: dict) -> dict:
    skin = np.array(meta["skin_type"], dtype=int)
    groups = {"light_I_III": skin <= 3, "dark_IV_VI": skin >= 4}
    out = {}
    for name, sel in groups.items():
        if sel.sum() < 8:
            continue
        item = {"n": int(sel.sum()), "classes": {}}
        for i, c in enumerate(CLASSES):
            yc, pc = y[sel, i], p[sel, i]
            auc = float(roc_auc_score(yc, pc)) if len(np.unique(yc)) > 1 else None
            pred = pc >= thresholds[c]
            sens = float(np.mean(pred[yc == 1])) if (yc == 1).sum() > 0 else None
            item["classes"][c] = {
                "n_pos": int(yc.sum()),
                "auroc": auc,
                "sensitivity": sens,
            }
        out[name] = item
    return out


def write_predictions(
    path: Path, df: pd.DataFrame, probs: np.ndarray, thresholds: dict, amber: dict, triage_results: list[dict]
) -> None:
    out = df[["patient_id", "scenario", "skin_type"] + list(CLASSES)].copy()
    for i, c in enumerate(CLASSES):
        out[f"prob_{c}"] = np.round(probs[:, i], 5)
    out["triage"] = [t["overall"] for t in triage_results]
    out["primary_finding"] = [t["primary"] or "" for t in triage_results]
    out.to_csv(path, index=False)


def write_model_card(
    path: Path,
    args: argparse.Namespace,
    model,
    metrics: dict,
    val_metrics: dict,
    thresholds: dict,
    amber: dict,
    operating: dict,
    temperature: float,
    guardrail: dict,
    class_counts: dict,
    subgroups: dict,
    n_params: int,
) -> None:
    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def num(x, digits=3):
        return "—" if x is None else f"{x:.{digits}f}"

    n_test = metrics.get("n", 0)

    lines = []
    lines.append("# AVF Sentinel — Model Card (Research Prototype)")
    lines.append("")
    lines.append(f"> **{DISCLAIMER}**")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(f"- Version: 0.1.0 (hackathon prototype)")
    lines.append(f"- Arm branch: `{args.arm_backbone}` (pretrained={not args.no_pretrained})")
    lines.append(f"- Audio branch: `{args.audio_backbone}` (2-channel log-Mel + 16 acoustic descriptors)")
    lines.append(f"- Trainable parameters: {n_params:,}")
    lines.append(f"- Training data: synthetic cohort in `{args.data}` (seed {args.seed})")
    lines.append(f"- Patient-grouped split. No patient appears in more than one split.")
    lines.append(f"- Calibration temperature: {temperature:.3f}")
    lines.append("")
    lines.append("## Intended use (prototype)")
    lines.append("")
    lines.append("A triage-support research prototype for **suspected AV-fistula stenosis, thrombosis, aneurysm, infection, and access-site hematoma**. It combines a paired arm-down/arm-raised photo with a 10-second phone-mic bruit recording. It is designed to route **red** cases to same-day vascular-access review, **amber** cases to a dialysis-nurse examination, and never to rule out disease.")
    lines.append("")
    lines.append("**Not intended for:** autonomous diagnosis, ruling out thrombosis/stenosis, replacing duplex ultrasound or physical examination, or use without clinician oversight and local regulatory approval.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("The demo is trained on a **procedurally generated synthetic cohort**. It intentionally includes diverse Fitzpatrick skin tones, lighting casts, backgrounds, camera angles, blur/exposure failures, ambient clinic noise, and co-occurring findings. The numbers below test the software pipeline; they are **not clinical evidence**.")
    lines.append("")
    lines.append(f"Held-out test patients included in the metrics: **{n_test}** (quality-passing pairs). See `predictions_test.csv` for patient-level outputs and `dataset_stats.json` for the full cohort composition.")
    lines.append("")
    lines.append("## Test performance at the sensitivity-first operating point")
    lines.append("")
    lines.append("| Finding | AUROC | Sensitivity | Specificity | PPV | Threshold | n positive |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in CLASSES:
        m = metrics["classes"][c]
        lines.append(
            f"| {CLASS_LABELS[c]} | {num(m['auroc'])} | {pct(m['sensitivity'])} | {pct(m['specificity'])} | "
            f"{pct(m['ppv'])} | {m['threshold']:.3f} | {m['n_pos']} |"
        )
    lines.append("")
    lines.append(
        f"Macro AUROC: **{num(metrics['macro_auroc'])}**; macro sensitivity: "
        f"**{pct(metrics['macro_sensitivity'])}**; macro specificity: **{pct(metrics['macro_specificity'])}**."
    )
    lines.append(
        "Thresholds were selected on the validation split, then frozen before the held-out test evaluation. "
        "The nominal targets are 95% sensitivity for thrombosis and 90% for the other findings. "
        f"An additional {args.sens_margin:.2f} sensitivity margin was requested, and the lowest threshold "
        f"meeting that target while keeping validation specificity >= {args.min_val_spec:.2f} was chosen "
        "(fallback: maximum specificity) to leave safety margin on a small validation split. "
        "Amber uses a higher-sensitivity (lower) threshold; red uses the target operating point."
    )
    lines.append("")
    lines.append("| Finding | Red threshold | Amber threshold | Validation sensitivity | Validation specificity | Target met |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for c in CLASSES:
        op = operating[c]["red"]
        lines.append(
            f"| {CLASS_LABELS[c]} | {thresholds[c]:.3f} | {amber[c]:.3f} | {pct(op['sensitivity'])} | "
            f"{pct(op['specificity'])} | {'yes' if op['target_met'] else 'no (best Youden)'} |"
        )
    lines.append("")
    lines.append("## Capture guardrails")
    lines.append("")
    lines.append(f"- Pairs rejected: **{guardrail.get('fail', '—')}/{guardrail.get('n', '—')}** ({100 * (1 - guardrail.get('pass_rate', 0)):.1f}%)")
    for issue, s in guardrail.get("caught_by_issue", {}).items():
        lines.append(f"- `{issue}`: {s['caught']}/{s['n']} caught")
    lines.append(f"- False rejections on clean captures: {guardrail.get('false_rejections_on_clean', '—')}")
    lines.append("")
    lines.append("## Fairness / subgroup audit (held-out test)")
    lines.append("")
    lines.append("| Skin-tone group | n | Finding | n positive | AUROC | Sensitivity |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for group, item in subgroups.items():
        for c in CLASSES:
            m = item["classes"][c]
            auc_s = "—" if m["auroc"] is None else f"{m['auroc']:.3f}"
            lines.append(
                f"| {group} | {item['n']} | {CLASS_SHORT_SAFE(c)} | {m['n_pos']} | "
                f"{auc_s} | {pct(m['sensitivity'])} |"
            )
    lines.append("")
    lines.append("Subgroup n is small in this synthetic demo; intervals are wide. A real study must pre-register subgroup analyses and report confidence intervals.")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append("```")
    lines.append("arm-down photo ─┐")
    lines.append("                ├─ shared ConvNeXt-Tiny ─► [f_down, f_up, f_up−f_down, f_down⊙f_up] ─► 512")
    lines.append("arm-raised photo┘")
    lines.append("")
    lines.append("10 s bruit ─► log-Mel (128×224, 2 ch) ─► ResNet18 ─► 256")
    lines.append("             └─ 16 acoustic descriptors ─► 32")
    lines.append("")
    lines.append("12 paired-photo descriptors ─► 32")
    lines.append("                    └─ concat + modality-presence flags ─► MLP ─► 5 sigmoid outputs")
    lines.append("```")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("")
    lines.append("- Synthetic data. Every metric in this card is a **software test**, not clinical validation.")
    lines.append("- No external site, scanner, phone, or demographic validation.")
    lines.append("- The acoustic branch cannot distinguish all causes of a silent access (e.g., hypotension, needle infiltration).")
    lines.append("- The visual branch cannot reliably detect deep stenosis; it targets surface and elevation-test signals.")
    lines.append("- Sensitivity-first thresholds produce false positives. The amber pathway requires nurse/clinician review.")
    lines.append("- Label definitions must be adjudicated against duplex ultrasound and clinical outcomes in a real study.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONPATH=src python -m avfsentinel.synth --out data/synthetic --patients 300")
    lines.append("PYTHONPATH=src python -m avfsentinel.prepare --data data/synthetic")
    lines.append(f"PYTHONPATH=src python -m avfsentinel.train --data data/synthetic --out {args.out} --epochs {args.epochs}")
    lines.append("PYTHONPATH=src python -m avfsentinel.infer --checkpoint artifacts/avf-demo/best.pt \\")
    lines.append("  --down path/down.jpg --up path/up.jpg --audio path/bruit.wav")
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def CLASS_SHORT_SAFE(c: str) -> str:
    from .constants import CLASS_SHORT

    return CLASS_SHORT[c]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    out = ensure_dir(args.out)
    data_dir = Path(args.data)
    print(f"Device: {device} | data: {data_dir} | out: {out}")

    feat_df = pd.read_csv(data_dir / "features.csv")
    feat_df["patient_id"] = feat_df["patient_id"].astype(str)
    if feat_df.groupby("patient_id")["split"].nunique().max() > 1:
        raise SystemExit("Patient leakage: a patient_id appears in more than one split.")
    print("Patients per split:", feat_df.groupby("split")["patient_id"].nunique().to_dict())
    print("Findings:", {c: int(feat_df[c].sum()) for c in CLASSES})

    train_ds, val_ds, test_ds = build_datasets(data_dir, image_size=args.image_size)
    print(f"Samples (quality-passing): train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.workers)

    resume = None
    if args.from_checkpoint:
        resume = torch.load(args.from_checkpoint, map_location="cpu", weights_only=False)
        resume_cfg = dict(resume["model_config"])
        resume_cfg.pop("pretrained", None)
        model = build_model(pretrained=False, **resume_cfg).to(device)
        print(f"Recalibrating from {args.from_checkpoint} (no training)")
    else:
        model = build_model(
            arm_backbone=args.arm_backbone,
            audio_backbone=args.audio_backbone,
            pretrained=not args.no_pretrained,
        ).to(device)
    n_params = count_parameters(model)
    print(f"Model: {n_params:,} trainable parameters")

    counts = class_counts(train_ds.df)
    pos_weight = []
    for c in CLASSES:
        npos, nneg = counts[c]["positive"], counts[c]["negative"]
        pos_weight.append(float(np.clip(np.sqrt(nneg / max(npos, 1)), 1.0, 4.0)))
    criterion = FocalLoss(gamma=2.0, alpha=0.65, pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)).to(device)
    print("pos_weight:", dict(zip(CLASSES, [round(w, 2) for w in pos_weight])))

    head_params = [p for n, p in model.named_parameters() if not (n.startswith("arm.encoder") or n.startswith("audio.encoder"))]
    backbone_params = [p for n, p in model.named_parameters() if (n.startswith("arm.encoder") or n.startswith("audio.encoder"))]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.backbone_lr * 0.2)

    best_auc, best_state, best_epoch = -1.0, None, -1
    patience = 0
    history: list[dict] = []
    t0 = time.time()
    if resume is not None:
        best_state = {k: v.detach().cpu().clone() for k, v in resume["state_dict"].items()}
        best_epoch = int(resume.get("best_epoch", -1))
        best_auc = float(resume.get("val_macro_auroc", float("nan")))
        history_path = out / "history.json"
        if history_path.exists():
            history = load_json(history_path)
    for epoch in range(1, 0 if resume is not None else args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        model.set_encoders_trainable(not frozen)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, frozen)
        val_loss, val_logits, val_labels, _ = evaluate_loss(model, val_loader, criterion, device)
        val_p = 1.0 / (1.0 + np.exp(-val_logits))
        val_auc = macro_auroc(val_labels, val_p)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_macro_auroc": round(float(np.nan_to_num(val_auc, nan=0.0)), 5),
                "frozen_backbone": frozen,
                "lr_head": optimizer.param_groups[1]["lr"],
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"macro AUROC {val_auc:.4f} | {'frozen' if frozen else 'fine-tune'} | {fmt_seconds(time.time() - t0)}",
            flush=True,
        )
        score = val_auc if np.isfinite(val_auc) else -val_loss
        if score > best_auc:
            best_auc, best_epoch, patience = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    print(f"Loaded best epoch {best_epoch} (val macro AUROC {best_auc:.4f})")
    save_json(history, out / "history.json")

    # Calibration + operating points on validation, then freeze for test.
    _, val_logits, val_labels, val_meta = evaluate_loss(model, val_loader, criterion, device)
    temperature = fit_temperature(val_logits, val_labels)
    val_p = 1.0 / (1.0 + np.exp(-(val_logits / temperature)))
    thresholds, amber, operating = choose_operating_points(
        val_labels, val_p, args.target_sens, args.sens_margin, args.min_val_spec
    )
    val_metrics = compute_metrics(val_labels, val_p, thresholds)
    val_metrics["temperature"] = temperature
    print("\nValidation at calibrated operating points:")
    print(format_metrics(val_metrics))

    _, test_logits, test_labels, test_meta = evaluate_loss(model, test_loader, criterion, device)
    test_p = 1.0 / (1.0 + np.exp(-(test_logits / temperature)))
    test_metrics = compute_metrics(test_labels, test_p, thresholds)
    test_metrics["temperature"] = temperature
    test_metrics["bootstrap"] = bootstrap_ci(test_labels, test_p, thresholds, n_boot=500, seed=args.seed)
    test_metrics["subgroups"] = subgroup_metrics(test_labels, test_p, test_meta, thresholds)
    triage_results = [
        triage({c: float(test_p[i, j]) for j, c in enumerate(CLASSES)}, thresholds, amber)
        for i in range(len(test_p))
    ]
    triage_counts = {k: sum(t["overall"] == k for t in triage_results) for k in ["red", "amber", "green"]}
    test_metrics["triage"] = triage_counts
    print("\nHeld-out test at calibrated operating points:")
    print(format_metrics(test_metrics))
    print("Triage distribution:", triage_counts)

    guardrail_path = data_dir / "quality_summary.json"
    guardrail = load_json(guardrail_path) if guardrail_path.exists() else {"note": "not available"}
    stats = load_json(data_dir / "feature_stats.json")
    test_df = test_ds.df.reset_index(drop=True)
    write_predictions(out / "predictions_test.csv", test_df, test_p, thresholds, amber, triage_results)

    checkpoint = {
        "state_dict": best_state,
        "model_config": model.config,
        "classes": list(CLASSES),
        "thresholds": thresholds,
        "amber_thresholds": amber,
        "temperature": temperature,
        "feature_stats": stats,
        "image_size": args.image_size,
        "target_sensitivity": args.target_sens,
        "best_epoch": best_epoch,
        "val_macro_auroc": float(best_auc),
        "triage_policy": "red if any finding >= red threshold; amber if any >= amber threshold; else green",
        "disclaimer": DISCLAIMER,
    }
    torch.save(checkpoint, out / "best.pt")
    save_json(
        {
            "args": vars(args),
            "model_config": model.config,
            "n_parameters": n_params,
            "best_epoch": best_epoch,
            "val_macro_auroc": float(best_auc),
            "temperature": temperature,
            "thresholds": thresholds,
            "amber_thresholds": amber,
            "operating_points": operating,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "triage_counts": triage_counts,
            "class_counts_train": counts,
            "guardrail": guardrail,
            "history": history,
            "versions": {
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "disclaimer": DISCLAIMER,
        },
        out / "metrics.json",
    )
    save_json({"thresholds": thresholds, "amber_thresholds": amber, "operating_points": operating}, out / "thresholds.json")
    save_json(guardrail, out / "guardrail.json")

    if not args.no_plots:
        test_df_all = feat_df[feat_df["split"] == "test"]
        plot_training(history, out / "training_curves.png")
        plot_roc_pr(test_labels, test_p, out / "roc_pr.png")
        plot_reliability(test_labels, test_p, out / "reliability.png")
        plot_confusion(test_labels, test_p, thresholds, out / "confusion.png")
        plot_triage_distribution(triage_counts, out / "triage.png")
        plot_preprocessing_grid(test_df_all, data_dir, out / "preprocessing_examples.png", n=3)
        plot_spectrogram_grid(test_df_all, data_dir, out / "spectrogram_examples.png")
        print(f"Plots written to {out}")

    write_model_card(
        out / "MODEL_CARD.md",
        args,
        model,
        test_metrics,
        val_metrics,
        thresholds,
        amber,
        operating,
        temperature,
        guardrail,
        counts,
        test_metrics["subgroups"],
        n_params,
    )
    print(f"\nSaved checkpoint and report to {out} in {fmt_seconds(time.time() - t0)}")
    print(f"Open {out / 'MODEL_CARD.md'} for the limitations and test metrics.")


if __name__ == "__main__":
    main()
