"""Evaluation, calibration, sensitivity-first operating points, and triage."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from .constants import CLASS_LABELS, CLASS_PRIORITY, CLASSES


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def choose_threshold(
    y: np.ndarray,
    p: np.ndarray,
    target_sensitivity: float = 0.90,
    grid: np.ndarray | None = None,
) -> dict:
    """Lowest-cost threshold that reaches the sensitivity target on this split.

    Among thresholds meeting the target, maximise specificity. If the target is
    unreachable, fall back to the best Youden point and flag it.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return {"threshold": 0.5, "sensitivity": None, "specificity": None, "target_met": False, "note": "no positives"}
    if grid is None:
        grid = np.linspace(0.005, 0.995, 199)
    best = None
    best_youden = None
    for t in grid:
        pred = p >= t
        sens = float(np.mean(pred[y == 1]))
        spec = float(np.mean(~pred[y == 0]))
        if sens >= target_sensitivity:
            key = (spec, t)
            if best is None or key > best[0]:
                best = (key, t, sens, spec)
        youden = sens + spec - 1.0
        if best_youden is None or youden > best_youden[0]:
            best_youden = (youden, t, sens, spec)
    if best is not None:
        _, t, sens, spec = best
        return {"threshold": float(t), "sensitivity": sens, "specificity": spec, "target_met": True}
    _, t, sens, spec = best_youden
    return {
        "threshold": float(t),
        "sensitivity": sens,
        "specificity": spec,
        "target_met": False,
        "note": f"sensitivity target {target_sensitivity:.2f} not reachable; used best Youden",
    }


def choose_sensitivity_threshold(
    y: np.ndarray,
    p: np.ndarray,
    target_sensitivity: float = 0.90,
    min_specificity: float = 0.80,
    grid: np.ndarray | None = None,
) -> dict:
    """Sensitivity-first threshold with a specificity floor.

    ``choose_threshold`` maximises specificity among thresholds that meet the
    sensitivity target. With a small validation split that produces a threshold
    sitting exactly on the lowest positive score, leaving no margin for the
    held-out set. Here we instead take the **lowest** threshold that still meets
    the sensitivity target while keeping validation specificity at or above a
    pre-specified floor. If no threshold satisfies both constraints, we fall
    back to the maximum-specificity threshold from ``choose_threshold``.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if grid is None:
        grid = np.linspace(0.005, 0.995, 199)
    candidates = []
    for t in grid:
        pred = p >= t
        sens = float(np.mean(pred[y == 1])) if (y == 1).any() else 0.0
        spec = float(np.mean(~pred[y == 0])) if (y == 0).any() else 0.0
        if sens >= target_sensitivity and spec >= min_specificity:
            candidates.append((float(t), sens, spec))
    if candidates:
        t, sens, spec = min(candidates, key=lambda x: x[0])
        return {
            "threshold": t,
            "sensitivity": sens,
            "specificity": spec,
            "target_met": True,
            "specificity_floor_met": True,
        }
    fallback = choose_threshold(y, p, target_sensitivity, grid)
    fallback["specificity_floor_met"] = False
    fallback["note"] = (
        fallback.get("note", "") + f" Specificity floor {min_specificity:.2f} not reachable."
    ).strip()
    return fallback


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(float(temperature), 1e-3)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Single global temperature for multi-label BCE (vector scaling lite)."""
    logits = np.asarray(logits, np.float64)
    targets = np.asarray(targets, np.float64)

    def objective(log_t: float) -> float:
        z = logits / np.exp(log_t)
        # Stable BCE-with-logits.
        loss = np.maximum(z, 0) - z * targets + np.log1p(np.exp(-np.abs(z)))
        return float(loss.mean())

    res = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if sel.sum() == 0:
            continue
        ece += (sel.mean()) * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1,
        "alert_rate": float(pred.mean()),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    at_05: bool = True,
) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: dict = {"classes": {}, "n": int(len(y_true)), "prevalence": {}}
    for i, c in enumerate(CLASSES):
        y = y_true[:, i].astype(int)
        p = y_prob[:, i]
        thr = float(thresholds[c])
        m = binary_metrics(y, p, thr)
        m.update(
            {
                "threshold": thr,
                "auroc": _safe_auc(y, p),
                "auprc": _safe_ap(y, p),
                "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else None,
                "ece": expected_calibration_error(y, p),
                "n_pos": int(y.sum()),
                "n_neg": int(len(y) - y.sum()),
            }
        )
        if at_05:
            m["at_0.5"] = binary_metrics(y, p, 0.5)
        out["classes"][c] = m
        out["prevalence"][c] = float(y.mean())
    aurocs = [out["classes"][c]["auroc"] for c in CLASSES if out["classes"][c]["auroc"] is not None]
    sens = [out["classes"][c]["sensitivity"] for c in CLASSES if out["classes"][c]["sensitivity"] is not None]
    specs = [out["classes"][c]["specificity"] for c in CLASSES if out["classes"][c]["specificity"] is not None]
    out["macro_auroc"] = float(np.mean(aurocs)) if aurocs else None
    out["macro_sensitivity"] = float(np.mean(sens)) if sens else None
    out["macro_specificity"] = float(np.mean(specs)) if specs else None
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    n_boot: int = 500,
    seed: int = 13,
) -> dict:
    """Patient-level bootstrap CI for AUROC and sensitivity."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out: dict = {}
    for i, c in enumerate(CLASSES):
        aucs, senss = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            y = y_true[idx, i]
            p = y_prob[idx, i]
            if len(np.unique(y)) < 2:
                continue
            aucs.append(roc_auc_score(y, p))
            pred = p >= thresholds[c]
            if y.sum() > 0:
                senss.append(float(np.mean(pred[y == 1])))
        out[c] = {
            "auroc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else None,
            "sensitivity_ci95": [float(np.percentile(senss, 2.5)), float(np.percentile(senss, 97.5))] if senss else None,
        }
    return out


def roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(y)) < 2:
        return np.array([0, 1]), np.array([0, 1])
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def triage(probs: dict[str, float], thresholds: dict[str, float], amber_thresholds: dict[str, float] | None = None) -> dict:
    """Map calibrated probabilities to red / amber / green with a safety net."""
    amber_thresholds = amber_thresholds or {}
    findings = []
    red, amber = [], []
    for c in CLASSES:
        p = float(probs[c])
        red_thr = float(thresholds[c])
        amber_thr = float(amber_thresholds.get(c, red_thr * 0.65))
        amber_thr = min(amber_thr, red_thr * 0.999)
        status = "red" if p >= red_thr else ("amber" if p >= amber_thr else "green")
        if status == "red":
            red.append(c)
        elif status == "amber":
            amber.append(c)
        findings.append(
            {
                "finding": c,
                "label": CLASS_LABELS[c],
                "probability": round(p, 4),
                "red_threshold": round(red_thr, 4),
                "amber_threshold": round(amber_thr, 4),
                "status": status,
            }
        )
    order = {c: i for i, c in enumerate(CLASS_PRIORITY)}
    findings.sort(key=lambda x: (0 if x["status"] == "red" else 1 if x["status"] == "amber" else 2, order[x["finding"]]))

    if red:
        overall = "red"
        primary = sorted(red, key=lambda c: order[c])[0]
        messages = {
            "thrombosis": "Possible access thrombosis. Same-day vascular-access review: check thrill/bruit, arrange duplex ultrasound, do not attempt routine cannulation.",
            "stenosis": "Possible haemodynamically significant stenosis. Urgent access review and duplex ultrasound; assess recirculation and dialysis adequacy.",
            "infection": "Possible access-site infection. Same-day clinical review; look for erythema, warmth, drainage, fever.",
            "aneurysm": "Large access aneurysm/pseudoaneurysm. Vascular-access surgical review; avoid cannulating the aneurysmal segment.",
            "hematoma": "Significant access-site hematoma. Assess size, pain, distal perfusion and haemoglobin; hold anticoagulation until reviewed.",
        }
        action = messages[primary]
    elif amber:
        overall = "amber"
        primary = sorted(amber, key=lambda c: order[c])[0]
        action = (
            "Borderline signal. Dialysis nurse should examine the access at the next session; "
            "repeat capture and escalate to duplex if clinically concerned. This is not a clearance."
        )
    else:
        overall = "green"
        primary = None
        action = "No high-risk pattern detected. Continue routine access surveillance; this result cannot rule out stenosis or thrombosis."
    return {
        "overall": overall,
        "primary": primary,
        "action": action,
        "findings": findings,
    }


def format_metrics(metrics: dict) -> str:
    lines = []
    lines.append(f"{'finding':<12} {'AUROC':>6} {'AUPRC':>6} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'thr':>6} {'n+':>4}")
    for c in CLASSES:
        m = metrics["classes"][c]
        def f(x: float | None, fmt: str = ".3f") -> str:
            return "  -  " if x is None else format(x, fmt)
        lines.append(
            f"{c:<12} {f(m['auroc']):>6} {f(m['auprc']):>6} {f(m['sensitivity']):>6} "
            f"{f(m['specificity']):>6} {f(m['ppv']):>6} {m['threshold']:>6.3f} {m['n_pos']:>4d}"
        )
    if metrics.get("macro_auroc") is not None:
        lines.append(
            f"{'macro':<12} {metrics['macro_auroc']:>6.3f} {'':>6} "
            f"{metrics['macro_sensitivity']:>6.3f} {metrics['macro_specificity']:>6.3f}"
        )
    return "\n".join(lines)
