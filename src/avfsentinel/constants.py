"""Shared constants and label definitions."""

from __future__ import annotations

# Ordered model outputs. Thresholds, metrics, and checkpoints follow this order.
CLASSES: tuple[str, ...] = (
    "stenosis",
    "thrombosis",
    "aneurysm",
    "infection",
    "hematoma",
)

CLASS_LABELS: dict[str, str] = {
    "stenosis": "Outflow stenosis",
    "thrombosis": "Access thrombosis",
    "aneurysm": "Access aneurysm",
    "infection": "Infection / erythema",
    "hematoma": "Hematoma / bruising",
}

CLASS_SHORT: dict[str, str] = {
    "stenosis": "Stenosis",
    "thrombosis": "Thrombosis",
    "aneurysm": "Aneurysm",
    "infection": "Infection",
    "hematoma": "Hematoma",
}

# Triage rank: which finding drives the overall color.
CLASS_PRIORITY: tuple[str, ...] = (
    "thrombosis",
    "stenosis",
    "infection",
    "aneurysm",
    "hematoma",
)

SCENARIOS: tuple[str, ...] = (
    "normal",
    "stenosis",
    "aneurysm",
    "thrombosis",
    "infection",
)

# Approximate Fitzpatrick base colours used only by the synthetic renderer.
SKIN_TONES: dict[int, tuple[int, int, int]] = {
    1: (247, 219, 196),
    2: (238, 197, 162),
    3: (221, 171, 126),
    4: (190, 132, 88),
    5: (139, 86, 58),
    6: (89, 55, 42),
}

SKIN_TONE_NAMES: dict[int, str] = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Quality guardrail defaults. Tuned for 384 px phone captures in this prototype;
# recalibrate on the deployment device before any field study.
QUALITY = {
    "min_blur_var": 4.0,        # Laplacian variance on 512 px grayscale
    "min_mean_luma": 38.0,
    "max_mean_luma": 222.0,
    "max_clipped_frac": 0.22,
    "max_angle_deg": 45.0,
    "warn_angle_deg": 35.0,
    "min_skin_frac": 0.08,
}

AUDIO = {
    "sample_rate": 16_000,
    "duration_s": 10.0,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 40.0,
    "fmax": 7_600.0,
    "target_frames": 224,
    "db_floor": -90.0,
    "db_ceil": 0.0,
}

DISCLAIMER = (
    "RESEARCH PROTOTYPE ONLY. Not a medical device. Not validated for clinical "
    "use. Every alert requires physical examination and duplex ultrasound by a "
    "qualified clinician. Do not use to rule out thrombosis or stenosis."
)
