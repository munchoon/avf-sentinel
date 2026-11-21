"""Image quality guardrails, arm ROI extraction, colour normalisation, and
interpretable paired-photo features.

The two photos are captured arm-down and arm-raised.  Measurements are made on
the *pair*, so camera distance/angle changes are partially cancelled by
normalising widths by arm length.  Deep features come from the shared CNN; the
handcrafted vector below gives the fusion head explicit, auditable cues.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .constants import QUALITY

PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "width_ratio_up_down",
    "area_ratio_up_down",
    "access_width_down_norm",
    "access_width_up_norm",
    "bulge_ratio_up_down",
    "redness_down",
    "redness_delta_up_minus_down",
    "vein_contrast_down",
    "vein_contrast_ratio_up_down",
    "dark_frac_down",
    "dark_frac_delta",
    "luma_delta_up_minus_down",
)


@dataclass
class QualityReport:
    blur_var: float
    mean_luma: float
    clipped_frac: float
    angle_deg: float
    skin_frac: float
    issues: list[str] = field(default_factory=list)
    hard_fail: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArmStats:
    angle_deg: float
    arm_length: float
    area_frac: float
    access_width_norm: float
    bulge_contrast: float
    redness: float
    luma: float
    vein_contrast: float
    dark_frac: float

    def as_dict(self) -> dict:
        return asdict(self)


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Heuristic YCrCb + HSV skin segmentation with a centre-prior fallback."""
    h, w = rgb.shape[:2]
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    y, cr, cb = cv2.split(ycrcb)
    hue, sat, val = cv2.split(hsv)

    m = (cr >= 132) & (cr <= 181) & (cb >= 76) & (cb <= 136) & (y >= 28)
    hue_ok = ((hue <= 28) | (hue >= 160)) & (sat >= 12) & (sat <= 230) & (val >= 35)
    mask = (m & hue_ok).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep the largest central-ish component that looks like a limb, not the
    # whole background. A component spanning nearly the full frame is almost
    # always a wall/curtain wrongly included by colour thresholds.
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best, best_score = 0, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.02 * h * w:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area > 0.86 * h * w or (bw > 0.94 * w and bh > 0.90 * h):
            continue
        cx, cy = centroids[i]
        centre_dist = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        score = area / (h * w) - 0.35 * centre_dist
        if score > best_score:
            best, best_score = i, score
    if best == 0:
        # Fallback: central vertical capsule. Quality reports low coverage.
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (((xx - w / 2) / (0.28 * w)) ** 2 + ((yy - h / 2) / (0.50 * h)) ** 2 <= 1).astype(np.uint8)
    else:
        mask = (labels == best).astype(np.uint8)
    # Fill holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _pca_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    # Point the axis toward the elbow (upward in image coordinates).
    if v[1] > 0:
        v = -v
    v = v / (np.linalg.norm(v) + 1e-8)
    n = np.array([-v[1], v[0]], np.float32)
    return centre, v, n


def _edge_angle(gray: np.ndarray) -> float | None:
    """Estimate the dominant near-vertical long edge, a mask-free angle fallback."""
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (min(w, 512), min(h, 512)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 40, 120)
    min_len = int(0.22 * min(small.shape[:2]))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(30, min_len // 3), minLineLength=min_len, maxLineGap=18)
    if lines is None:
        return None
    vals = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        if dy < 10:
            continue
        angle = float(np.degrees(np.arctan2(dx, dy)))
        if angle <= 75:
            vals.append((float(np.hypot(dx, dy)), angle))
    if not vals:
        return None
    vals.sort(reverse=True)
    top = vals[:6]
    weights = np.array([v[0] for v in top], np.float32)
    angles = np.array([v[1] for v in top], np.float32)
    return float(np.average(angles, weights=weights))


def quality_check(rgb: np.ndarray, mask: np.ndarray | None = None) -> QualityReport:
    h, w = rgb.shape[:2]
    if mask is None:
        mask = skin_mask(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    blur_var = float(cv2.Laplacian(small, cv2.CV_64F).var())

    if mask.sum() > 50:
        luma = gray[mask]
    else:
        luma = gray.reshape(-1)
    mean_luma = float(np.mean(luma))
    clipped = float(np.mean((luma < 8) | (luma > 248)))
    skin_frac = float(mask.mean())

    centre, v, _ = _pca_axis(mask)
    angle = float(np.degrees(np.arccos(np.clip(abs(v[1]), 0, 1))))  # from vertical
    edge_angle = _edge_angle(gray)
    if edge_angle is not None and angle < 5.0 and edge_angle > 18.0:
        # Skin mask fell back to a central capsule (common on beige walls).
        angle = edge_angle
    issues: list[str] = []
    if blur_var < QUALITY["min_blur_var"]:
        issues.append("Image looks blurry — hold still and tap to focus.")
    if mean_luma < QUALITY["min_mean_luma"]:
        issues.append("Too dark — move to a brighter area or turn on the light.")
    if mean_luma > QUALITY["max_mean_luma"]:
        issues.append("Overexposed — avoid direct light or flash on the skin.")
    if clipped > QUALITY["max_clipped_frac"]:
        issues.append("Strong glare/clipping — change the angle slightly.")
    if angle > QUALITY["warn_angle_deg"]:
        issues.append("Camera is tilted — keep the forearm vertical and camera level.")
    if skin_frac < QUALITY["min_skin_frac"]:
        issues.append("Forearm not fully visible — fill the frame with the access arm.")
    hard = (
        blur_var < QUALITY["min_blur_var"]
        or mean_luma < QUALITY["min_mean_luma"]
        or mean_luma > QUALITY["max_mean_luma"]
        or clipped > QUALITY["max_clipped_frac"]
        or angle > QUALITY["max_angle_deg"]
        or skin_frac < QUALITY["min_skin_frac"]
    )
    return QualityReport(blur_var, mean_luma, clipped, angle, skin_frac, issues, hard)


def _reference_white(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Find a compact, near-neutral bright patch (colour-reference sticker)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, sat, val = cv2.split(hsv)
    cand = ((sat < 38) & (val > 175) & mask).astype(np.uint8)
    cand = cv2.erode(cand, np.ones((3, 3), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    best, best_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 15 or area > 0.02 * rgb.shape[0] * rgb.shape[1]:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 2.5:
            continue
        if area > best_area:
            best, best_area = i, area
    if best is None:
        return None
    px = rgb[labels == best].reshape(-1, 3).astype(np.float32)
    ref = np.percentile(px, 60, axis=0)
    if ref.mean() < 120:
        return None
    return ref


def white_balance(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Single-image white balance, kept for ad-hoc use.

    Prefer :meth:`Preprocessor.process_pair`, which shares one illuminant
    estimate across the paired captures.
    """
    ref = _reference_white(rgb, mask)
    if ref is None:
        return rgb, "none"
    gains = float(ref.mean()) / np.maximum(ref, 1.0)
    gains = np.clip(gains, 0.75, 1.35)
    gains = gains / gains.mean()
    out = np.clip(rgb.astype(np.float32) * gains[None, None, :], 0, 255)
    return out.astype(np.uint8), "sticker"


def _mild_gray_world(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Very conservative gray-world on background pixels.

    Full gray-world turns brown skin gray. We only nudge 25% toward neutral and
    use low-saturation pixels outside the arm when possible.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    px = rgb[(~mask) & (sat < 70)].reshape(-1, 3).astype(np.float32)
    if len(px) < 300:
        px = rgb.reshape(-1, 3).astype(np.float32)
    med = np.median(px, axis=0)
    gains = float(med.mean()) / np.maximum(med, 1.0)
    gains = np.clip(gains, 0.88, 1.14)
    gains = 1.0 + 0.25 * (gains - 1.0)
    out = np.clip(rgb.astype(np.float32) * gains[None, None, :], 0, 255)
    return out.astype(np.uint8), "gray_world_mild"


def _shared_white_balance(
    down: np.ndarray, up: np.ndarray, mask_d: np.ndarray, mask_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    refs = [r for r in (_reference_white(down, mask_d), _reference_white(up, mask_u)) if r is not None]
    if refs:
        ref = np.mean(refs, axis=0)
        gains = float(ref.mean()) / np.maximum(ref, 1.0)
        gains = np.clip(gains, 0.75, 1.35)
        gains = gains / gains.mean()
        method = "sticker"
        out_d = np.clip(down.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
        out_u = np.clip(up.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
        return out_d, out_u, method
    return (_mild_gray_world(down, mask_d)[0], _mild_gray_world(up, mask_u)[0], "gray_world_mild")


def _shared_crop_box(
    masks: list[np.ndarray], shape: tuple[int, int], pad: float = 1.18
) -> tuple[int, int, int, int]:
    """One crop box for both photos so the paired comparison stays aligned."""
    h, w = shape
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            continue
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if not boxes:
        side = int(min(h, w) * 0.9)
        x0, y0 = (w - side) // 2, (h - side) // 2
        return x0, y0, x0 + side, y0 + side
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    side = int(max(x1 - x0, y1 - y0) * pad)
    side = max(side, int(0.70 * min(h, w)))
    side = min(side, min(h, w))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return x0, y0, x0 + side, y0 + side


def apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def crop_arm(rgb: np.ndarray, mask: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        h, w = rgb.shape[:2]
        side = int(min(h, w) * 0.82)
        x0, y0 = (w - side) // 2, (h - side) // 2
        return cv2.resize(rgb[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_AREA), mask[y0 : y0 + side, x0 : x0 + side]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    side = int(max(w, h) * 1.16)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = int(np.clip(cx - side // 2, 0, rgb.shape[1] - 1))
    y0 = int(np.clip(cy - side // 2, 0, rgb.shape[0] - 1))
    x1 = int(np.clip(x0 + side, 1, rgb.shape[1]))
    y1 = int(np.clip(y0 + side, 1, rgb.shape[0]))
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    crop = rgb[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    cmask = cv2.resize(cmask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return crop, cmask


def analyse_arm(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[ArmStats, dict]:
    if mask is None:
        mask = skin_mask(rgb)
    centre, v, n = _pca_axis(mask)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rel = pts - centre
    along = rel @ v
    across = rel @ n
    lo, hi = float(along.min()), float(along.max())
    arm_len = max(hi - lo, 1.0)
    t = (along - lo) / arm_len  # 0 = wrist, 1 = elbow
    bins = np.linspace(0, 1, 34)
    widths = np.zeros(len(bins) - 1, np.float32)
    for i in range(len(bins) - 1):
        sel = (t >= bins[i]) & (t < bins[i + 1])
        if sel.sum() > 12:
            widths[i] = np.percentile(across[sel], 95) - np.percentile(across[sel], 5)
        else:
            widths[i] = np.nan
    if np.all(np.isnan(widths)):
        widths[:] = 1.0
    med = np.nanmedian(widths)
    widths = np.where(np.isnan(widths), med, widths)
    access = widths[(bins[:-1] >= 0.16) & (bins[:-1] < 0.44)]
    access_width = float(np.median(access) / arm_len) if len(access) else float(np.median(widths) / arm_len)
    bulge = float(np.max(widths[(bins[:-1] >= 0.04) & (bins[:-1] < 0.7)]) / (med + 1e-6))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan = lab[..., 0], lab[..., 1]
    l_vals = l_chan[mask]
    a_vals = a_chan[mask]
    width_at_t = np.interp(t, bins[:-1], widths)
    access_sel = (t >= 0.15) & (t <= 0.48) & (np.abs(across) <= 0.55 * width_at_t)
    if access_sel.sum() < 30:
        access_sel = np.ones_like(access_sel, dtype=bool)
    redness = float(np.mean(a_vals[access_sel]))
    luma = float(np.mean(l_vals))
    # Local contrast in the central access band picks up a distended vein.
    central = (t >= 0.10) & (t <= 0.55) & (np.abs(across) <= 0.30 * width_at_t)
    hp = l_chan - cv2.GaussianBlur(l_chan, (0, 0), 5.0)
    hp_vals = hp[mask]
    vein_contrast = float(np.percentile(np.abs(hp_vals[central]), 92)) if central.sum() > 20 else 0.0
    dark_frac = float(np.mean(l_vals[access_sel] < np.percentile(l_vals, 22))) if access_sel.sum() > 20 else 0.0
    angle = float(np.degrees(np.arccos(np.clip(abs(v[1]), 0, 1))))
    stats = ArmStats(
        angle_deg=angle,
        arm_length=arm_len,
        area_frac=float(mask.mean()),
        access_width_norm=access_width,
        bulge_contrast=bulge,
        redness=redness,
        luma=luma,
        vein_contrast=vein_contrast,
        dark_frac=dark_frac,
    )
    details = {"width_profile": widths.tolist(), "n_mask_px": int(mask.sum())}
    return stats, details


def pair_features(down: ArmStats, up: ArmStats) -> np.ndarray:
    return np.array(
        [
            up.access_width_norm / (down.access_width_norm + 1e-6),
            up.area_frac / (down.area_frac + 1e-6),
            down.access_width_norm,
            up.access_width_norm,
            up.bulge_contrast / (down.bulge_contrast + 1e-6),
            down.redness,
            up.redness - down.redness,
            down.vein_contrast,
            up.vein_contrast / (down.vein_contrast + 1e-6),
            down.dark_frac,
            up.dark_frac - down.dark_frac,
            up.luma - down.luma,
        ],
        dtype=np.float32,
    )


class Preprocessor:
    def __init__(self, size: int = 256) -> None:
        self.size = size

    def _finish(self, rgb: np.ndarray, mask: np.ndarray, quality: QualityReport | None, wb_method: str, details_extra: dict | None = None) -> dict:
        crop, cmask = crop_arm(rgb, mask, size=self.size)
        crop = apply_clahe(crop)
        stats, details = analyse_arm(crop, cmask)
        if details_extra:
            details.update(details_extra)
        return {
            "image": crop,
            "mask": cmask,
            "quality": quality,
            "stats": stats,
            "details": details,
            "wb_method": wb_method,
        }

    def process(self, rgb: np.ndarray, check_quality: bool = True) -> dict:
        mask = skin_mask(rgb)
        quality = quality_check(rgb, mask) if check_quality else None
        wb, wb_method = white_balance(rgb, mask)
        return self._finish(wb, mask, quality, wb_method)

    def process_pair(self, down_rgb: np.ndarray, up_rgb: np.ndarray, check_quality: bool = True) -> dict:
        """Preprocess both photos with a shared crop and illuminant estimate.

        The paired elevation test depends on comparing like with like; independent
        bounding boxes would shift/scale the two images and destroy that signal.
        """
        h, w = down_rgb.shape[:2]
        if up_rgb.shape[:2] != (h, w):
            up_rgb = cv2.resize(up_rgb, (w, h), interpolation=cv2.INTER_AREA)
        md = skin_mask(down_rgb)
        mu = skin_mask(up_rgb)
        qd = quality_check(down_rgb, md) if check_quality else None
        qu = quality_check(up_rgb, mu) if check_quality else None
        wb_d, wb_u, wb_method = _shared_white_balance(down_rgb, up_rgb, md, mu)
        x0, y0, x1, y1 = _shared_crop_box([md, mu], (h, w))
        out = {}
        for name, rgb, mask, quality in [("down", wb_d, md, qd), ("up", wb_u, mu, qu)]:
            crop = rgb[y0:y1, x0:x1]
            cmask = mask[y0:y1, x0:x1]
            if crop.size == 0:
                crop = rgb
                cmask = mask
            crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
            cmask = cv2.resize(cmask.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(bool)
            crop = apply_clahe(crop)
            stats, details = analyse_arm(crop, cmask)
            details["crop_box"] = [int(x0), int(y0), int(x1), int(y1)]
            out[name] = {
                "image": crop,
                "mask": cmask,
                "quality": quality,
                "stats": stats,
                "details": details,
                "wb_method": wb_method,
            }
        out["wb_method"] = wb_method
        out["crop_box"] = [int(x0), int(y0), int(x1), int(y1)]
        return out

    @staticmethod
    def read(path: str | Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
