"""Synthetic AV-fistula dataset generator.

WHY THIS EXISTS
---------------
There is no public, labelled, patient-level dataset of AV-fistula arm photos with
paired arm-elevation images, bruit audio, and duplex ground truth. To make the
training pipeline runnable today, this module renders a deliberately *hard*
synthetic cohort with realistic nuisance factors (skin tone, lighting, camera
angle, blur, background clutter, ambient noise, co-occurring findings).

The numbers produced by training on this cohort are SOFTWARE TEST METRICS, not
clinical evidence. Swap in real, consented data via ``manifest.csv`` before any
clinical claim. See docs/REAL_DATA.md.

Finding model (multi-label):
  stenosis   = haemodynamically significant outflow stenosis (>=50% or turbulent)
  thrombosis = occluded access with absent/faint bruit
  aneurysm   = access-site aneurysm / prominent bulge
  infection  = erythema, swelling, warmth (visual proxy)
  hematoma   = bruising around cannulation/access site
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split

from .audio import synthesize_bruit, write_wav
from .constants import CLASSES, SCENARIOS, SKIN_TONES
from .utils import ensure_dir, save_json, seed_everything

# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------
_BG_STYLES = {
    "curtain": ((126, 158, 160), (86, 116, 126)),
    "chair": ((55, 75, 92), (32, 46, 60)),
    "home": ((196, 178, 152), (156, 138, 116)),
    "sheet": ((214, 218, 218), (164, 174, 180)),
    "clinic": ((172, 196, 178), (128, 158, 146)),
}


def _background(size: int, style: str, rng: np.random.Generator) -> np.ndarray:
    top, bottom = _BG_STYLES[style]
    yy = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    img = np.array(top, np.float32)[None, None, :] * (1 - yy) + np.array(bottom, np.float32)[None, None, :] * yy
    img = np.broadcast_to(img, (size, size, 3)).copy()
    xx = np.linspace(0, 1, size, dtype=np.float32)[None, :, None]

    if style == "curtain":
        folds = 0.90 + 0.10 * np.sin(xx * rng.uniform(18, 34) + rng.uniform(0, 6.28))
        folds *= 0.96 + 0.08 * np.sin(xx * rng.uniform(3, 7))
        img *= folds
        seam = np.exp(-((yy - rng.uniform(0.55, 0.8)) / 0.015) ** 2)
        img -= 18 * seam
    elif style == "chair":
        # Seat and armrest silhouettes, heavily out of focus.
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        d.rounded_rectangle(
            [int(0.08 * size), int(0.58 * size), int(0.95 * size), int(1.05 * size)],
            radius=int(0.08 * size),
            fill=(28, 38, 52),
        )
        d.rounded_rectangle(
            [int(0.62 * size), int(0.36 * size), int(0.98 * size), int(0.75 * size)],
            radius=int(0.06 * size),
            fill=(42, 56, 72),
        )
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(2, 5)))
        img = np.asarray(pil, np.float32)
    elif style == "home":
        baseboard = np.exp(-((yy - 0.86) / 0.025) ** 2)
        img -= 28 * baseboard
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil, "RGBA")
        # A picture frame / wall clutter.
        x0, y0 = rng.uniform(0.05, 0.55), rng.uniform(0.08, 0.35)
        w, h = rng.uniform(0.18, 0.3), rng.uniform(0.2, 0.32)
        d.rectangle([int(x0 * size), int(y0 * size), int((x0 + w) * size), int((y0 + h) * size)], fill=(90, 70, 60, 120), outline=(60, 45, 40, 160), width=3)
        d.rectangle([int((x0 + 0.02) * size), int((y0 + 0.02) * size), int((x0 + w - 0.02) * size), int((y0 + h - 0.02) * size)], fill=(150, 160, 150, 90))
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(1, 3)))
        img = np.asarray(pil, np.float32)
    elif style == "sheet":
        folds = 0.95 + 0.05 * np.sin(xx * rng.uniform(8, 18) + yy * rng.uniform(2, 6))
        img *= folds
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil, "RGBA")
        for _ in range(rng.integers(2, 5)):
            y = rng.uniform(0.2, 0.9) * size
            d.line([(0, y), (size, y + rng.uniform(-40, 40))], fill=(255, 255, 255, 40), width=rng.integers(2, 7))
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3.5)))
        img = np.asarray(pil, np.float32)
    else:  # clinic
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil, "RGBA")
        d.rectangle([int(0.72 * size), 0, size, int(0.55 * size)], fill=(230, 232, 226, 70))
        d.rectangle([int(0.76 * size), int(0.08 * size), int(0.97 * size), int(0.5 * size)], fill=(180, 198, 190, 60))
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(2, 4)))
        img = np.asarray(pil, np.float32)

    # Out-of-focus bokeh lights.
    bokeh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bokeh)
    for _ in range(rng.integers(3, 9)):
        r = rng.uniform(0.015, 0.06) * size
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        bd.ellipse([x - r, y - r, x + r, y + r], fill=(255, 244, 210, rng.integers(10, 35)))
    bokeh = bokeh.filter(ImageFilter.GaussianBlur(rng.uniform(6, 16)))
    b = np.asarray(bokeh, np.float32)
    alpha = b[..., 3:4] / 255.0
    img = img * (1 - alpha) + b[..., :3] * alpha
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------------------
# Arm geometry and rendering
# ---------------------------------------------------------------------------
def _capsule_polygon(
    W: int,
    cx: float,
    cy: float,
    theta: float,
    length: float,
    wrist_r: float,
    elbow_r: float,
    swell_t: float,
    swell_amp: float,
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
    u = np.array([math.sin(theta), -math.cos(theta)], np.float32)
    n = np.array([math.cos(theta), math.sin(theta)], np.float32)
    ts = np.linspace(0, 1, 120)
    radii = wrist_r + (elbow_r - wrist_r) * ts
    radii = radii * (1.0 + swell_amp * np.exp(-((ts - swell_t) / 0.16) ** 2))
    centers = np.stack([cx + u[0] * (ts - 0.48) * length, cy + u[1] * (ts - 0.48) * length], axis=1)
    left = centers + n[None, :] * radii[:, None]
    right = centers - n[None, :] * radii[:, None]
    poly = [tuple(p) for p in left] + [tuple(p) for p in right[::-1]]
    return poly, u, n, radii


def _mask_from_polygon(W: int, poly: list[tuple[float, float]]) -> np.ndarray:
    m = Image.new("L", (W, W), 0)
    ImageDraw.Draw(m).polygon(poly, fill=255)
    return np.asarray(m, np.float32) / 255.0


def _ellipse_layer(W: int, center: tuple[float, float], radii: tuple[float, float], color: tuple[int, int, int], alpha: float) -> Image.Image:
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = center
    rx, ry = radii
    d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(*color, int(255 * alpha)))
    return layer


def _blend_pil(img: np.ndarray, layer: Image.Image, mask: np.ndarray | None = None, blur: float = 0.0) -> np.ndarray:
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(layer, np.float32)
    alpha = a[..., 3:4] / 255.0
    if mask is not None:
        alpha = alpha * mask[..., None]
    return img * (1 - alpha) + a[..., :3] * alpha


def _draw_vein(img: np.ndarray, W: int, scene: dict, state: str) -> np.ndarray:
    cx, cy = scene["cx"], scene["cy"]
    u, n = scene["u"], scene["n"]
    theta = scene["theta"]
    length = scene["length"]
    access_t = scene["access_t"]
    collapse = 1.0 if state == "down" else scene["collapse_ratio"]
    base_w = scene["vein_w"] * collapse
    bulge = scene["aneurysm_sev"] * (1.0 if state == "down" else 0.55 + 0.45 * scene["collapse_ratio"])
    mask = scene["mask"]

    def point(t: float, lateral: float = 0.0) -> tuple[float, float]:
        along = (t - 0.48) * length
        return (cx + u[0] * along + n[0] * lateral, cy + u[1] * along + n[1] * lateral)

    # Vein path along the forearm, gently curved.
    tt = np.linspace(0.08, 0.62, 90)
    lateral = 0.012 * W * np.sin((tt - 0.08) / 0.54 * np.pi * 1.3 + scene["vein_phase"])
    width = np.interp(tt, [0.08, 0.2, 0.62], [base_w * 0.75, base_w, base_w * 0.55])

    # Smooth tapered tube rather than a stack of line segments.
    left = []
    right = []
    for i in range(len(tt)):
        x, y = point(tt[i], lateral[i])
        w = max(1.0, width[i]) * 0.5
        left.append((x + n[0] * w, y + n[1] * w))
        right.append((x - n[0] * w, y - n[1] * w))
    poly = left + right[::-1]
    shadow_poly = [(x + 3, y + 5) for x, y in poly]
    high_poly = []
    for i in range(len(tt)):
        x, y = point(tt[i], lateral[i] - 0.9)
        w = max(1.0, width[i]) * 0.16
        high_poly.append((x + n[0] * w, y + n[1] * w))
    for i in range(len(tt) - 1, -1, -1):
        x, y = point(tt[i], lateral[i] - 0.9)
        w = max(1.0, width[i]) * 0.16
        high_poly.append((x - n[0] * w, y - n[1] * w))

    shadow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).polygon(shadow_poly, fill=(70, 40, 35, 72))
    main = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(main).polygon(poly, fill=(*scene["vein_color"], 135))
    high = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(high).polygon(high_poly, fill=(255, 225, 205, 75))

    img = _blend_pil(img, shadow, mask=mask, blur=5.0)
    img = _blend_pil(img, main, mask=mask, blur=1.6)
    img = _blend_pil(img, high, mask=mask, blur=1.2)

    # Aneurysmal bulge: nested ellipses give a raised, dome-like appearance.
    if bulge > 0.02:
        bx, by = point(access_t)
        rx = (0.060 + 0.055 * bulge) * W
        ry = (0.050 + 0.050 * bulge) * W
        # Ellipse is oriented along the arm axis: approximate with a rotated layer.
        layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for frac, tone, alpha in [
            (1.0, (92, 52, 42), 130),
            (0.88, (scene["skin"][0] * 0.80, scene["skin"][1] * 0.66, scene["skin"][2] * 0.62), 210),
            (0.70, (scene["skin"][0] * 0.92, scene["skin"][1] * 0.80, scene["skin"][2] * 0.76), 225),
            (0.46, (min(255, scene["skin"][0] * 1.03), min(255, scene["skin"][1] * 0.94), min(255, scene["skin"][2] * 0.90)), 190),
            (0.20, (min(255, scene["skin"][0] * 1.10), min(255, scene["skin"][1] * 1.02), min(255, scene["skin"][2] * 0.98)), 120),
        ]:
            ld.ellipse([bx - rx * frac, by - ry * frac, bx + rx * frac, by + ry * frac], fill=(*[int(max(0, min(255, c))) for c in tone], alpha))
        layer = layer.rotate(math.degrees(theta), center=(bx, by), resample=Image.BICUBIC)
        img = _blend_pil(img, layer, mask=mask, blur=2.0)

        # Specular highlight.
        hx, hy = bx - n[0] * rx * 0.28 - u[0] * ry * 0.20, by - n[1] * rx * 0.28 - u[1] * ry * 0.20
        hl = _ellipse_layer(W, (hx, hy), (rx * 0.22, ry * 0.16), (255, 240, 225), 0.30)
        img = _blend_pil(img, hl, mask=mask, blur=3.0)
    return img


def _draw_skin_details(img: np.ndarray, W: int, scene: dict, rng: np.random.Generator) -> np.ndarray:
    mask = scene["mask"]
    cx, cy = scene["cx"], scene["cy"]
    u, n = scene["u"], scene["n"]
    length = scene["length"]
    skin = scene["skin"]

    # Fine texture and pores.
    small = rng.normal(0, 1, (24, 24)).astype(np.float32)
    texture = cv2.resize(small, (W, W), interpolation=cv2.INTER_CUBIC)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
    tex = 1.0 + 0.025 * texture
    img = img * (1 - mask[..., None] * 0.55) + (img * tex[..., None]) * (mask[..., None] * 0.55)
    pores = rng.normal(0, 2.0, (W, W, 1)).astype(np.float32)
    img += pores * mask[..., None]

    # Hair strokes.
    hair_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    hair_d = ImageDraw.Draw(hair_layer)
    for _ in range(int(scene["hair"] * rng.integers(35, 90))):
        t = rng.uniform(0.05, 0.85)
        lat = rng.uniform(-0.7, 0.7) * np.interp(t, np.linspace(0, 1, len(scene["radii"])), scene["radii"])
        x, y = cx + u[0] * (t - 0.48) * length + n[0] * lat, cy + u[1] * (t - 0.48) * length + n[1] * lat
        ang = rng.uniform(-0.6, 0.6)
        ln = rng.uniform(0.01, 0.035) * W
        x2, y2 = x + math.sin(ang) * ln, y + math.cos(ang) * ln
        color = (max(0, int(skin[0] * 0.55)), max(0, int(skin[1] * 0.5)), max(0, int(skin[2] * 0.5)), rng.integers(30, 80))
        hair_d.line([(x, y), (x2, y2)], fill=color, width=1)
    img = _blend_pil(img, hair_layer, mask=mask, blur=0.5)

    # Elbow and wrist creases.
    creases = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cd = ImageDraw.Draw(creases)
    for t, alpha in [(0.90, 45), (0.82, 30), (0.07, 35)]:
        p0 = (cx + u[0] * (t - 0.48) * length + n[0] * scene["radii"][0] * 0.55,
              cy + u[1] * (t - 0.48) * length + n[1] * scene["radii"][0] * 0.55)
        p1 = (cx + u[0] * (t - 0.48) * length - n[0] * scene["radii"][0] * 0.55,
              cy + u[1] * (t - 0.48) * length - n[1] * scene["radii"][0] * 0.55)
        cd.line([p0, p1], fill=(70, 40, 32, alpha), width=max(1, int(0.003 * W)))
    img = _blend_pil(img, creases, mask=mask, blur=2.0)

    # Access scar and cannulation marks.
    if scene["scar"]:
        t = scene["access_t"]
        r = float(np.interp(t, np.linspace(0, 1, len(scene["radii"])), scene["radii"]))
        p0 = (cx + u[0] * (t - 0.48) * length + n[0] * r * 0.72,
              cy + u[1] * (t - 0.48) * length + n[1] * r * 0.72)
        p1 = (cx + u[0] * (t - 0.48) * length - n[0] * r * 0.72,
              cy + u[1] * (t - 0.48) * length - n[1] * r * 0.72)
        scar = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        sd = ImageDraw.Draw(scar)
        sd.line([p0, p1], fill=(238, 214, 196, 205), width=max(2, int(0.0045 * W)))
        for off in (-0.25, 0.0, 0.25):
            px = p0[0] + (p1[0] - p0[0]) * (0.5 + off)
            py = p0[1] + (p1[1] - p0[1]) * (0.5 + off)
            sd.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(150, 110, 95, 170))
        img = _blend_pil(img, scar, mask=mask, blur=0.8)

    # Erythema.
    if scene["erythema"] > 0.01:
        bx, by = cx + u[0] * (scene["access_t"] - 0.48) * length, cy + u[1] * (scene["access_t"] - 0.48) * length
        r = 0.13 * W
        local = scene["erythema"] * np.exp(-(((np.arange(W)[None, :] - bx) ** 2 + (np.arange(W)[:, None] - by) ** 2) / (r ** 2)))
        mottle = cv2.GaussianBlur(rng.normal(0, 0.18, (W, W)).astype(np.float32), (0, 0), 3)
        local = np.clip(local * (1 + mottle), 0, 1) * mask
        img[..., 0] += 48 * local
        img[..., 1] -= 22 * local
        img[..., 2] -= 18 * local
        # Warm border.
        img[..., 0] += 10 * np.clip(local - 0.5, 0, 1)
        img = np.clip(img, 0, 255)

    # Hematoma.
    if scene["hematoma"] > 0.01:
        t = scene["access_t"] + rng.uniform(-0.10, 0.10)
        bx = cx + u[0] * (t - 0.48) * length + n[0] * rng.uniform(-0.10, 0.10) * W
        by = cy + u[1] * (t - 0.48) * length + n[1] * rng.uniform(-0.10, 0.10) * W
        r = (0.07 + 0.05 * scene["hematoma"]) * W
        local = np.exp(-(((np.arange(W)[None, :] - bx) ** 2 + (np.arange(W)[:, None] - by) ** 2) / (r ** 2)))
        local *= mask * scene["hematoma"]
        img[..., 0] *= 1 - 0.28 * local
        img[..., 1] *= 1 - 0.50 * local
        img[..., 2] *= 1 - 0.36 * local
        img[..., 0] += 25 * local
        img[..., 1] += 5 * local
        # Older bruise: yellow-green rim.
        rim = np.clip(local * 2.0, 0, 1) - local
        img[..., 1] += 22 * rim
        img[..., 2] -= 10 * rim
        img = np.clip(img, 0, 255)
    return img


def _draw_sticker(img: np.ndarray, W: int, scene: dict) -> np.ndarray:
    if not scene["sticker"]:
        return img
    cx, cy = scene["cx"], scene["cy"]
    u, n = scene["u"], scene["n"]
    t = 0.11
    r_arm = float(np.interp(t, np.linspace(0, 1, len(scene["radii"])), scene["radii"]))
    lat = -0.55 * r_arm
    x = cx + u[0] * (t - 0.48) * scene["length"] + n[0] * lat
    y = cy + u[1] * (t - 0.48) * scene["length"] + n[1] * lat
    r = 0.032 * W
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(60, 60, 60, 100))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(238, 238, 232, 245), outline=(150, 150, 145, 230), width=2)
    d.ellipse([x - r * 0.35, y - r * 0.35, x + r * 0.35, y + r * 0.35], fill=(190, 42, 42, 220))
    # 1 cm black bar for scale.
    d.rectangle([x - r * 0.55, y + r * 0.55, x + r * 0.55, y + r * 0.72], fill=(20, 20, 20, 220))
    return _blend_pil(img, layer, mask=None, blur=0.4)


def render_state(scene: dict, state: str, rng: np.random.Generator) -> np.ndarray:
    """Render one of the paired photos ('down' or 'up')."""
    W = scene["W"]
    img = scene["background"].copy()
    cx, cy = scene["cx"], scene["cy"]
    u, n = scene["u"], scene["n"]

    # Arm shadow on the background.
    shadow_mask = _mask_from_polygon(W, scene["polygon"])
    shadow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.polygon(scene["polygon"], fill=(0, 0, 0, 62))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    a = np.asarray(shadow, np.float32)
    alpha = a[..., 3:4] / 255.0 * 0.5
    img = img * (1 - alpha) + np.array([20, 18, 18], np.float32) * alpha

    # Skin base.
    poly = scene["polygon"]
    skin = np.array(scene["skin"], np.float32)
    arm = np.zeros((W, W, 3), np.float32)
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
    along = (xx - cx) * u[0] + (yy - cy) * u[1]
    across = (xx - cx) * n[0] + (yy - cy) * n[1]
    t = np.clip(along / scene["length"] + 0.48, 0, 1)
    r_local = np.interp(t, np.linspace(0, 1, len(scene["radii"])), scene["radii"])
    cylinder = np.sqrt(np.clip(1.0 - (across / np.maximum(r_local, 1e-3)) ** 2, 0, 1))
    shade = 0.70 + 0.36 * cylinder
    # Light comes from a random side; the highlight shifts across the cylinder.
    side = scene["light_side"]
    highlight = np.exp(-((across / np.maximum(r_local, 1e-3) - 0.38 * side) / 0.27) ** 2) * 0.22
    along_shade = 0.94 + 0.06 * t
    for c in range(3):
        arm[..., c] = skin[c] * shade * along_shade + 32.0 * highlight
    img = img * (1 - shadow_mask[..., None]) + arm * shadow_mask[..., None]

    # Slight swelling can alter the silhouette; already in the polygon.
    img = _draw_vein(img, W, scene, state)
    img = _draw_skin_details(img, W, scene, rng)
    img = _draw_sticker(img, W, scene)

    # Global lighting cast, direction, vignette.
    cast = np.array(scene["cast"], np.float32)
    grad_x = 1.0 + scene["light_dir"] * 0.09 * (xx / W - 0.5)
    grad_y = 1.0 - 0.08 * (yy / W - 0.5)
    vignette = 1.0 - 0.16 * (((xx - W / 2) ** 2 + (yy - W / 2) ** 2) / (0.5 * W) ** 2)
    img = img * cast[None, None, :] * grad_x[..., None] * grad_y[..., None] * vignette[..., None]
    img *= scene["exposure"]

    # Specular glare.
    gx, gy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.1, 0.5) * W
    glare = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.10 * W) ** 2)))
    img += glare[..., None] * np.array([20, 18, 14], np.float32) * scene["glare"]

    # Sensor noise, camera blur, jitter, downsample.
    img += rng.normal(0, scene["noise"], (W, W, 3))
    img = np.clip(img, 0, 255)
    if scene["motion_blur"] > 0:
        k = int(scene["motion_blur"] * 6) * 2 + 1
        kernel = np.zeros((k, k), np.float32)
        cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1.0, 1)
        kernel /= kernel.sum()
        img = cv2.filter2D(img, -1, kernel)
    if scene["blur"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), scene["blur"])
    dx, dy = rng.uniform(-0.012, 0.012) * W, rng.uniform(-0.012, 0.012) * W
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (W, W), borderMode=cv2.BORDER_REFLECT)
    out = cv2.resize(img, (scene["out_size"], scene["out_size"]), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Patient generation
# ---------------------------------------------------------------------------
def make_patient(idx: int, rng: np.random.Generator, out_size: int = 384) -> dict:
    scenario = str(rng.choice(SCENARIOS, p=[0.32, 0.24, 0.16, 0.14, 0.14]))
    skin_type = int(rng.integers(1, 7))
    base = np.array(SKIN_TONES[skin_type], np.float32)
    base = np.clip(base + rng.normal(0, 7, 3), 30, 255)
    skin = tuple(int(c) for c in base)

    # Multi-label co-occurrence, with thrombosis kept distinct from stenosis.
    stenosis = scenario == "stenosis" or (scenario == "aneurysm" and rng.random() < 0.35) or (scenario == "normal" and rng.random() < 0.03)
    thrombosis = scenario == "thrombosis"
    if thrombosis:
        stenosis = False
    aneurysm = scenario == "aneurysm" or (scenario == "stenosis" and rng.random() < 0.25) or (scenario == "thrombosis" and rng.random() < 0.20)
    infection = scenario == "infection" or (scenario == "thrombosis" and rng.random() < 0.18) or (scenario == "stenosis" and rng.random() < 0.05)
    hematoma = (scenario == "thrombosis" and rng.random() < 0.30) or (scenario == "infection" and rng.random() < 0.25) or (rng.random() < 0.05)

    sev = {c: float(rng.uniform(0.35, 1.0)) for c in CLASSES}
    # 12% of positive cases are deliberately borderline ("mild"), which is where
    # sensitivity-first thresholding and the amber pathway matter.
    mild = rng.random() < 0.12
    if mild:
        for c in CLASSES:
            sev[c] *= rng.uniform(0.35, 0.65)

    if thrombosis:
        collapse = rng.uniform(0.90, 1.02)
    elif stenosis:
        collapse = rng.uniform(0.72, 0.98)
    elif aneurysm:
        collapse = rng.uniform(0.50, 0.80)
    elif infection:
        collapse = rng.uniform(0.42, 0.72)
    else:
        collapse = rng.uniform(0.32, 0.62)
    if mild:
        collapse = 0.65 * collapse + 0.35  # less visible collapse

    # Bruit type. Thrombosis is usually silent; stenosis is turbulent.
    r = rng.random()
    if thrombosis:
        bruit = "turbulent" if (r < 0.15 and not mild) else ("absent" if r < 0.85 else "laminar")
    elif stenosis:
        bruit = "turbulent" if r < (0.60 if mild else 0.88) else "laminar"
    elif aneurysm:
        bruit = "turbulent" if r < 0.22 else ("absent" if r < 0.30 else "laminar")
    else:
        bruit = "turbulent" if r < 0.06 else ("absent" if r < 0.12 else "laminar")

    # Quality nuisance. 4% get a bad angle, 5% blur, 5% exposure problems.
    q = rng.choice(["none", "blur", "dark", "bright", "angle"], p=[0.86, 0.05, 0.03, 0.02, 0.04])
    theta = math.radians(rng.normal(0, 11))
    if q == "angle":
        theta = math.radians(rng.choice([-1, 1]) * rng.uniform(48, 66))
    theta = float(np.clip(theta, -1.1, 1.1))
    exposure = 1.0
    blur = rng.uniform(0, 0.7)
    motion = 0.0
    if q == "dark":
        exposure = rng.uniform(0.30, 0.46)
    elif q == "bright":
        exposure = rng.uniform(1.40, 1.65)
    elif q == "blur":
        blur = rng.uniform(2.4, 4.0)
        motion = rng.uniform(0, 0.5)

    style = str(rng.choice(list(_BG_STYLES.keys())))
    W = 768
    bg = _background(W, style, rng)
    bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.3, 1.4))
    cx = W * rng.uniform(0.46, 0.56)
    cy = W * rng.uniform(0.48, 0.54)
    length = W * rng.uniform(1.35, 1.55)
    wrist_r = W * rng.uniform(0.120, 0.145)
    elbow_r = W * rng.uniform(0.170, 0.200)
    access_t = rng.uniform(0.18, 0.34)
    swelling = 0.0
    if thrombosis:
        swelling = rng.uniform(0.03, 0.11)
    elif infection:
        swelling = rng.uniform(0.02, 0.08)

    u = np.array([math.sin(theta), -math.cos(theta)], np.float32)
    n = np.array([math.cos(theta), math.sin(theta)], np.float32)
    # Raised arm sits slightly higher in frame, mimicking the elevation test.
    polygon, u, n, radii = _capsule_polygon(
        W, cx, cy, theta, length, wrist_r, elbow_r, access_t, swelling
    )
    mask = _mask_from_polygon(W, polygon)

    cast = rng.choice(
        [
            (1.06, 1.00, 0.91),
            (0.94, 1.00, 1.07),
            (1.00, 1.00, 1.00),
            (1.03, 1.01, 0.97),
        ]
    )
    vein_color = (
        int(np.clip(skin[0] * 0.91 + 12, 0, 255)),
        int(np.clip(skin[1] * 0.85 + 4, 0, 255)),
        int(np.clip(skin[2] * 0.88 + 8, 0, 255)),
    )
    scene = {
        "W": W,
        "out_size": out_size,
        "background": bg,
        "polygon": polygon,
        "mask": mask,
        "cx": cx,
        "cy": cy,
        "u": u,
        "n": n,
        "theta": theta,
        "length": length,
        "radii": radii,
        "access_t": access_t,
        "skin": skin,
        "skin_type": skin_type,
        "vein_w": W * rng.uniform(0.055, 0.075),
        "vein_color": vein_color,
        "vein_phase": rng.uniform(0, 6.28),
        "collapse_ratio": float(collapse),
        "aneurysm_sev": float(sev["aneurysm"]) if aneurysm else 0.0,
        "erythema": float(sev["infection"]) if infection else 0.0,
        "hematoma": float(sev["hematoma"]) if hematoma else 0.0,
        "scar": bool(rng.random() < 0.85),
        "sticker": bool(rng.random() < 0.88),
        "hair": float(rng.uniform(0, 1)),
        "blur": blur,
        "motion_blur": motion,
        "noise": float(rng.uniform(1.2, 4.0)),
        "exposure": exposure,
        "cast": cast,
        "light_side": float(rng.choice([-1, 1])),
        "light_dir": float(rng.uniform(-1, 1)),
        "glare": float(rng.uniform(0.2, 1.0)),
        "quality_issue": str(q),
    }

    labels = {
        "stenosis": int(stenosis),
        "thrombosis": int(thrombosis),
        "aneurysm": int(aneurysm),
        "infection": int(infection),
        "hematoma": int(hematoma),
    }
    # Doppler-style ground truth for real-data compatibility.
    if thrombosis:
        doppler_pct = 100.0
    elif stenosis:
        doppler_pct = float(np.clip(rng.normal(68, 12), 50, 95))
    else:
        doppler_pct = float(np.clip(rng.normal(22, 14), 0, 45))
    if mild and stenosis:
        doppler_pct = float(np.clip(rng.normal(52, 8), 50, 65))

    return {
        "patient_id": f"SYN{idx:04d}",
        "scenario": scenario,
        "skin_type": skin_type,
        "labels": labels,
        "severity": sev,
        "mild": int(mild),
        "collapse_ratio": float(collapse),
        "aneurysm_sev": float(sev["aneurysm"]) if aneurysm else 0.0,
        "infection_sev": float(sev["infection"]) if infection else 0.0,
        "hematoma_sev": float(sev["hematoma"]) if hematoma else 0.0,
        "bruit_type": bruit,
        "bruit_present": int(bruit != "absent"),
        "heart_rate": float(rng.uniform(58, 98)),
        "doppler_stenosis_pct": doppler_pct,
        "quality_issue": q,
        "scene": scene,
    }


def generate_patient_files(patient: dict, images_dir: Path, audio_dir: Path, rng: np.random.Generator) -> dict:
    scene = patient["scene"]
    rows = {}
    for state, shift in [("down", 0.0), ("up", 0.038 * scene["W"])]:
        local_scene = dict(scene)
        dx, dy = scene["u"][0] * shift, scene["u"][1] * shift
        local_scene["cx"] = scene["cx"] + dx
        local_scene["cy"] = scene["cy"] + dy
        # Translate the existing geometry so swelling and taper are preserved.
        poly = [(x + dx, y + dy) for x, y in scene["polygon"]]
        local_scene["polygon"] = poly
        local_scene["mask"] = _mask_from_polygon(scene["W"], poly)
        img = render_state(local_scene, state, rng)
        path = images_dir / f"{patient['patient_id']}_{state}.jpg"
        Image.fromarray(img).save(path, quality=88, subsampling=1)
        rows[f"image_{state}"] = f"images/{path.name}"

    sev = patient["severity"]
    y = synthesize_bruit(
        rng,
        bruit_type=patient["bruit_type"],
        severity=sev["stenosis"] if patient["bruit_type"] == "turbulent" else sev["aneurysm"],
        heart_rate=patient["heart_rate"],
        ambient_level=float(rng.uniform(0.020, 0.060)),
        gain_db=float(rng.uniform(-4, 4)),
    )
    audio_name = f"{patient['patient_id']}.wav"
    write_wav(audio_dir / audio_name, y)
    rows["audio"] = f"audio/{audio_name}"
    rows.update(
        {
            "patient_id": patient["patient_id"],
            "scenario": patient["scenario"],
            "skin_type": patient["skin_type"],
            "collapse_ratio": f"{patient['collapse_ratio']:.4f}",
            "bruit_type": patient["bruit_type"],
            "bruit_present": patient["bruit_present"],
            "doppler_stenosis_pct": f"{patient['doppler_stenosis_pct']:.1f}",
            "mild": patient["mild"],
            "quality_issue": patient["quality_issue"],
        }
    )
    for c in CLASSES:
        rows[c] = patient["labels"][c]
    return rows


def generate_dataset(out: str | Path, n_patients: int = 300, seed: int = 7, out_size: int = 384) -> dict:
    out = ensure_dir(out)
    images_dir = ensure_dir(out / "images")
    audio_dir = ensure_dir(out / "audio")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    t0 = time.time()
    for i in range(n_patients):
        patient = make_patient(i, rng, out_size=out_size)
        rows.append(generate_patient_files(patient, images_dir, audio_dir, rng))
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  generated {i + 1}/{n_patients} patients ({time.time() - t0:.1f}s)", flush=True)

    # Patient-level, scenario-stratified split.
    idx = np.arange(n_patients)
    strata_fine = [f"{r['scenario']}_{'dark' if int(r['skin_type']) >= 4 else 'light'}" for r in rows]
    strata_coarse = [r["scenario"] for r in rows]

    def split_two(
        indices: np.ndarray, test_size: float, strata_candidates: list[list[str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        for strata in strata_candidates:
            try:
                a, b = train_test_split(
                    indices, test_size=test_size, random_state=seed, stratify=[strata[i] for i in indices]
                )
                return np.array(a), np.array(b)
            except ValueError:
                continue
        a, b = train_test_split(indices, test_size=test_size, random_state=seed)
        return np.array(a), np.array(b)

    train_idx, temp_idx = split_two(idx, 0.30, [strata_fine, strata_coarse])
    val_idx, test_idx = split_two(temp_idx, 0.50, [strata_fine, strata_coarse])
    split = np.array(["train"] * n_patients, dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    for i, row in enumerate(rows):
        row["split"] = split[i]

    fields = [
        "patient_id", "split", "scenario", "skin_type", "image_down", "image_up", "audio",
        "stenosis", "thrombosis", "aneurysm", "infection", "hematoma",
        "collapse_ratio", "bruit_type", "bruit_present", "doppler_stenosis_pct", "mild", "quality_issue",
    ]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        "patients": n_patients,
        "splits": {s: int(np.sum(split == s)) for s in ["train", "val", "test"]},
        "scenarios": {s: sum(r["scenario"] == s for r in rows) for s in SCENARIOS},
        "labels": {c: sum(int(r[c]) for r in rows) for c in CLASSES},
        "skin_types": {str(k): sum(int(r["skin_type"]) == k for r in rows) for k in range(1, 7)},
        "quality_issues": {q: sum(r["quality_issue"] == q for r in rows) for q in ["none", "blur", "dark", "bright", "angle"]},
        "seed": seed,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Synthetic demo cohort. Metrics on this data are software tests, not clinical evidence.",
    }
    save_json(counts, out / "dataset_stats.json")
    print(f"Dataset written to {out} ({n_patients} patients, {time.time() - t0:.1f}s)")
    print("Splits:", counts["splits"])
    print("Labels:", counts["labels"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic AVF demo cohort.")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--patients", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    seed_everything(args.seed)
    generate_dataset(args.out, args.patients, args.seed, args.size)


if __name__ == "__main__":
    main()
