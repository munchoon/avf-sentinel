"""Inference CLI + self-contained HTML triage report."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .audio import acoustic_features, log_mel, read_wav
from .constants import CLASS_LABELS, CLASSES, DISCLAIMER, IMAGENET_MEAN, IMAGENET_STD
from .metrics import triage
from .models import build_model
from .preprocessing import PAIR_FEATURE_NAMES, Preprocessor, pair_features
from .utils import ensure_dir, get_device, load_json


def _b64_image(arr: np.ndarray, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, quality=90)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_spectrogram(db: np.ndarray) -> str:
    lo = float(np.percentile(db, 5))
    hi = float(np.percentile(db, 99.5))
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _b64_image(img, "PNG")


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = get_device(device)
        self.checkpoint_path = str(checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["model_config"])
        cfg.pop("pretrained", None)
        self.model = build_model(pretrained=False, **cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ckpt = ckpt
        self.classes = ckpt.get("classes", list(CLASSES))
        self.thresholds = ckpt["thresholds"]
        self.amber = ckpt["amber_thresholds"]
        self.temperature = float(ckpt["temperature"])
        self.feature_stats = ckpt["feature_stats"]
        self.image_size = int(ckpt.get("image_size", 224))
        self.pre = Preprocessor(size=256)

    def _tensor_pair(self, down_rgb: np.ndarray, up_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        def prep(arr):
            t = TF.resize(torch.from_numpy(arr).permute(2, 0, 1), [self.image_size, self.image_size], antialias=True)
            return (t.float() / 255.0 - mean) / std

        return prep(down_rgb).unsqueeze(0), prep(up_rgb).unsqueeze(0)

    def predict(
        self,
        down_rgb: np.ndarray,
        up_rgb: np.ndarray,
        audio_path: str | Path | None = None,
        force: bool = False,
    ) -> dict:
        pair = self.pre.process_pair(down_rgb, up_rgb)
        pd_ = pair["down"]
        pu_ = pair["up"]
        hard_fail = pd_["quality"].hard_fail or pu_["quality"].hard_fail
        quality = {
            "down": pd_["quality"].as_dict(),
            "up": pu_["quality"].as_dict(),
            "wb_down": pair["wb_method"],
            "wb_up": pair["wb_method"],
        }
        result: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(getattr(self, "checkpoint_path", "")),
            "quality": quality,
            "capture_ok": not hard_fail,
            "has_audio": False,
            "probabilities": {c: None for c in CLASSES},
            "triage": None,
            "disclaimer": DISCLAIMER,
        }
        if hard_fail and not force:
            issues = [i for q in ("down", "up") for i in quality[q]["issues"]]
            result["triage"] = {
                "overall": "retake",
                "primary": None,
                "action": "Retake both photos before running the model. " + " ".join(sorted(set(issues))),
                "findings": [],
            }
            result["images"] = {
                "down_raw": _b64_image(down_rgb),
                "up_raw": _b64_image(up_rgb),
                "down": _b64_image(pd_["image"]),
                "up": _b64_image(pu_["image"]),
            }
            return result

        down_t, up_t = self._tensor_pair(pd_["image"], pu_["image"])
        pf = np.clip(pair_features(pd_["stats"], pu_["stats"]), -8, 8)
        pf = (pf - np.array(self.feature_stats["pair_mean"], np.float32)) / np.array(self.feature_stats["pair_std"], np.float32)

        has_audio = False
        spec_t = None
        ac_t = None
        db = None
        if audio_path is not None:
            y = read_wav(audio_path)
            db = log_mel(y)
            base = (np.clip(db, -90, 0) + 90.0) / 90.0 * 2.0 - 1.0
            smooth = cv2.GaussianBlur(db, (0, 0), sigmaX=2.5, sigmaY=2.5)
            contrast = np.clip((db - smooth) / 8.0, -1.0, 1.0)
            spec_t = torch.from_numpy(np.stack([base, contrast], axis=0)).float().unsqueeze(0).to(self.device)
            af = np.clip(acoustic_features(db, y), -12, 12)
            af = (af - np.array(self.feature_stats["acoustic_mean"], np.float32)) / np.array(self.feature_stats["acoustic_std"], np.float32)
            ac_t = torch.from_numpy(af).float().unsqueeze(0).to(self.device)
            has_audio = True

        with torch.inference_mode():
            logits = self.model(
                down_t.to(self.device),
                up_t.to(self.device),
                spec_t,
                torch.from_numpy(pf).float().unsqueeze(0).to(self.device),
                ac_t,
                has_audio=has_audio,
            )
        probs = 1.0 / (1.0 + np.exp(-(logits.float().cpu().numpy()[0] / self.temperature)))
        prob_map = {c: float(probs[i]) for i, c in enumerate(CLASSES)}
        tr = triage(prob_map, self.thresholds, self.amber)
        result.update(
            {
                "has_audio": has_audio,
                "probabilities": prob_map,
                "triage": tr,
                "images": {
                    "down_raw": _b64_image(down_rgb),
                    "up_raw": _b64_image(up_rgb),
                    "down": _b64_image(pd_["image"]),
                    "up": _b64_image(pu_["image"]),
                },
            }
        )
        if db is not None:
            result["spectrogram"] = _b64_spectrogram(db)
        return result


def render_report(result: dict, out_path: str | Path, back_url: str | None = None) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    tri = result["triage"]
    overall = tri["overall"]
    colors = {
        "red": ("#a4161a", "#fff5f5"),
        "amber": ("#a15c00", "#fff9e8"),
        "green": ("#1b6b4a", "#f1fbf6"),
        "retake": ("#4a4a4a", "#f4f4f4"),
    }
    accent, bg = colors.get(overall, ("#333", "#fff"))
    title = {
        "red": "RED — same-day access review",
        "amber": "AMBER — nurse examination / repeat capture",
        "green": "GREEN — no high-risk pattern detected",
        "retake": "RETAKE — capture quality guardrail",
    }.get(overall, overall.upper())

    def esc(s: Any) -> str:
        return html.escape(str(s))

    rows = []
    for f in tri["findings"]:
        p = f["probability"]
        bars = int(round(p * 100))
        rows.append(
            f"""
            <tr>
              <td><strong>{esc(f['label'])}</strong><br><span class="muted">{esc(f['finding'])}</span></td>
              <td class="prob">
                <div class="bar"><span style="width:{bars}%"></span></div>
                <span>{p*100:.1f}%</span>
              </td>
              <td>{f['red_threshold']*100:.1f}%</td>
              <td>{f['amber_threshold']*100:.1f}%</td>
              <td><span class="pill {f['status']}">{f['status'].upper()}</span></td>
            </tr>"""
        )
    if not rows:
        rows.append('<tr><td colspan="5" class="muted">No model findings because capture was rejected.</td></tr>')

    back_html = ""
    if back_url:
        back_html = f'<a href="{html.escape(back_url)}" style="display:inline-block;margin-bottom:12px;color:#0f766e;font-weight:700;font-size:13px;text-decoration:none">← New capture</a>'

    issues = []
    for view in ("down", "up"):
        q = result["quality"][view]
        for issue in q["issues"]:
            issues.append(f"<li><strong>{view.upper()}:</strong> {esc(issue)}</li>")
    issues_html = f"<ul class='issues'>{''.join(issues)}</ul>" if issues else "<p class='muted'>Both captures passed the quality guardrail.</p>"

    imgs = result.get("images", {})
    spec_html = ""
    if result.get("spectrogram"):
        spec_html = f"""
        <div class="card">
          <h2>Audio evidence</h2>
          <img class="spec" src="{result['spectrogram']}" alt="log-Mel spectrogram">
          <p class="muted">10-second bruit → 128 × 224 log-Mel "photo". Audio analysed: {result['has_audio']}.</p>
        </div>"""

    prob_rows = ""

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AVF Sentinel report</title>
<style>
  :root {{ --accent: {accent}; --bg: {bg}; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f6f7f9; color: #1d1d1f; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}
  .banner {{ background: var(--accent); color: white; border-radius: 14px; padding: 18px 22px; }}
  .banner h1 {{ margin: 0 0 6px; font-size: 22px; }}
  .banner p {{ margin: 0; opacity: .95; line-height: 1.45; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }}
  .card {{ background: white; border: 1px solid #e4e7ec; border-radius: 14px; padding: 18px; }}
  .card h2 {{ margin: 0 0 12px; font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: #5b6472; }}
  .photos {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  .photos figure {{ margin: 0; }}
  .photos img {{ width: 100%; border-radius: 8px; display: block; }}
  .photos figcaption {{ font-size: 11px; color: #6b7280; margin-top: 4px; }}
  .spec {{ width: 100%; border-radius: 8px; image-rendering: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 9px 8px; border-bottom: 1px solid #eef0f3; vertical-align: middle; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }}
  .bar {{ height: 8px; background: #eef0f3; border-radius: 99px; overflow: hidden; margin-bottom: 3px; }}
  .bar span {{ display: block; height: 100%; background: var(--accent); border-radius: 99px; }}
  .prob {{ min-width: 110px; font-variant-numeric: tabular-nums; }}
  .pill {{ display: inline-block; font-size: 10px; font-weight: 700; padding: 3px 7px; border-radius: 99px; letter-spacing: .04em; }}
  .pill.red {{ background: #ffe3e3; color: #a4161a; }}
  .pill.amber {{ background: #fff1cc; color: #8a5200; }}
  .pill.green {{ background: #d9f5e8; color: #1b6b4a; }}
  .muted {{ color: #6b7280; font-size: 12px; }}
  .issues li {{ margin: 4px 0; font-size: 13px; }}
  .disclaimer {{ margin-top: 18px; background: #1f2937; color: #f3f4f6; border-radius: 12px; padding: 16px 18px; font-size: 12px; line-height: 1.55; }}
  .foot {{ margin-top: 12px; font-size: 11px; color: #6b7280; }}
  @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} .photos {{ grid-template-columns: 1fr 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  {back_html}
  <div class="banner">
    <h1>{esc(title)}</h1>
    <p>{esc(tri['action'])}</p>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Paired capture</h2>
      <div class="photos">
        <figure><img src="{imgs.get('down_raw','')}" alt="raw arm down"><figcaption>Raw — arm down</figcaption></figure>
        <figure><img src="{imgs.get('down','')}" alt="processed arm down"><figcaption>ROI / WB / CLAHE</figcaption></figure>
        <figure><img src="{imgs.get('up_raw','')}" alt="raw arm raised"><figcaption>Raw — arm raised</figcaption></figure>
        <figure><img src="{imgs.get('up','')}" alt="processed arm raised"><figcaption>ROI / WB / CLAHE</figcaption></figure>
      </div>
    </div>
    <div class="card">
      <h2>Capture quality</h2>
      {issues_html}
      <p class="muted">White balance: {esc(result['quality'].get('wb_down','—'))} / {esc(result['quality'].get('wb_up','—'))}. Blur, exposure, angle and skin coverage are checked before inference.</p>
    </div>
  </div>

  <div class="card" style="margin-top:18px">
    <h2>Findings — calibrated probabilities</h2>
    <table>
      <thead><tr><th>Finding</th><th>Probability</th><th>Red thr</th><th>Amber thr</th><th>Status</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="muted">Thresholds were selected on a validation split for high sensitivity. Probabilities are research outputs, not diagnoses. {'' if result.get('has_audio') else 'No audio was supplied, so the acoustic branch was masked.'}</p>
  </div>

  {spec_html}

  <div class="disclaimer">
    <strong>Not a medical device. Research prototype only.</strong><br>
    {esc(DISCLAIMER)}<br><br>
    This report is generated from a synthetic-data demo model. Do not use it to rule out thrombosis or stenosis. A normal result does not exclude access failure.
  </div>
  <div class="foot">Generated {esc(result['generated_at'])} · checkpoint {esc(result.get('checkpoint',''))}</div>
</div>
</body>
</html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def _demo_case(data_dir: str | Path, scenario: str | None = None) -> tuple[np.ndarray, np.ndarray, Path, dict]:
    import pandas as pd

    df = pd.read_csv(Path(data_dir) / "features.csv")
    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] == 1]
    if scenario:
        sub = df[df["scenario"] == scenario]
        if len(sub):
            df = sub
    # Prefer a red-ish high-risk case for the demo if none specified.
    row = df.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    pre = Preprocessor()
    down = pre.read(Path(data_dir) / row["raw_image_down"])
    up = pre.read(Path(data_dir) / row["raw_image_up"])
    return down, up, Path(data_dir) / row["raw_audio"], row.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AVF Sentinel on one paired capture.")
    parser.add_argument("--checkpoint", default="artifacts/avf-demo/best.pt")
    parser.add_argument("--down", help="arm-down photo")
    parser.add_argument("--up", help="arm-raised photo")
    parser.add_argument("--audio", help="10 s bruit WAV")
    parser.add_argument("--out", default="artifacts/reports")
    parser.add_argument("--report", default="")
    parser.add_argument("--json", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="run even if the quality guardrail rejects")
    parser.add_argument("--demo", action="store_true", help="sample a case from --data")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--data", default="data/synthetic")
    args = parser.parse_args()

    if args.demo:
        down, up, audio, row = _demo_case(args.data, args.scenario or None)
        print(f"Demo case: {row['patient_id']} scenario={row['scenario']} labels=" + ", ".join(c for c in CLASSES if int(row[c]) == 1))
    else:
        if not args.down or not args.up:
            raise SystemExit("Provide --down and --up, or use --demo.")
        pre = Preprocessor()
        down, up = pre.read(args.down), pre.read(args.up)
        audio = Path(args.audio) if args.audio else None
        row = {"patient_id": "input"}

    predictor = Predictor(args.checkpoint, device=args.device)
    predictor.checkpoint_path = str(args.checkpoint)
    result = predictor.predict(down, up, audio, force=args.force)
    result["case"] = {k: row.get(k) for k in ("patient_id", "scenario", "skin_type", "quality_issue") if k in row}
    result["inputs"] = {
        "down": args.down or (str(row.get("raw_image_down", "")) if args.demo else ""),
        "up": args.up or (str(row.get("raw_image_up", "")) if args.demo else ""),
        "audio": str(audio) if audio else None,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report) if args.report else Path(args.out) / f"report_{stamp}.html"
    render_report(result, report_path)
    json_path = Path(args.json) if args.json else report_path.with_suffix(".json")
    json_result = {k: v for k, v in result.items() if k not in ("images", "spectrogram")}
    json_path.write_text(json.dumps(json_result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("capture_ok", "has_audio", "probabilities", "triage")}, indent=2))
    print(f"Report: {report_path}")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
