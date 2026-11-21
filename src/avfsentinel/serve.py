"""FastAPI demo UI for AVF Sentinel.

    PYTHONPATH=src python -m avfsentinel.serve --checkpoint artifacts/avf-demo/best.pt
    open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from .constants import CLASS_LABELS, CLASSES, DISCLAIMER
from .infer import Predictor, _demo_case, render_report
from .utils import ensure_dir, load_json

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AVF Sentinel — research demo</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; background:#0f172a; color:#e5e7eb; }
  .wrap { max-width:1050px; margin:0 auto; padding:32px 22px 56px; }
  .kicker { color:#7dd3fc; font-size:12px; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }
  h1 { font-size:38px; margin:10px 0 8px; letter-spacing:-.03em; }
  .lede { color:#a8b3c4; max-width:760px; line-height:1.6; font-size:15px; }
  .badge { display:inline-block; background:#3b1d1d; color:#fca5a5; border:1px solid #7f1d1d; padding:5px 9px; border-radius:99px; font-size:11px; font-weight:700; letter-spacing:.05em; margin-top:12px; }
  .grid { display:grid; grid-template-columns:1.15fr .85fr; gap:20px; margin-top:28px; }
  .card { background:#172033; border:1px solid #263247; border-radius:16px; padding:20px; }
  .card h2 { font-size:14px; margin:0 0 14px; text-transform:uppercase; letter-spacing:.08em; color:#93a4bd; }
  label { display:block; font-size:12px; color:#a8b3c4; margin:12px 0 5px; }
  input[type=file] { width:100%; background:#0f172a; color:#dbe4f0; border:1px solid #2b3b55; border-radius:9px; padding:9px; font-size:12px; }
  input[type=file]::file-selector-button { background:#24344e; color:#dbe4f0; border:0; border-radius:6px; padding:6px 9px; margin-right:8px; cursor:pointer; }
  button { background:#2dd4bf; color:#06201d; border:0; border-radius:10px; padding:13px 17px; font-weight:800; font-size:14px; cursor:pointer; width:100%; margin-top:18px; }
  button:hover { background:#5eead4; }
  button.demo { width:auto; margin:6px 6px 0 0; padding:9px 12px; font-size:12px; background:#24344e; color:#dbe4f0; border:1px solid #34496a; }
  button.demo:hover { background:#2f4463; }
  .hint { color:#7e8da3; font-size:12px; line-height:1.55; margin-top:10px; }
  .stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px; }
  .stat { background:#0f172a; border:1px solid #263247; border-radius:10px; padding:10px; }
  .stat b { display:block; font-size:20px; color:#5eead4; }
  .stat span { font-size:11px; color:#8494aa; }
  .flow { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .step { font-size:11px; color:#9fb0c7; background:#0f172a; border:1px solid #263247; border-radius:8px; padding:7px 9px; }
  .foot { color:#64748b; font-size:11px; margin-top:24px; line-height:1.6; }
  .error { background:#3b1d1d; color:#fecaca; border:1px solid #7f1d1d; border-radius:10px; padding:12px; font-size:13px; margin-top:14px; }
  @media (max-width:800px) { .grid { grid-template-columns:1fr; } h1 { font-size:30px; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="kicker">MIT Hack Medicine · research prototype</div>
  <h1>AVF Sentinel</h1>
  <p class="lede">A two-stream vision model for suspected AV-fistula failure: paired arm-down / arm-raised photos for structural and elevation-test signals, plus a 10-second phone-mic bruit clip rendered as a log-Mel spectrogram. Built to demonstrate a sensitivity-first, human-in-the-loop triage workflow.</p>
  <div class="badge">NOT A MEDICAL DEVICE · SYNTHETIC DEMO WEIGHTS</div>

  <div class="grid">
    <div class="card">
      <h2>Run a capture</h2>
      <form id="form" action="/predict" method="post" enctype="multipart/form-data">
        <label>Arm DOWN photo (access arm, whole forearm)</label>
        <input type="file" name="down" accept="image/*" required>
        <label>Arm RAISED above heart for 30 s — photo</label>
        <input type="file" name="up" accept="image/*" required>
        <label>Bruit audio (10 s WAV; hold mic 1–2 cm from anastomosis)</label>
        <input type="file" name="audio" accept="audio/wav,audio/x-wav,audio/*">
        <button type="submit">Analyse capture</button>
        <p class="hint" id="status">The quality guardrail will ask for a retake if the pair is blurry, badly lit, tilted, or the arm is not visible. Audio is optional but recommended.</p>
      </form>
    </div>

    <div class="card">
      <h2>Demo cases</h2>
      <p class="hint" style="margin-top:0">Synthetic patients from the held-out test split. Each opens a full triage report.</p>
      <div>
        <button class="demo" onclick="demo('normal')">Normal</button>
        <button class="demo" onclick="demo('stenosis')">Stenosis</button>
        <button class="demo" onclick="demo('thrombosis')">Thrombosis</button>
        <button class="demo" onclick="demo('aneurysm')">Aneurysm</button>
        <button class="demo" onclick="demo('infection')">Infection</button>
      </div>
      <h2 style="margin-top:24px">Pipeline</h2>
      <div class="flow">
        <span class="step">1 · guardrail</span>
        <span class="step">2 · ROI + white balance</span>
        <span class="step">3 · paired CNN</span>
        <span class="step">4 · log-Mel ViT/CNN</span>
        <span class="step">5 · fusion + thresholds</span>
      </div>
      <h2 style="margin-top:24px">Demo metrics</h2>
      <div class="stats">__STATS__</div>
      <p class="hint">Synthetic-data software tests, not clinical evidence. Read the model card before citing any number.</p>
    </div>
  </div>

  <div class="foot">
    <strong>Safety:</strong> This demo cannot rule out thrombosis or stenosis. A green result is not a clearance. Every red/amber result requires physical examination and duplex ultrasound by a qualified clinician. No patient data is uploaded to a remote service; this server processes files locally.
    <br><br>__DISCLAIMER__
  </div>
</div>
<script>
  function demo(s) { window.location.href = '/?demo=' + encodeURIComponent(s); }
  document.getElementById('form').addEventListener('submit', function() {
    document.getElementById('status').textContent = 'Running guardrail → preprocessing → two-stream inference…';
  });
</script>
</body>
</html>"""


def build_index(metrics: dict | None, stats_override: str | None = None) -> str:
    stats = ""
    if stats_override is not None:
        stats = stats_override
    elif metrics:
        tm = metrics.get("test_metrics", {})
        stats = (
            f'<div class="stat"><b>{(tm.get("macro_auroc") or 0):.3f}</b><span>macro AUROC (synthetic test)</span></div>'
            f'<div class="stat"><b>{100 * (tm.get("macro_sensitivity") or 0):.1f}%</b><span>macro sensitivity</span></div>'
            f'<div class="stat"><b>{tm.get("triage", {}).get("red", 0)}</b><span>red alerts in test set</span></div>'
            f'<div class="stat"><b>{tm.get("n", 0)}</b><span>held-out patients</span></div>'
        )
    else:
        stats = '<p class="hint">Train a model to populate metrics.</p>'
    return INDEX_HTML.replace("__STATS__", stats).replace("__DISCLAIMER__", DISCLAIMER)


def create_app(checkpoint: str, data_dir: str = "data/synthetic", reports: str = "artifacts/reports") -> FastAPI:
    app = FastAPI(title="AVF Sentinel", version="0.1.0")
    report_dir = ensure_dir(reports)
    predictor = Predictor(checkpoint)
    metrics_path = Path(checkpoint).parent / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else None
    feats_path = Path(data_dir) / "features.csv"

    def _report_response(result: dict, name: str) -> HTMLResponse:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = report_dir / f"{name}_{stamp}.html"
        render_report(result, path, back_url="/")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        scenario = request.query_params.get("demo")
        if scenario:
            if not feats_path.exists():
                return HTMLResponse(build_index(metrics, '<div class="error">Demo dataset not found. Run the quickstart first.</div>'))
            df = pd.read_csv(feats_path)
            sub = df[(df["split"] == "test") & (df["scenario"] == scenario) & (df["quality_pass"] == 1)]
            if len(sub) == 0:
                sub = df[(df["scenario"] == scenario) & (df["quality_pass"] == 1)]
            if len(sub) == 0:
                return HTMLResponse(build_index(metrics, '<div class="error">No case for that scenario.</div>'))
            from .preprocessing import Preprocessor

            pre = Preprocessor()
            # Prefer a case whose triage matches the scenario: green for normal,
            # red for a findings scenario. Fall back to the first qualifying case.
            desired = "green" if scenario == "normal" else "red"
            chosen = None
            for _, row in sub.head(14).iterrows():
                down = pre.read(Path(data_dir) / row["raw_image_down"])
                up = pre.read(Path(data_dir) / row["raw_image_up"])
                result = await run_in_threadpool(
                    predictor.predict, down, up, Path(data_dir) / row["raw_audio"], True
                )
                if chosen is None:
                    chosen = (row, result)
                if result["triage"]["overall"] == desired:
                    chosen = (row, result)
                    break
            if chosen is None:
                return HTMLResponse(build_index(metrics, '<div class="error">No usable case for that scenario.</div>'))
            row, result = chosen
            result["case"] = {
                "patient_id": row["patient_id"],
                "scenario": row["scenario"],
                "skin_type": int(row["skin_type"]),
            }
            return _report_response(result, f"demo_{scenario}")
        return HTMLResponse(build_index(metrics))

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(down: UploadFile = File(...), up: UploadFile = File(...), audio: UploadFile | None = File(None)):
        try:
            down_bytes = await down.read()
            up_bytes = await up.read()
            d = cv2.imdecode(np.frombuffer(down_bytes, np.uint8), cv2.IMREAD_COLOR)
            u = cv2.imdecode(np.frombuffer(up_bytes, np.uint8), cv2.IMREAD_COLOR)
            if d is None or u is None:
                raise ValueError("Could not decode one of the images. Upload JPEG or PNG photos.")
            d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
            u = cv2.cvtColor(u, cv2.COLOR_BGR2RGB)
            audio_path = None
            if audio is not None and audio.filename:
                raw = await audio.read()
                suffix = Path(audio.filename).suffix.lower() or ".wav"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(raw)
                tmp.close()
                audio_path = tmp.name
            result = await run_in_threadpool(predictor.predict, d, u, audio_path, False)
            result["case"] = {"patient_id": "upload", "scenario": "", "skin_type": None}
            return _report_response(result, "upload")
        except Exception as exc:
            page = build_index(metrics, f'<div class="error">Could not analyse this capture: {str(exc)}</div>')
            return HTMLResponse(page, status_code=400)

    @app.get("/health")
    async def health():
        return {"status": "ok", "checkpoint": str(checkpoint), "device": str(predictor.device)}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the AVF Sentinel demo.")
    parser.add_argument("--checkpoint", default="artifacts/avf-demo/best.pt")
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--reports", default="artifacts/reports")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Run the quickstart first.")
    app = create_app(args.checkpoint, args.data, args.reports)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
