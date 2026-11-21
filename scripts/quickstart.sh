#!/usr/bin/env bash
# End-to-end demo: generate synthetic cohort -> preprocess -> train -> report.
# Usage:
#   ./scripts/quickstart.sh                 # 300 patients, 14 epochs
#   PATIENTS=120 EPOCHS=2 ./scripts/quickstart.sh
#   SKIP_TRAIN=1 ./scripts/quickstart.sh    # data + preprocessing only
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
PATIENTS="${PATIENTS:-300}"
EPOCHS="${EPOCHS:-14}"
BATCH_SIZE="${BATCH_SIZE:-12}"
DATA="${DATA:-data/synthetic}"
OUT="${OUT:-artifacts/avf-demo}"
SEED="${SEED:-7}"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "==> Checking dependencies"
"$PYTHON" - <<'PY'
import importlib
missing = []
for name in ("torch", "torchvision", "cv2", "numpy", "pandas", "scipy", "sklearn", "fastapi"):
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing: " + ", ".join(missing) +
        "\nInstall with: pip install -r requirements.txt"
    )
print("dependencies ok")
PY

if [ ! -f "$DATA/manifest.csv" ]; then
  echo "==> Generating synthetic cohort ($PATIENTS patients) in $DATA"
  "$PYTHON" -m avfsentinel.synth --out "$DATA" --patients "$PATIENTS" --seed "$SEED"
else
  echo "==> Reusing existing manifest: $DATA/manifest.csv"
fi

echo "==> Preprocessing (guardrails, ROI, white balance, log-Mel)"
"$PYTHON" -m avfsentinel.prepare --data "$DATA"

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "==> SKIP_TRAIN=1: stopping before training"
  exit 0
fi

echo "==> Training ($EPOCHS epochs) -> $OUT"
"$PYTHON" -m avfsentinel.train --data "$DATA" --out "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --seed "$SEED"

echo "==> Generating a demo report"
"$PYTHON" -m avfsentinel.infer --checkpoint "$OUT/best.pt" --demo --scenario thrombosis \
  --data "$DATA" --out "$OUT" --report "$OUT/demo_report.html"

echo
echo "Done."
echo "  Model card : $OUT/MODEL_CARD.md"
echo "  Metrics    : $OUT/metrics.json"
echo "  Report     : $OUT/demo_report.html"
echo "  Demo UI    : PYTHONPATH=src $PYTHON -m avfsentinel.serve --checkpoint $OUT/best.pt"
