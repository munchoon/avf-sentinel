# Real data: from prototype to a credible study

> **There is no public, labelled, patient-level dataset of paired AV-fistula arm
> photos, bruit audio, and duplex ground truth.** This prototype ships with a
> synthetic cohort so the pipeline is runnable and testable. The instructions
> below are the path to replacing it with real, consented data.

## 1. What "real data" must contain

One row per **capture session** (a patient can contribute multiple sessions, but
splits must stay patient-grouped):

| Column | Required | Example | Notes |
|---|---|---|---|
| `patient_id` | yes | `AVF-0042` | Stable pseudonymous ID; never the MRN |
| `session_id` | recommended | `AVF-0042-2026-03-11` | Repeat captures over time |
| `image_down` | yes | `images/AVF-0042_down.jpg` | Arm at heart level, access arm, whole forearm |
| `image_up` | yes | `images/AVF-0042_up.jpg` | Same arm raised above heart for 30 s |
| `audio` | yes* | `audio/AVF-0042.wav` | 10 s, 16 kHz+, phone mic 1–2 cm from anastomosis |
| `stenosis` | yes | `0/1` | Duplex: PSV ratio ≥ 2.5 or ≥ 50% diameter reduction |
| `thrombosis` | yes | `0/1` | Occlusion / absent thrill-bruit + ultrasound confirmation |
| `aneurysm` | yes | `0/1` | Focal dilatation, clinically significant |
| `infection` | yes | `0/1` | Clinical diagnosis (erythema, warmth, drainage, fever) |
| `hematoma` | yes | `0/1` | Clinically significant access-site hematoma |
| `split` | yes | `train/val/test` | Patient-grouped; site-held-out for external test |
| `skin_type` | recommended | `1–6` | Fitzpatrick, for equity audit only |
| `scenario` | optional | `stenosis` | Free-text stratum for sampling |
| `quality_issue` | optional | `none` | Known capture problem, for guardrail audit |
| `collapse_ratio` | optional | `0.18` | Elevation-test measurement, if adjudicated |
| `doppler_stenosis_pct` | optional | `65` | Reference standard detail |
| `site_id` | recommended | `BWH-01` | For site-held-out validation |

\* Audio may be missing in real deployments. The model supports missing audio,
but a missing-audio subgroup must be analysed separately.

### Label definitions (pre-register these)

- **Stenosis**: duplex peak systolic velocity ratio ≥ 2.5 across the lesion, or
  ≥ 50% diameter reduction on angiography/duplex, plus a clinical decision.
  Record the reference modality and reader.
- **Thrombosis**: absent or barely audible bruit/thrill with ultrasound
  non-compressibility/occlusion; do not label "suspected" cases as positive
  without imaging or adjudication.
- **Aneurysm**: focal access-site dilatation above a pre-specified diameter or
  ratio threshold; record the threshold.
- **Infection**: clinician diagnosis with signs of erythema, warmth, drainage,
  or fever; cultures when available. The photo is a proxy, not the gold standard.
- **Hematoma**: clinically significant bruising/swelling, ideally with size or
  ultrasound confirmation.

Have **two blinded adjudicators** label a sample and report inter-rater
agreement. If agreement is poor, fix the definition before training.

## 2. Capture protocol

### Photos

1. Same phone for both images; flash off; HDR off if possible.
2. Arm **down** at heart level, palm up, whole forearm and access site visible.
3. Raise the arm above heart level for **30 seconds**, then capture **up** with
   the same framing.
4. Avoid direct overhead light; avoid strong coloured backgrounds; keep the
   access site unobstructed.
5. Take both images within one minute. The app should reject blur, tilt > 35–45°,
   dark/bright exposure, and missing arm.

### Audio

1. Quiet room; phone mic 1–2 cm from the anastomosis, not touching the skin.
2. Record **10 seconds** at 16 kHz or higher, mono if possible.
3. Note concurrent dialysis, hypotension, or needle position.
4. Label absent/faint/present bruit and any other clinical context.

### Metadata

Record phone model, OS, app version, site, lighting, Fitzpatrick type,
age/sex (for fairness analysis only), access type (radiocephalic,
brachiocephalic, brachiobasilic, graft), and access age.

## 3. Splits and sample size

- **Patient-grouped**: never split images/audio from one patient across sets.
- **Time-aware**: if data span months, prefer a temporal holdout (train on
  earlier, test on later) to detect drift.
- **Site-held-out**: the strongest test is train on sites A/B, test on site C.
- **Stratify** by scenario, skin-tone group, access type, and site.

For a hackathon-style pilot, aim for at least **100–300 patients** to exercise
the pipeline. For a credible clinical study, plan for **thousands of captures**
across sites, with pre-registered sensitivity/specificity targets and confidence
intervals. Do not power a study from AUROC alone; use the operating point you
will actually deploy.

## 4. Import into AVF Sentinel

```bash
# 1. Put files under data/real/
#    data/real/images/*.jpg
#    data/real/audio/*.wav
# 2. Build manifest.csv with the columns above. If you need patient-grouped folds:
python scripts/grouped_folds.py --manifest data/real/manifest.csv \
  --out data/real/manifest.csv --folds 5 --write-splits
# 3. Preprocess and cache
PYTHONPATH=src python -m avfsentinel.prepare --data data/real
# 4. Train with a real backbone and more epochs
PYTHONPATH=src python -m avfsentinel.train --data data/real \
  --out artifacts/real-v1 --epochs 30 --batch-size 12 --target-sens 0.90
# 5. Evaluate the frozen test split, then read MODEL_CARD.md
```

Before trusting any number:

- confirm `features.csv` row counts and label prevalences;
- inspect `quality_summary.json` and the preprocessing contact sheet;
- check `predictions_test.csv` for patient-level errors;
- read the subgroup table and confidence intervals;
- have a clinician review the false negatives, not just the AUROC.

## 5. Open-access images are not a training set

Open-i / PubMed Central figures can help you sanity-check code and build a
**qualitative** demo of preprocessing. They are not a labelled, paired,
patient-level cohort, and they must not be used to claim clinical performance.
`scripts/fetch_openi.py` downloads figures for visual inspection only; it refuses
to create labels. The Open-i API sometimes returns an HTML maintenance page
(observed in November 2026); the script detects that and exits with a clear
error rather than writing HTML as if it were an image.

## 6. Governance

- Written consent or a waiver approved by an IRB/ethics board.
- De-identify images and audio; strip EXIF; avoid faces, tattoos, jewellery, and
  hospital identifiers.
- Encrypt at rest and in transit; restrict access; log every inference.
- Define a data-retention and deletion policy.
- Report subgroup performance and harms, including false positives and alert
  fatigue.
- Obtain regulatory advice before any clinical use. In the US, software that
  informs diagnosis/treatment may be a medical device; in the EU, check MDR
  classification.
