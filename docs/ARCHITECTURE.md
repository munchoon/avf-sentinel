# Architecture

AVF Sentinel is a **two-stream, late-fusion, multi-label** model. The design goal is
not to diagnose from one signal but to combine three complementary views of an
arteriovenous fistula (AVF):

1. **Structure** — paired arm-down / arm-raised photos.
2. **Flow** — a 10-second phone-microphone bruit recording.
3. **Capture validity** — explicit quality guardrails that can refuse to score.

```
                    ┌──────────────────────────────┐
 arm-down photo ───►│  shared ConvNeXt-Tiny        │──┐
                    │  (ImageNet pretrained)       │  │  [f_down, f_up, f_up−f_down, f_down⊙f_up]
 arm-raised photo ─►│  same weights, both views    │──┘        │
                    └──────────────────────────────┘           ▼
                                                        LayerNorm + Linear → 512
                                                                 │
 10 s bruit ──► log-Mel (128×224, 2 ch) ──► ResNet18 ──► 256 ────┤
             └─ 16 acoustic descriptors ──► MLP ──► 32 ──────────┤
                                                                │
 12 paired-photo descriptors ──► MLP ──► 32 ────────────────────┤
                                                                │
 modality-presence flags (arm, audio) ──► 2 ────────────────────┤
                                                                ▼
                                              LayerNorm → 834 → 384 → 192 → 5
                                                                │
                                              sigmoid per finding + temperature
                                                                │
                                              red / amber / green / retake triage
```

## 1. Capture and guardrails

`Preprocessor.process_pair()` analyses both photos jointly:

| Check | Feature | Default reject |
|---|---|---|
| Blur | Laplacian variance on 512 px grayscale | `< 4.0` |
| Dark | mean luma | `< 38` |
| Overexposed | mean luma | `> 222` |
| Clipping/glare | fraction of pixels at 0 or 255 | `> 0.22` |
| Tilt | PCA forearm axis + Hough edge fallback | `> 45°` |
| Arm not visible | skin-mask fraction | `< 0.08` |

Warnings start at 35° tilt. A hard failure produces a **retake** result with no
model score. This matters clinically: a confident-looking score on a blurry or
tilted photo is worse than no score.

The two photos in a pair share one crop box and one white-balance estimate. This
keeps the arm-elevation comparison valid; per-image auto-white-balance would
erase the colour/width change that the elevation test depends on.

## 2. Image branch

- Backbone: `torchvision` **ConvNeXt-Tiny**, ImageNet-1k weights, classifier
  removed (768-d pooled features).
- Weights are **shared** between the down and raised views, so the model cannot
  cheat by learning a view-specific shortcut.
- The pair representation is
  `[f_down, f_up, f_up − f_down, f_down ⊙ f_up]` (3072-d), projected to 512-d
  with LayerNorm, GELU, dropout.
- The difference and product terms make the elevation change explicit rather
  than forcing the fusion MLP to reconstruct it.

Alternative arm backbones: `resnet18`, `efficientnet_b0`, `vit_b_16`, and
optionally `biomedclip` (`open_clip_torch`).

### Handcrafted paired descriptors

Twelve auditable features are computed from segmentation and colour statistics:

```
width_ratio_up_down, area_ratio_up_down,
access_width_down_norm, access_width_up_norm,
bulge_ratio_up_down,
redness_down, redness_delta_up_minus_down,
vein_contrast_down, vein_contrast_ratio_up_down,
dark_frac_down, dark_frac_delta, luma_delta_up_minus_down
```

These give the fusion head explicit cues for collapse on elevation, focal bulge,
erythema, and bruising. They are standardised with training-split statistics and
clipped to ±8 before use.

## 3. Audio branch

- 16 kHz mono, 10 s, 1024-point Hann STFT, 512 hop, 128 Mel bands (40–7600 Hz).
- The STFT is scaled by `n_fft/2` so the log-Mel values are approximately dBFS.
- Two input channels:
  1. absolute dB mapped from `[-90, 0]` to `[-1, 1]`;
  2. local contrast against a Gaussian-smoothed spectrogram, clipped to ±1.
- Backbone: **ResNet18**, first convolution adapted from 3 to 2 channels by
  averaging ImageNet weights. Output 512 → 256.
- 16 handcrafted descriptors (spectral centroid/bandwidth/rolloff/flatness,
  high/low band ratio, pulse periodicity, envelope statistics, etc.) → 32.
- Spectrogram augmentation masks frequency and time bands during training.

The acoustic branch is deliberately **not** asked to diagnose from a single
number. It provides a turbulence/periodicity signal that is fused with the
visual and paired-photo evidence.

## 4. Modality dropout and missing audio

Audio is optional in the demo. During training, audio is randomly zeroed with
probability 0.08 (and the arm branch with 0.04), and an explicit audio-presence
flag is passed to fusion. At inference, if no audio is supplied, the acoustic
embedding is zeroed and the flag is set to 0. This prevents "missing audio" from
being silently interpreted as "quiet, therefore thrombosed".

## 5. Multi-label objective

Findings can co-occur (for example, stenosis + aneurysm, or infection +
hematoma). The model therefore uses five independent sigmoid outputs, not a
softmax.

- **Focal BCE** with `gamma=2.0`, `alpha=0.65`.
- Per-class `pos_weight = sqrt(negatives / positives)`, clipped to `[1, 4]`.
- This up-weights dangerous positives without letting a very rare class
  dominate the gradient.

## 6. Training recipe

| Setting | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 12 |
| Head LR | 3e-4 |
| Backbone LR | 2e-5 |
| Weight decay | 0.03 |
| Optimiser | AdamW |
| Schedule | cosine, 14 epochs |
| Frozen backbone | first 2 epochs |
| Early stopping | val macro AUROC, patience 5 |
| Augmentation | shared crop/flip/rotation, paired colour jitter, spectrogram masks |
| Device | auto (CUDA → MPS → CPU) |

Splits are **patient-grouped** and stratified by scenario and skin-tone group.
The training script refuses to run if a `patient_id` appears in more than one
split.

## 7. Calibration, thresholds, and triage

1. Fit one global temperature on the validation split (multi-label BCE).
2. For each finding select a threshold on validation:
   - nominal sensitivity targets: thrombosis **95%**, others **90%**;
   - a **+0.05 sensitivity margin** is requested to leave slack on a small
     validation split;
   - the **lowest** threshold meeting that target while keeping validation
     specificity ≥ **0.80** is chosen. If both constraints cannot be met, the
     maximum-specificity threshold at the target is used as a fallback.
   This is intentionally different from "maximise specificity at the target":
   with a tiny validation set, that rule picks a threshold sitting exactly on
   the lowest positive score and generalises poorly.
3. Freeze thresholds, then evaluate once on the held-out test split.
4. Amber threshold targets an additional ~5 sensitivity points (capped at 99%),
   and is forced below the red threshold.

Triage is intentionally simple and safety-first:

- **red** if any finding ≥ its red threshold;
- **amber** if any finding ≥ its amber threshold;
- **green** otherwise;
- **retake** if the guardrail fails (model not run).

A green result is **not** a clearance. Every report says so.

## 8. Evaluation

- Per-finding AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, Brier, ECE.
- Patient-level bootstrap 95% CIs (500 resamples).
- Skin-tone subgroup audit (`I–III` vs `IV–VI`), with explicit small-n caveat.
- Guardrail audit: rejection rate, per-issue catch rate, false rejections on
  clean captures.
- Artifacts: `metrics.json`, `thresholds.json`, `predictions_test.csv`,
  `MODEL_CARD.md`, training/ROC/PR/reliability/confusion/triage plots,
  preprocessing and spectrogram contact sheets.

All demo numbers come from a **synthetic cohort** and are software tests, not
clinical evidence. See `docs/REAL_DATA.md` for the path to real data.

## 9. Production gaps

This prototype does not yet include:

- prospective or external validation;
- device/site calibration;
- de-identification and audit logging;
- regulatory clearance (CE/FDA);
- integration with dialysis-unit workflow or EHR;
- monitoring for drift, subgroup harm, or alert fatigue;
- a security review for handling patient photos and audio.

Those are prerequisites for any clinical deployment, not optional extras.
