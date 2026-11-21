# Clinical protocol (research use only)

> **This is a research prototype, not a medical device.** It is not validated for
> clinical use. Nothing here authorises autonomous diagnosis or withholding of
> standard care. Every result requires clinician review, and duplex ultrasound
> remains the reference standard.

## 1. Purpose and setting

AVF Sentinel is a **triage-support** tool for adults with a haemodialysis
arteriovenous fistula (AVF) or arteriovenous graft (AVG). It is designed for
dialysis-unit use, with a nurse or trained technician capturing two photos and a
10-second bruit clip. It is not a home self-diagnosis app.

The prototype answers one question: *does this capture contain a high-risk
pattern that should prompt same-day vascular-access review?* It does **not**
answer "is this access patent?" and it must never be used to rule out
thrombosis or stenosis.

## 2. Eligibility

**Include** adults (≥ 18 years) with an AVF/AVG who can provide consent (or have
a legally authorised representative) and can tolerate a 30-second arm raise.

**Exclude or defer** if:

- the access arm cannot be photographed (bandage, active bleeding, severe pain);
- the patient is acutely unwell (sepsis, symptomatic hypotension) — treat first;
- the capture cannot meet the quality guardrail after two attempts;
- the patient declines audio recording;
- the access site is obscured by a dressing that cannot be removed safely.

Do not delay urgent clinical care to obtain a capture.

## 3. Capture workflow

### Step 1 — Prepare

- Explain the purpose and the limits: "This is a research triage aid. It cannot
  rule out a problem."
- Confirm the access arm and identify the anastomosis and cannulation sites.
- Clean the phone camera lens; use the same phone for both photos.
- Turn off flash and HDR. Use a plain background if possible.

### Step 2 — Arm-down photo

- Arm at heart level, palm up, access site and whole forearm in frame.
- The forearm should be roughly vertical in the image.
- Hold still; tap to focus on the access site.
- Check the preview for blur, glare, and tilt.

### Step 3 — Arm-raised photo

- Raise the arm above heart level for **30 seconds**.
- Keep the same distance, angle, and background.
- Capture the same field of view, then lower the arm.

### Step 4 — Bruit audio

- Quiet room. Phone mic **1–2 cm** from the anastomosis, not touching skin.
- Record **10 seconds**; ask the patient to stay still and breathe normally.
- Note whether the bruit is absent, faint, or present.
- If dialysis is running, note the needle position and pump speed.

### Step 5 — Guardrail

The app checks blur, exposure, clipping, tilt, and arm visibility. If it asks
for a retake:

- blur → brace the phone, tap to focus, hold still;
- dark → move to brighter, diffuse light;
- overexposed/glare → change the angle, avoid direct light;
- tilted → keep the forearm vertical and the phone level;
- arm not visible → fill the frame with the access arm.

**Maximum two retakes.** If quality still fails, stop and document why. Do not
force a score.

## 4. Result handling

### RED — same-day vascular-access review

- Check thrill and bruit; examine for swelling, erythema, and tenderness.
- Arrange duplex ultrasound and same-day/next-day vascular-access review.
- **Do not attempt routine cannulation** of a suspected thrombosed access.
- Escalate immediately for signs of ischaemia, severe pain, bleeding, or sepsis.

### AMBER — nurse examination / repeat capture

- Examine the access at the next session.
- Repeat the capture if the patient is stable and the quality was borderline.
- Escalate to duplex if clinically concerned, or if amber persists.
- Amber is not a clearance and not a diagnosis.

### GREEN — routine surveillance

- Continue routine access surveillance.
- A green result **does not exclude** stenosis or thrombosis.
- If the patient has symptoms (pain, swelling, bleeding, inadequate dialysis),
  escalate regardless of the app result.

### RETAKE — quality failure

- No model score was produced.
- Follow the retake guidance or stop and document.
- Do not interpret a retake as "normal".

## 5. Safety controls

- **Human-in-the-loop always.** No result is acted on without a clinician.
- **No autonomous rule-out.** The report states this on every page.
- **No silent failure.** Missing audio is flagged; a failed guardrail blocks
  scoring.
- **Sensitivity-first thresholds.** The system is tuned to miss fewer dangerous
  cases, accepting more false positives. Alert fatigue must be monitored.
- **Stop rule.** If the unit observes a harmful false negative or a workflow
  hazard, pause the study and report it.
- **Audit trail.** Log the capture, model version, scores, triage, reviewer, and
  clinical outcome (with consent and ethics approval).

## 6. Equity and human factors

- Recruit across Fitzpatrick skin types I–VI, access types, ages, and sites.
- Report sensitivity and false-positive rates by skin-tone group; do not pool
  them into a single number.
- Test under real clinic lighting, including fluorescent and window light.
- Provide plain-language materials in the patient's language.
- Measure capture time, retake rate, and nurse workload.
- Include patients with tremor, limited mobility, and low digital literacy in
  usability testing.

## 7. Data handling

- Obtain written consent or an approved waiver.
- De-identify images and audio; remove EXIF; crop out faces and identifiers.
- Store on encrypted, access-controlled infrastructure.
- Do not upload patient media to third-party services.
- Define retention, deletion, and breach procedures.
- Follow local law (HIPAA, GDPR, or equivalent) and institutional policy.

## 8. What would be required before clinical use

1. Prospective, multi-site validation against duplex ultrasound.
2. Pre-registered operating points and subgroup analyses.
3. External validation on a held-out site and device.
4. Regulatory classification and clearance (FDA/CE or local equivalent).
5. Clinical safety monitoring, drift detection, and post-market surveillance.
6. Integration testing with dialysis-unit workflow and EHR.
7. Health-economic and alert-fatigue evaluation.

Until then, this repository is a demonstration of the *pipeline and the safety
thinking*, not evidence that the model works on patients.
