import { useState } from "react";

export type CareStatus = "red" | "amber" | "green" | "retake" | "unavailable";

type Symptom =
  | "pain"
  | "swelling"
  | "dialysis_concern"
  | "severe_pain"
  | "access_bleeding"
  | "hand_cold_numb"
  | "fever_or_very_unwell";

type AdvicePlan = {
  destination: string;
  title: string;
  message: string;
  timing: string;
  urgency: "urgent" | "prompt" | "routine" | "fallback";
};

type PatientCareAdviceProps = {
  status: CareStatus;
  finding: string;
  onBackToCapture: () => void;
};

const symptomOptions: Array<{ value: Symptom; label: string; urgent: boolean }> = [
  { value: "pain", label: "Pain around the access", urgent: false },
  { value: "swelling", label: "Swelling around the access", urgent: false },
  { value: "dialysis_concern", label: "Recent dialysis concern", urgent: false },
  { value: "severe_pain", label: "Severe pain", urgent: true },
  { value: "access_bleeding", label: "Bleeding that needs urgent attention", urgent: true },
  { value: "hand_cold_numb", label: "Hand is cold, pale, blue, or numb", urgent: true },
  { value: "fever_or_very_unwell", label: "Fever or feeling very unwell", urgent: true },
];

const adviceByStatus: Record<CareStatus, AdvicePlan> = {
  red: {
    destination: "Contact your dialysis center today",
    title: "Same-day access review",
    message:
      "The demo scan showed a pattern that needs a qualified clinician to assess. This is not a diagnosis.",
    timing: "Contact your dialysis care team now to arrange same-day review.",
    urgency: "urgent",
  },
  amber: {
    destination: "Contact your dialysis center",
    title: "Nurse examination recommended",
    message:
      "The demo scan is uncertain and should be reviewed by your dialysis care team.",
    timing:
      "Arrange an examination at your next dialysis session. Call sooner if symptoms or concerns persist.",
    urgency: "prompt",
  },
  green: {
    destination: "Continue routine dialysis-center care",
    title: "Routine surveillance",
    message:
      "This demo result does not rule out stenosis or thrombosis and is not a clearance.",
    timing:
      "Continue routine care. Contact your dialysis team if you have symptoms or concerns.",
    urgency: "routine",
  },
  retake: {
    destination: "No destination selected — no scan result",
    title: "Capture needs to be repeated",
    message:
      "The image quality check rejected the capture. A retake is not a normal result.",
    timing:
      "Follow the capture instructions and retry only if appropriate. Contact your care team if you have symptoms or concerns.",
    urgency: "fallback",
  },
  unavailable: {
    destination: "Use your clinic’s usual care pathway",
    title: "Unable to assess safely",
    message:
      "The demo assessment is unavailable. No model result or model-based advice was produced.",
    timing:
      "Do not treat this as a normal result. Contact your dialysis care team for guidance.",
    urgency: "fallback",
  },
};

export default function PatientCareAdvice({
  status,
  finding,
  onBackToCapture,
}: PatientCareAdviceProps) {
  const [selectedSymptoms, setSelectedSymptoms] = useState<Symptom[]>([]);
  const urgentSymptomSelected = symptomOptions.some(
    (option) => option.urgent && selectedSymptoms.includes(option.value),
  );
  const otherSymptomSelected = selectedSymptoms.some(
    (symptom) =>
      symptom === "pain" || symptom === "swelling" || symptom === "dialysis_concern",
  );

  const advice: AdvicePlan = urgentSymptomSelected
    ? {
        destination: "Seek immediate clinical help",
        title: "Urgent symptoms need prompt attention",
        message:
          "You selected a symptom that should not wait for an app result. Do not rely on this demo scan.",
        timing:
          "Use your clinic’s urgent pathway or local emergency services now. Follow the care team’s instructions.",
        urgency: "urgent",
      }
    : otherSymptomSelected
      ? {
          destination: "Contact your dialysis center promptly",
          title: "Symptoms need clinical review",
          message:
            "Symptoms or dialysis concerns should be reviewed even if a demo result appears green.",
          timing: "Contact your dialysis care team promptly for guidance.",
          urgency: "prompt",
        }
      : adviceByStatus[status];

  function toggleSymptom(symptom: Symptom) {
    setSelectedSymptoms((current) =>
      current.includes(symptom)
        ? current.filter((item) => item !== symptom)
        : [...current, symptom],
    );
  }

  return (
    <section className="care-page" aria-live="polite">
      <div className={`care-banner urgency-${advice.urgency}`}>
        <p className="eyebrow">PATIENT CARE GUIDANCE · DEMO</p>
        <h2>{advice.title}</h2>
        <p>{advice.message}</p>
      </div>

      <div className="care-grid">
        <section className="panel destination-panel">
          <div className="panel-heading">
            <span className="step-number">A</span>
            <h2>Where to go</h2>
          </div>
          <h3>{advice.destination}</h3>
          <p>{advice.timing}</p>
          {advice.urgency === "urgent" && (
            <p className="emergency-note">
              If you feel acutely unwell or symptoms are severe, seek immediate
              clinical care rather than waiting for the next dialysis session.
            </p>
          )}
        </section>

        <section className="panel">
          <div className="panel-heading">
            <span className="step-number">B</span>
            <h2>Symptoms check</h2>
          </div>
          <p className="hint">
            Select any symptoms. They can change the suggested urgency regardless
            of the demo scan result.
          </p>
          <div className="symptom-list">
            {symptomOptions.map((option) => (
              <label className="symptom-option" key={option.value}>
                <input
                  type="checkbox"
                  checked={selectedSymptoms.includes(option.value)}
                  onChange={() => toggleSymptom(option.value)}
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <span className="step-number">C</span>
            <h2>Scan summary</h2>
          </div>
          <p><strong>Demo finding:</strong> {finding}</p>
          <p><strong>Selected scan status:</strong> {status.toUpperCase()}</p>
          <p className="hint">
            The scan status is simulated in this frontend prototype and must not
            be used to diagnose or rule out an AV fistula problem.
          </p>
        </section>

        <section className="panel alert-preview-panel">
          <div className="panel-heading">
            <span className="step-number">D</span>
            <h2>Care-team notification preview</h2>
          </div>
          <p>
            {status === "red" || status === "amber"
              ? "A dialysis-center review notification would be prepared for clinician review."
              : "No urgent dialysis-center notification is prepared by this demo result."}
          </p>
          <span className="simulation-badge">PREVIEW ONLY · NOT SENT</span>
        </section>
      </div>

      <div className="care-disclaimer">
        <strong>Research prototype only.</strong> This page provides fixed demo
        guidance, not a medical diagnosis. It does not contact a dialysis center
        or hospital. Follow your clinic’s instructions and seek care for symptoms
        regardless of the app result.
      </div>

      <button className="secondary-button back-button" type="button" onClick={onBackToCapture}>
        Back to image capture
      </button>
    </section>
  );
}
