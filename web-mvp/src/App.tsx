import { useEffect, useState } from "react";
import DialysisCenterInbox, {
  type AlertWorkflowStatus,
  type CenterAlert,
} from "./DialysisCenterInbox";
import CenterAlertDetails from "./CenterAlertDetails";
import LoginPage from "./LoginPage";
import PatientProfile from "./PatientProfile";
import PatientCareAdvice, { type CareStatus } from "./PatientCareAdvice";
import type { DemoUser } from "./demoAuth";
import { createDemoWatchSnapshot } from "./demoWatchData";

type Scenario = "thrombosis" | "stenosis" | "normal" | "retake" | "unavailable";

type DemoResult = {
  status: CareStatus;
  title: string;
  summary: string;
  finding: string;
  adviceTitle: string;
  advice: string;
  centerAlert: string;
  patientAlert: string;
};

const DEMO_RESULTS: Record<Scenario, DemoResult> = {
  thrombosis: {
    status: "red",
    title: "RED — same-day access review",
    summary: "Simulated high-priority pattern. Clinician review is required.",
    finding: "Simulated access-flow concern",
    adviceTitle: "Contact the dialysis team today",
    advice:
      "Demo guidance only, not a diagnosis. A qualified clinician should assess the access. Seek immediate clinical help for severe pain, bleeding, signs of poor circulation, or feeling acutely unwell.",
    centerAlert:
      "DEMO ALERT: clinician review requested today. Confirm the capture and assess the patient.",
    patientAlert:
      "DEMO MESSAGE: please contact your dialysis care team today for advice.",
  },
  stenosis: {
    status: "amber",
    title: "AMBER — nurse examination",
    summary: "Simulated finding that needs clinical review.",
    finding: "Simulated access-flow concern",
    adviceTitle: "Arrange a nurse examination",
    advice:
      "Discuss this result with the dialysis care team. Escalate sooner if symptoms are present or concern persists.",
    centerAlert:
      "DEMO ALERT: nurse examination at the next session; escalate if symptoms or concern persist.",
    patientAlert:
      "DEMO MESSAGE: discuss this result with your dialysis care team at your next session.",
  },
  normal: {
    status: "green",
    title: "GREEN — routine surveillance",
    summary: "Simulated low-risk demo case; this does not rule out a problem.",
    finding: "No simulated high-risk finding",
    adviceTitle: "Continue routine access surveillance",
    advice:
      "A green demo result is not a clearance. Contact the dialysis team for pain, swelling, bleeding, inadequate dialysis, or other concerns.",
    centerAlert:
      "DEMO STATUS: no urgent alert preview. Continue routine surveillance.",
    patientAlert:
      "DEMO MESSAGE: continue routine care and contact your team if symptoms or concerns arise.",
  },
  retake: {
    status: "retake",
    title: "RETAKE — capture needs attention",
    summary: "The demo guardrail rejected this capture. No model result is shown.",
    finding: "No assessment produced",
    adviceTitle: "Retake the capture if appropriate",
    advice:
      "A rejected capture is not a normal result. Follow the capture instructions or contact the dialysis team if there are symptoms or concerns.",
    centerAlert:
      "NO CLINICAL ALERT: capture was rejected; no triage score was produced.",
    patientAlert:
      "DEMO MESSAGE: no assessment was made. Do not interpret this as a normal result.",
  },
  unavailable: {
    status: "unavailable",
    title: "UNAVAILABLE — unable to assess safely",
    summary: "The simulated analysis service is unavailable. No score was produced.",
    finding: "Assessment unavailable",
    adviceTitle: "Use the normal care pathway",
    advice:
      "No model-based advice is available. Do not treat this as a normal result. Contact the dialysis team for guidance, and seek immediate clinical help for urgent symptoms.",
    centerAlert:
      "FALLBACK: no clinical alert was sent. Use the clinic's normal escalation process.",
    patientAlert:
      "FALLBACK: no assessment is available. Contact your dialysis care team for guidance.",
  },
};

const SCENARIOS: Array<{ value: Scenario; label: string }> = [
  { value: "thrombosis", label: "High-risk example" },
  { value: "stenosis", label: "Review example" },
  { value: "normal", label: "Routine example" },
  { value: "retake", label: "Retake guardrail" },
  { value: "unavailable", label: "AI unavailable fallback" },
];

const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

function validateImage(file: File | null): string | null {
  if (!file) return "Choose both arm-down and arm-raised images.";
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
    return "Images must be JPEG, PNG, or WebP.";
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return "Each image must be 12 MB or smaller.";
  }
  return null;
}

async function checkImageDimensions(file: File): Promise<boolean> {
  const bitmap = await createImageBitmap(file);
  const valid = bitmap.width >= 320 && bitmap.height >= 320;
  bitmap.close();
  return valid;
}

export default function App() {
  const [currentUser, setCurrentUser] = useState<DemoUser | null>(null);
  const [down, setDown] = useState<File | null>(null);
  const [up, setUp] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [downUrl, setDownUrl] = useState("");
  const [upUrl, setUpUrl] = useState("");
  const [scenario, setScenario] = useState<Scenario>("stenosis");
  const [result, setResult] = useState<DemoResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [qualityMessage, setQualityMessage] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const [alerts, setAlerts] = useState<CenterAlert[]>([]);
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const [page, setPage] = useState<"capture" | "result" | "advice">("capture");

  useEffect(() => {
    if (!down) {
      setDownUrl("");
      return;
    }

    const url = URL.createObjectURL(down);
    setDownUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [down]);

  useEffect(() => {
    if (!up) {
      setUpUrl("");
      return;
    }

    const url = URL.createObjectURL(up);
    setUpUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [up]);

  function runDemo(selected: Scenario, patientRef: string, patientName: string) {
    const demoResult = DEMO_RESULTS[selected];
    queueCenterAlert(demoResult, patientRef, patientName);
    setScenario(selected);
    setResult(demoResult);
    setAcknowledged(false);
    setError("");
    setQualityMessage("");
    setPage(demoResult.status === "red" || demoResult.status === "amber" ? "advice" : "result");
  }

  function queueCenterAlert(
    demoResult: DemoResult,
    patientRef: string,
    patientName: string,
  ) {
    if (demoResult.status !== "red" && demoResult.status !== "amber") return;

    const alert: CenterAlert = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      patientName,
      patientRef,
      createdAt: new Date().toLocaleString(),
      triage: demoResult.status,
      finding: demoResult.finding,
      message: demoResult.centerAlert,
      workflowStatus: "new",
      imageDown: down,
      imageUp: up,
      audio,
      watch: createDemoWatchSnapshot(),
    };
    setAlerts((current) => [alert, ...current]);
  }

  function updateAlert(id: string, workflowStatus: AlertWorkflowStatus) {
    setAlerts((current) =>
      current.map((alert) =>
        alert.id === id ? { ...alert, workflowStatus } : alert,
      ),
    );
  }

  async function analyzeUploadedImages(
    event: React.FormEvent<HTMLFormElement>,
    patientRef: string,
    patientName: string,
  ) {
    event.preventDefault();
    setResult(null);
    setError("");
    setQualityMessage("");

    const downError = validateImage(down);
    const upError = validateImage(up);
    if (downError || upError || !down || !up) {
      setError(downError ?? upError ?? "Invalid image.");
      return;
    }

    setBusy(true);
    try {
      const [downDimensionsOk, upDimensionsOk] = await Promise.all([
        checkImageDimensions(down),
        checkImageDimensions(up),
      ]);

      if (!downDimensionsOk || !upDimensionsOk) {
        setQualityMessage("Basic file check failed: images must be at least 320 × 320 pixels.");
        setResult(DEMO_RESULTS.retake);
        setPage("result");
        return;
      }

      setQualityMessage(
        "Basic browser checks passed. This frontend-only prototype does not run the medical image model.",
      );
      const demoResult = DEMO_RESULTS[scenario];
      queueCenterAlert(demoResult, patientRef, patientName);
      setResult(demoResult);
      setAcknowledged(false);
      setPage(demoResult.status === "red" || demoResult.status === "amber" ? "advice" : "result");
    } catch {
      setResult(DEMO_RESULTS.unavailable);
      setPage("result");
    } finally {
      setBusy(false);
    }
  }

  function logout() {
    setCurrentUser(null);
    setPage("capture");
    setResult(null);
    setSelectedAlertId(null);
  }

  const selectedAlert = alerts.find((alert) => alert.id === selectedAlertId);

  if (!currentUser) {
    return <LoginPage onLogin={setCurrentUser} />;
  }

  if (currentUser.role === "admin") {
    return (
      <main className="shell">
        <header className="header role-header">
          <div>
            <p className="eyebrow">DIALYSIS CENTER · ADMIN ACCESS</p>
            <h1>{selectedAlert ? "Patient device details" : "Center alert desk"}</h1>
            <p className="intro">Signed in as {currentUser.name}</p>
          </div>
          <button className="secondary-button" type="button" onClick={logout}>
            Log out
          </button>
        </header>
        {selectedAlert ? (
          <CenterAlertDetails
            alert={selectedAlert}
            onBack={() => setSelectedAlertId(null)}
          />
        ) : (
          <DialysisCenterInbox
            alerts={alerts}
            onUpdateAlert={updateAlert}
            onSelectAlert={setSelectedAlertId}
            onLogout={logout}
          />
        )}
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="header">
        <div>
          <p className="eyebrow">AVF SENTINEL · PATIENT ACCESS</p>
          <h1>Access check</h1>
          <p className="intro">Signed in as {currentUser.name} · {currentUser.patientRef}</p>
        </div>
        <div className="header-actions">
          <span className="warning-badge">RESEARCH DEMO · NOT A MEDICAL DEVICE</span>
          <button className="secondary-button" type="button" onClick={logout}>
            Log out
          </button>
        </div>
      </header>

      {page === "capture" && <PatientProfile user={currentUser} />}

      {page === "capture" && <section className="grid">
        <form
          className="panel"
          onSubmit={(event) =>
            analyzeUploadedImages(event, currentUser.patientRef, currentUser.name)
          }
        >
          <div className="panel-heading">
            <span className="step-number">01</span>
            <h2>Forearm image scan</h2>
          </div>

          <label htmlFor="down">Arm-down photo</label>
          <input
            id="down"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(event) => setDown(event.target.files?.[0] ?? null)}
          />

          <label htmlFor="up">Arm-raised photo</label>
          <input
            id="up"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(event) => setUp(event.target.files?.[0] ?? null)}
          />

          <label htmlFor="audio">Bruit audio (optional WAV)</label>
          <input
            id="audio"
            type="file"
            accept="audio/wav,.wav"
            onChange={(event) => setAudio(event.target.files?.[0] ?? null)}
          />

          <label htmlFor="scenario">Simulated result for showcase</label>
          <select
            id="scenario"
            value={scenario}
            onChange={(event) => setScenario(event.target.value as Scenario)}
          >
            {SCENARIOS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>

          <button type="submit" disabled={busy}>
            {busy ? "Checking files…" : "Run browser demo"}
          </button>

          {error && <p className="error" role="alert">{error}</p>}
          {qualityMessage && <p className="status-note">{qualityMessage}</p>}

          <p className="hint">
            Browser checks only validate file type, size, and minimum dimensions.
            They do not check blur, lighting, arm visibility, or clinical findings.
          </p>
        </form>

        <section className="panel">
          <div className="panel-heading">
            <span className="step-number">02</span>
            <h2>Try a demo case</h2>
          </div>
          <p className="hint">
            These buttons display fixed mock results to demonstrate the workflow.
            They do not run the model.
          </p>
          <div className="demo-buttons">
            {SCENARIOS.map((item) => (
              <button
                className="secondary-button"
                key={item.value}
                type="button"
                onClick={() =>
                  runDemo(item.value, currentUser.patientRef, currentUser.name)
                }
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="pipeline">
            <span>Capture</span><b>→</b><span>Quality gate</span><b>→</b>
            <span>Review</span><b>→</b><span>Alert preview</span>
          </div>
        </section>
      </section>}

      {page === "capture" && (downUrl || upUrl) && (
        <section className="panel">
          <div className="panel-heading">
            <span className="step-number">03</span>
            <h2>Capture preview</h2>
          </div>
          <div className="photo-grid">
            {downUrl && (
              <figure>
                <img src={downUrl} alt="Selected arm-down forearm capture" />
                <figcaption>Arm down · {down?.name}</figcaption>
              </figure>
            )}
            {upUrl && (
              <figure>
                <img src={upUrl} alt="Selected arm-raised forearm capture" />
                <figcaption>Arm raised · {up?.name}</figcaption>
              </figure>
            )}
          </div>
          {audio && <p className="hint">Selected audio: {audio.name}</p>}
        </section>
      )}

      {result && page === "advice" && (
        <PatientCareAdvice
          status={result.status}
          finding={result.finding}
          onBackToCapture={() => setPage("capture")}
        />
      )}

      {result && page === "result" && (
        <section className="results" aria-live="polite">
          <div className={`result-banner ${result.status}`}>
            <p className="eyebrow">SIMULATED TRIAGE</p>
            <h2>{result.title}</h2>
            <p>{result.summary}</p>
          </div>

          <div className="grid">
            <section className="panel">
              <div className="panel-heading">
                <span className="step-number">04</span>
                <h2>Patient care advice</h2>
              </div>
              <h3>{result.adviceTitle}</h3>
              <p>{result.advice}</p>
              <p className="hint">
                Fixed demo guidance only. Follow the clinic&apos;s approved protocol.
              </p>
            </section>

            <section className="panel">
              <div className="panel-heading">
                <span className="step-number">05</span>
                <h2>Safety guardrails</h2>
              </div>
              <ul className="guardrails">
                <li>Retake or unavailable states never display as normal.</li>
                <li>A green result does not rule out stenosis or thrombosis.</li>
                <li>Missing audio is not treated as evidence of a normal result.</li>
                <li>Fallback guidance is fixed text; no generative AI is connected.</li>
              </ul>
              <p className="hint">Demo finding: {result.finding}</p>
            </section>
          </div>

          <button
            className="secondary-button"
            type="button"
            onClick={() => setPage("advice")}
          >
            View patient care guidance
          </button>

          <section className="panel">
            <div className="panel-heading alert-heading">
              <div className="panel-heading">
                <span className="step-number">06</span>
                <h2>Patient and dialysis-center alerts</h2>
              </div>
              <span className="simulation-badge">PREVIEW ONLY · NOT SENT</span>
            </div>

            <div className="grid">
              <article className="alert-card">
                <p className="eyebrow">DIALYSIS CENTER</p>
                <p>{result.centerAlert}</p>
                <span className="delivery-status">Delivery not configured</span>
              </article>
              <article className="alert-card">
                <p className="eyebrow">PATIENT</p>
                <p>{result.patientAlert}</p>
                <span className="delivery-status">Delivery not configured</span>
              </article>
            </div>

            <button
              className="secondary-button acknowledge"
              type="button"
              onClick={() => setAcknowledged(true)}
            >
              {acknowledged ? "Preview acknowledged (demo)" : "Acknowledge preview (demo)"}
            </button>
            <p className="hint">
              This does not send SMS, email, push notifications, or messages to a clinic.
            </p>
          </section>
          <button
            className="secondary-button"
            type="button"
            onClick={() => setPage("capture")}
          >
            Back to image capture
          </button>
        </section>
      )}

      <footer>
        <strong>Prototype only.</strong> Uses fixed simulated results; no medical
        model, patient database, or notification provider is connected.
      </footer>
    </main>
  );
}
