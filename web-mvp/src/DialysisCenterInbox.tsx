import type { CareStatus } from "./PatientCareAdvice";
import type { DemoWatchSnapshot } from "./demoWatchData";

export type AlertWorkflowStatus = "new" | "acknowledged" | "resolved";

export type CenterAlert = {
  id: string;
  patientName: string;
  patientRef: string;
  createdAt: string;
  triage: Extract<CareStatus, "red" | "amber">;
  finding: string;
  message: string;
  workflowStatus: AlertWorkflowStatus;
  imageDown: File | null;
  imageUp: File | null;
  audio: File | null;
  watch: DemoWatchSnapshot;
};

type DialysisCenterInboxProps = {
  alerts: CenterAlert[];
  onUpdateAlert: (id: string, status: AlertWorkflowStatus) => void;
  onSelectAlert: (id: string) => void;
  onLogout: () => void;
};

export default function DialysisCenterInbox({
  alerts,
  onUpdateAlert,
  onSelectAlert,
  onLogout,
}: DialysisCenterInboxProps) {
  const activeAlerts = alerts.filter((alert) => alert.workflowStatus !== "resolved");
  const redCount = activeAlerts.filter((alert) => alert.triage === "red").length;
  const newCount = alerts.filter((alert) => alert.workflowStatus === "new").length;

  return (
    <section className="center-inbox" aria-live="polite">
      <header className="inbox-header">
        <div>
          <p className="eyebrow">DIALYSIS CENTER · MOCK ALERT DESK</p>
          <h2>AVF review inbox</h2>
          <p className="hint">
            Demo-only patient references and alerts generated from simulated
            results. No real patient record or notification is connected.
          </p>
        </div>
        <span className="simulation-badge">SIMULATION ONLY</span>
      </header>

      <div className="inbox-stats">
        <article className="inbox-stat">
          <span>Open alerts</span>
          <strong>{activeAlerts.length}</strong>
        </article>
        <article className="inbox-stat urgent-stat">
          <span>Red · same-day review</span>
          <strong>{redCount}</strong>
        </article>
        <article className="inbox-stat">
          <span>New</span>
          <strong>{newCount}</strong>
        </article>
      </div>

      {alerts.length === 0 ? (
        <div className="empty-inbox">
          <h3>No demo alerts yet</h3>
          <p>
            Log out and sign in as the patient demo account. Run a high-risk or
            review case, then return here to inspect its alert and attached demo data.
          </p>
          <button className="secondary-button" type="button" onClick={onLogout}>
            Log out
          </button>
        </div>
      ) : (
        <div className="alert-list">
          {alerts.map((alert) => (
            <article
              className={`center-alert ${alert.triage} ${alert.workflowStatus}`}
              key={alert.id}
            >
              <div className="center-alert-top">
                <div>
                  <span className={`triage-pill ${alert.triage}`}>
                    {alert.triage === "red" ? "RED · SAME DAY" : "AMBER · REVIEW"}
                  </span>
                  <span className={`workflow-pill ${alert.workflowStatus}`}>
                    {alert.workflowStatus}
                  </span>
                </div>
                <time>{alert.createdAt}</time>
              </div>

              <h3>Patient {alert.patientRef}</h3>
              <p><strong>Demo finding:</strong> {alert.finding}</p>
              <p>{alert.message}</p>
              <p className="alert-route">
                {alert.triage === "red"
                  ? "Suggested workflow: contact the care team for same-day review."
                  : "Suggested workflow: nurse examination at the next session; escalate if needed."}
              </p>

              <button
                className="alert-view-button"
                type="button"
                onClick={() => onSelectAlert(alert.id)}
              >
                View patient profile, device data, images & audio
              </button>

              {alert.workflowStatus !== "resolved" ? (
                <div className="alert-actions">
                  {alert.workflowStatus === "new" && (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => onUpdateAlert(alert.id, "acknowledged")}
                    >
                      Acknowledge
                    </button>
                  )}
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onUpdateAlert(alert.id, "resolved")}
                  >
                    Mark follow-up complete
                  </button>
                </div>
              ) : (
                <p className="resolved-note">Demo follow-up marked complete.</p>
              )}
            </article>
          ))}
        </div>
      )}

      <div className="inbox-safety-note">
        <strong>Not connected:</strong> these cards are stored only in this page’s
        temporary browser state. They do not send SMS, email, push notifications,
        or messages to a dialysis center.
      </div>

      <button className="secondary-button back-button" type="button" onClick={onLogout}>
        Log out
      </button>
    </section>
  );
}
