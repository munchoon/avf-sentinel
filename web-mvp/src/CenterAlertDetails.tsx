import { useEffect, useState } from "react";
import type { CenterAlert } from "./DialysisCenterInbox";

type CenterAlertDetailsProps = {
  alert: CenterAlert;
  onBack: () => void;
};

export default function CenterAlertDetails({
  alert,
  onBack,
}: CenterAlertDetailsProps) {
  const [downImageUrl, setDownImageUrl] = useState("");
  const [upImageUrl, setUpImageUrl] = useState("");
  const [audioUrl, setAudioUrl] = useState("");

  useEffect(() => {
    const downUrl = alert.imageDown ? URL.createObjectURL(alert.imageDown) : "";
    const upUrl = alert.imageUp ? URL.createObjectURL(alert.imageUp) : "";
    const clipUrl = alert.audio ? URL.createObjectURL(alert.audio) : "";

    setDownImageUrl(downUrl);
    setUpImageUrl(upUrl);
    setAudioUrl(clipUrl);

    return () => {
      if (downUrl) URL.revokeObjectURL(downUrl);
      if (upUrl) URL.revokeObjectURL(upUrl);
      if (clipUrl) URL.revokeObjectURL(clipUrl);
    };
  }, [alert.id, alert.imageDown, alert.imageUp, alert.audio]);

  return (
    <section className="alert-details">
      <div className={`detail-banner ${alert.triage}`}>
        <p className="eyebrow">DIALYSIS CENTER · PATIENT CASE · DEMO</p>
        <h2>{alert.triage === "red" ? "Same-day review alert" : "Access review alert"}</h2>
        <p>{alert.message}</p>
        <span className="simulation-badge">SAMPLE DATA · NOT A CLINICAL RECORD</span>
      </div>

      <section className="panel detail-profile">
        <div>
          <p className="eyebrow">PATIENT PROFILE</p>
          <h3>{alert.patientName}</h3>
          <p>
            Reference: {alert.patientRef} · AV fistula access · Left arm (demo profile)
          </p>
        </div>
        <div className="detail-state">
          <span className={`triage-pill ${alert.triage}`}>
            {alert.triage === "red" ? "RED · SAME DAY" : "AMBER · REVIEW"}
          </span>
          <span className={`workflow-pill ${alert.workflowStatus}`}>
            {alert.workflowStatus}
          </span>
          <time>{alert.createdAt}</time>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <span className="step-number">01</span>
          <h2>Smartwatch device data</h2>
        </div>
        <div className="device-summary">
          <span><strong>Device:</strong> {alert.watch.device.name}</span>
          <span><strong>Device ID:</strong> {alert.watch.device.deviceId}</span>
          <span><strong>Connection:</strong> {alert.watch.device.connection}</span>
          <span>
            <strong>Last sync:</strong> {new Date(alert.watch.lastSync).toLocaleString()}
          </span>
        </div>
        <div className="watch-metrics">
          {alert.watch.readings.map((reading) => (
            <article className="watch-metric" key={reading.label}>
              <span className="metric-label">{reading.label}</span>
              <strong>
                {reading.value}
                {reading.unit && <small> {reading.unit}</small>}
              </strong>
              <span className="metric-note">{reading.note}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <span className="step-number">02</span>
          <h2>Forearm image captures</h2>
        </div>
        {downImageUrl || upImageUrl ? (
          <div className="detail-image-grid">
            {downImageUrl && (
              <figure>
                <img src={downImageUrl} alt="Patient arm-down forearm capture" />
                <figcaption>Arm down · {alert.imageDown?.name}</figcaption>
              </figure>
            )}
            {upImageUrl && (
              <figure>
                <img src={upImageUrl} alt="Patient arm-raised forearm capture" />
                <figcaption>Arm raised · {alert.imageUp?.name}</figcaption>
              </figure>
            )}
          </div>
        ) : (
          <p className="empty-media">
            No image files were attached to this demo alert. Try uploading images
            before running a simulated result to preview them here.
          </p>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <span className="step-number">03</span>
          <h2>Bruit audio</h2>
        </div>
        {audioUrl ? (
          <>
            <p>{alert.audio?.name}</p>
            <audio controls preload="metadata" src={audioUrl}>
              Your browser does not support audio playback.
            </audio>
          </>
        ) : (
          <p className="empty-media">No audio file was attached to this demo alert.</p>
        )}
      </section>

      <div className="detail-safety-note">
        This mock page displays only in-memory demo data. It is not connected to a
        watch, medical model, database, or notification service. Follow clinic
        privacy and access-control requirements before using real patient data.
      </div>

      <button className="secondary-button back-button" type="button" onClick={onBack}>
        Back to alert inbox
      </button>
    </section>
  );
}
