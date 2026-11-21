import { useState } from "react";
import type { DemoUser } from "./demoAuth";
import { DEMO_WATCH_READINGS } from "./demoWatchData";

type PatientUser = Extract<DemoUser, { role: "patient" }>;

type PatientProfileProps = {
  user: PatientUser;
};

export default function PatientProfile({ user }: PatientProfileProps) {
  const [lastSync, setLastSync] = useState(() => new Date());

  function refreshDemoReadings() {
    setLastSync(new Date());
  }

  return (
    <section className="patient-profile">
      <div className="profile-card">
        <div className="profile-heading">
          <div className="profile-avatar" aria-hidden="true">
            {user.name.slice(0, 1).toUpperCase()}
          </div>
          <div className="profile-identity">
            <p className="eyebrow">PATIENT PROFILE</p>
            <h2>{user.name}</h2>
            <p>{user.patientRef} · AV fistula access · Left arm (demo profile)</p>
          </div>
          <span className="profile-demo-badge">SAMPLE PROFILE</span>
        </div>
      </div>

      <div className="watch-card">
        <div className="watch-heading">
          <div>
            <p className="eyebrow">CONNECTED DEVICE</p>
            <h2>Smartwatch data</h2>
          </div>
          <span className="watch-connection">
            <span className="connection-dot" />
            Demo connected
          </span>
        </div>

        <div className="watch-meta">
          <span>Last demo sync: {lastSync.toLocaleTimeString()}</span>
          <button
            className="secondary-button sync-button"
            type="button"
            onClick={refreshDemoReadings}
          >
            Refresh demo data
          </button>
        </div>

        <div className="watch-metrics">
          {DEMO_WATCH_READINGS.map((reading) => (
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

        <p className="watch-disclaimer">
          These are illustrative values, not readings from a paired watch.
          Consumer smartwatch measurements are not used to diagnose AV fistula
          problems and do not replace clinical assessment.
        </p>
      </div>
    </section>
  );
}
