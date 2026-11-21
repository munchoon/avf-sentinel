import { useState } from "react";
import {
  DEMO_LOGINS,
  loginDemo,
  type DemoRole,
  type DemoUser,
} from "./demoAuth";

type LoginPageProps = {
  onLogin: (user: DemoUser) => void;
};

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [role, setRole] = useState<DemoRole>("patient");
  const [email, setEmail] = useState("");
  const [accessCode, setAccessCode] = useState("");
  const [error, setError] = useState("");

  function chooseRole(nextRole: DemoRole) {
    setRole(nextRole);
    setEmail("");
    setAccessCode("");
    setError("");
  }

  function fillDemoLogin() {
    setEmail(DEMO_LOGINS[role].email);
    setAccessCode(DEMO_LOGINS[role].accessCode);
    setError("");
  }

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const user = loginDemo(role, email, accessCode);
    if (!user) {
      setError("The demo email, access code, or selected role does not match.");
      return;
    }
    onLogin(user);
  }

  return (
    <main className="login-shell">
      <section className="login-card">
        <p className="eyebrow">AVF SENTINEL · HACKATHON WEB MVP</p>
        <h1>Sign in</h1>
        <p className="login-intro">
          Choose a demo role. Patient and dialysis-center access are separated.
        </p>
        <span className="warning-badge">DEMO AUTHENTICATION · NOT SECURE</span>

        <div className="role-picker" aria-label="Select demo role">
          <button
            className={role === "patient" ? "role-option selected" : "role-option"}
            type="button"
            aria-pressed={role === "patient"}
            onClick={() => chooseRole("patient")}
          >
            Patient
          </button>
          <button
            className={role === "admin" ? "role-option selected" : "role-option"}
            type="button"
            aria-pressed={role === "admin"}
            onClick={() => chooseRole("admin")}
          >
            Dialysis center admin
          </button>
        </div>

        <form onSubmit={submit}>
          <label htmlFor="login-email">Demo email</label>
          <input
            id="login-email"
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />

          <label htmlFor="login-code">Demo access code</label>
          <input
            id="login-code"
            type="password"
            autoComplete="current-password"
            required
            value={accessCode}
            onChange={(event) => setAccessCode(event.target.value)}
          />

          {error && <p className="error" role="alert">{error}</p>}

          <button className="login-submit" type="submit">
            Sign in as {role === "patient" ? "patient" : "center admin"}
          </button>
          <button className="secondary-button fill-demo" type="button" onClick={fillDemoLogin}>
            Fill demo credentials
          </button>
        </form>

        <div className="login-safety">
          <strong>Prototype login only.</strong> Credentials are embedded in
          frontend code. Do not use real accounts, patient data, or this login
          design for production authentication.
        </div>
      </section>
    </main>
  );
}
