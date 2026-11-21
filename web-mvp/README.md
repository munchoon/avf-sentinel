# AVF Sentinel Web MVP

This is a frontend-only React/TypeScript showcase in a separate folder. It does
not modify or connect to the Python backend. Results, patient guidance, and
alert messages are fixed demo content; the UI does not run an AI model or send
notifications.

## Run locally

```powershell
cd web-mvp
npm install
npm run dev
```

Open the local URL printed by Vite. Use the demo scenario buttons to show the
triage states, care-guidance panel, alert previews, and unavailable/retake
fallbacks. Uploading images only checks file type, size, and minimum dimensions.

## Role-separated demo login

The login is mock frontend authentication only. Use the patient role to create a
simulated red/amber result, then log out and use the admin role to review the
alert in the center inbox.

| Role | Demo email | Demo access code |
|---|---|---|
| Patient | `patient.demo@avf.test` | `PatientDemo2026` |
| Dialysis center admin | `center.admin@avf.test` | `CenterDemo2026` |

Alerts remain in memory while the page is open. Refreshing the browser clears
them. The credentials are public demo values and must not be used for real
authentication or patient data.

## Patient smartwatch profile

The patient capture page displays sample heart-rate, blood-oxygen, activity,
sleep, and battery values. These are illustrative frontend values; no smartwatch
or wearable API is connected. The refresh button updates only the displayed
demo sync time.

## Build and type-check

```powershell
npm run build
```
