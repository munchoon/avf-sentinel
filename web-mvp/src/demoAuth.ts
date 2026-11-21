export type DemoRole = "patient" | "admin";

export type DemoUser =
  | {
      role: "patient";
      name: string;
      email: string;
      patientRef: string;
    }
  | {
      role: "admin";
      name: string;
      email: string;
    };

export const DEMO_LOGINS: Record<
  DemoRole,
  { email: string; accessCode: string; name: string }
> = {
  patient: {
    email: "patient.demo@avf.test",
    accessCode: "PatientDemo2026",
    name: "Demo Patient",
  },
  admin: {
    email: "center.admin@avf.test",
    accessCode: "CenterDemo2026",
    name: "Dialysis Center Admin",
  },
};

export function loginDemo(
  role: DemoRole,
  email: string,
  accessCode: string,
): DemoUser | null {
  const credentials = DEMO_LOGINS[role];
  if (
    email.trim().toLowerCase() !== credentials.email ||
    accessCode !== credentials.accessCode
  ) {
    return null;
  }

  if (role === "patient") {
    return {
      role,
      name: credentials.name,
      email: credentials.email,
      patientRef: "DEMO-PATIENT-001",
    };
  }

  return {
    role,
    name: credentials.name,
    email: credentials.email,
  };
}
