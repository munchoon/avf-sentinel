export type WatchReading = {
  label: string;
  value: string;
  unit: string;
  note: string;
};

export type DemoWatchDevice = {
  name: string;
  deviceId: string;
  connection: string;
};

export type DemoWatchSnapshot = {
  device: DemoWatchDevice;
  lastSync: string;
  readings: WatchReading[];
};

export const DEMO_WATCH_DEVICE: DemoWatchDevice = {
  name: "AVF Sentinel demo wearable",
  deviceId: "DEMO-WATCH-001",
  connection: "Simulated connection",
};

export const DEMO_WATCH_READINGS: WatchReading[] = [
  { label: "Resting heart rate", value: "68", unit: "bpm", note: "Today" },
  { label: "Blood oxygen", value: "97", unit: "%", note: "If supported by watch" },
  { label: "Steps", value: "5,842", unit: "steps", note: "Today" },
  { label: "Active time", value: "32", unit: "min", note: "Today" },
  { label: "Sleep", value: "7h 18m", unit: "", note: "Last night" },
  { label: "Watch battery", value: "84", unit: "%", note: "Demo device" },
];

export function createDemoWatchSnapshot(): DemoWatchSnapshot {
  return {
    device: { ...DEMO_WATCH_DEVICE },
    lastSync: new Date().toISOString(),
    readings: DEMO_WATCH_READINGS.map((reading) => ({ ...reading })),
  };
}
