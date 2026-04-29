import type { AssistantSettings, ScenarioRuntime } from "../engine/types";

const SETTINGS_KEY = "mci-live-assistant:settings";
const RUNTIME_KEY = "mci-live-assistant:runtime";

export const defaultSettings: AssistantSettings = {
  visualEnabled: true,
  audioEnabled: true,
  cameraActive: true,
  micActive: true,
  nudgeDurationSeconds: 10
};

export const defaultRuntime: ScenarioRuntime = {
  scenarioId: "b2-shanta-blood-test",
  stepIndex: 0,
  taskCompleted: false,
  closedForever: false,
  snoozedUntil: null,
  flaggedWrong: false
};

export function loadSettings(): AssistantSettings {
  return loadValue(SETTINGS_KEY, defaultSettings);
}

export function saveSettings(settings: AssistantSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

export function loadRuntime(): ScenarioRuntime {
  return loadValue(RUNTIME_KEY, defaultRuntime);
}

export function saveRuntime(runtime: ScenarioRuntime) {
  localStorage.setItem(RUNTIME_KEY, JSON.stringify(runtime));
}

function loadValue<T>(key: string, fallback: T): T {
  try {
    const stored = localStorage.getItem(key);
    return stored ? { ...fallback, ...JSON.parse(stored) } : fallback;
  } catch {
    return fallback;
  }
}
