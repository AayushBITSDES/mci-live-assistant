export type SurfaceMode = "desktop" | "mobile" | "quest";

export type ScenarioId = "b2-shanta-blood-test" | "b1-sanjay-client-deck";

export type DismissalMode = "temporary" | "completed" | "forever" | "flagged";

export type ScenarioEvent =
  | "routine_started"
  | "interruption"
  | "risk_window"
  | "task_completed"
  | "dismissed"
  | "closed_forever";

export interface ScenarioStep {
  id: string;
  title: string;
  ambientDetail: string;
  event: ScenarioEvent;
  riskScore: number;
}

export interface ScenarioDefinition {
  id: ScenarioId;
  code: string;
  persona: string;
  title: string;
  summary: string;
  targetSurface: SurfaceMode;
  context: string[];
  plannedNudge: string;
  steps: ScenarioStep[];
}

export interface AssistantSettings {
  visualEnabled: boolean;
  audioEnabled: boolean;
  cameraActive: boolean;
  micActive: boolean;
  nudgeDurationSeconds: number;
}

export interface ScenarioRuntime {
  scenarioId: ScenarioId;
  stepIndex: number;
  taskCompleted: boolean;
  closedForever: boolean;
  snoozedUntil: number | null;
  flaggedWrong: boolean;
}

export interface NudgeDecision {
  shouldNudge: boolean;
  sentence: string;
  reason: string;
  priority: "low" | "medium" | "high";
}
