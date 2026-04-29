import { scenarios } from "./scenarios";
import type { NudgeDecision, ScenarioRuntime } from "./types";

const RETURN_AFTER_AUTO_DISMISS_MS = 6000;

export function evaluateNudge(runtime: ScenarioRuntime, now = Date.now()): NudgeDecision {
  const scenario = scenarios[runtime.scenarioId];
  const step = scenario.steps[runtime.stepIndex];

  if (!step) {
    return silent("No scenario step is active.");
  }

  if (runtime.taskCompleted) {
    return silent("The task has been marked complete.");
  }

  if (runtime.closedForever) {
    return silent("The user permanently closed this task.");
  }

  if (runtime.snoozedUntil && runtime.snoozedUntil > now) {
    return silent("The nudge is temporarily dismissed.");
  }

  if (step.event !== "risk_window" || step.riskScore < 0.7) {
    return silent("The current signal is not urgent enough.");
  }

  return {
    shouldNudge: true,
    sentence: scenario.plannedNudge,
    reason: `${scenario.code} risk window is active with score ${step.riskScore.toFixed(2)}.`,
    priority: step.riskScore > 0.9 ? "high" : "medium"
  };
}

export function autoDismiss(runtime: ScenarioRuntime, now = Date.now()): ScenarioRuntime {
  return {
    ...runtime,
    snoozedUntil: now + RETURN_AFTER_AUTO_DISMISS_MS
  };
}

export function dismissTemporarily(runtime: ScenarioRuntime): ScenarioRuntime {
  return {
    ...runtime,
    snoozedUntil: Date.now() + 30_000
  };
}

export function markDone(runtime: ScenarioRuntime): ScenarioRuntime {
  return {
    ...runtime,
    taskCompleted: true,
    snoozedUntil: null
  };
}

export function closeForever(runtime: ScenarioRuntime): ScenarioRuntime {
  return {
    ...runtime,
    closedForever: true,
    snoozedUntil: null
  };
}

export function flagWrong(runtime: ScenarioRuntime): ScenarioRuntime {
  return {
    ...runtime,
    flaggedWrong: true,
    snoozedUntil: Date.now() + 60_000
  };
}

function silent(reason: string): NudgeDecision {
  return {
    shouldNudge: false,
    sentence: "",
    reason,
    priority: "low"
  };
}
