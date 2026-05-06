"""TriggerGate: orchestrates per-frame detection -> rule evaluation -> candidates.

This is the cheap layer that runs on every frame. It does not call the
LLM. When it returns a TriggerCandidate, the caller (app/main.py in a
later phase) decides whether to escalate to Grok/Gemini.

Responsibilities:
  - Track per-frame observation counts (debouncing)
  - Open kitchen_activity when threshold met
  - Run all rules, collect candidates
  - Emit events for state transitions so the audit trail captures them
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from context.manager import ContextManager
from triggers.risk_window import open_risk_window_event
from triggers.rules import (
    MIN_BOTTLE_OBSERVATION_FRAMES,
    MIN_KITCHEN_OBSERVATION_FRAMES,
    CandidateKind,
    DetectionResult,
    TriggerCandidate,
    detect_kitchen_scene,
    kitchen_abandoned_rule,
    medication_rule,
)


# Type alias for the event emitter the gate uses to push to the audit log.
EventEmitter = Callable[[dict], None]


@dataclass
class GateState:
    """Internal counters owned by the gate (not persisted in ContextManager)."""
    consecutive_kitchen_frames: int = 0
    consecutive_bottle_frames: int = 0


class TriggerGate:
    def __init__(self, context: ContextManager, emit_event: Optional[EventEmitter] = None) -> None:
        """
        Args:
            context: ContextManager whose state is read AND mutated via record_event.
            emit_event: optional sink for the JSONL audit trail. Tests can omit it.
        """
        self._context = context
        self._emit = emit_event or (lambda _e: None)
        self._gate_state = GateState()

    # --- Main entry point --------------------------------------------------

    def process_frame(self, detection: DetectionResult) -> list[TriggerCandidate]:
        """Apply all rules to this frame. Returns candidates to escalate.

        Side effects (via record_event + emit):
          - open kitchen_activity when threshold met
          - close kitchen_activity if it was open and gets re-engaged
        """
        candidates: list[TriggerCandidate] = []

        self._update_kitchen_activity(detection)
        self._update_bottle_observation(detection)

        # Run every rule against the (possibly updated) state.
        state = self._context.state

        # Kitchen abandonment
        # Gate the candidate behind the open-window check too: at 5 FPS once
        # the threshold is crossed, the rule fires every frame; without this
        # guard the LLM dispatcher would burn ~300 calls per 60s of absence.
        kitchen_candidate = kitchen_abandoned_rule(detection, state, now=detection.timestamp)
        if kitchen_candidate is not None and not self._has_open_window(state, CandidateKind.KITCHEN_ABANDONED.value):
            candidates.append(kitchen_candidate)
            event = open_risk_window_event(CandidateKind.KITCHEN_ABANDONED.value, now=detection.timestamp)
            self._record_and_emit(event)

        # Medication: only check after enough consecutive bottle frames so
        # that someone walking past does not trigger.
        if self._gate_state.consecutive_bottle_frames >= MIN_BOTTLE_OBSERVATION_FRAMES:
            med_candidate = medication_rule(detection, state, now=detection.timestamp)
            if med_candidate is not None and not self._has_open_window(state, med_candidate.kind.value):
                candidates.append(med_candidate)
                event = open_risk_window_event(med_candidate.kind.value, now=detection.timestamp)
                self._record_and_emit(event)

        return candidates

    # --- Activity tracking -------------------------------------------------

    def _update_kitchen_activity(self, detection: DetectionResult) -> None:
        in_kitchen = detect_kitchen_scene(detection)

        if in_kitchen:
            self._gate_state.consecutive_kitchen_frames += 1
            already_active = self._context.state.current_activity == "kitchen_activity"
            if (
                self._gate_state.consecutive_kitchen_frames >= MIN_KITCHEN_OBSERVATION_FRAMES
                and not already_active
            ):
                event = {
                    "event_type": "activity_started",
                    "activity": "kitchen_activity",
                    "timestamp": detection.timestamp.isoformat(),
                    "objects_observed": list(detection.objects),
                }
                self._record_and_emit(event)
        else:
            # Person not present OR no kitchen objects -> reset the counter.
            self._gate_state.consecutive_kitchen_frames = 0

    def _update_bottle_observation(self, detection: DetectionResult) -> None:
        """Track consecutive frames where a bottle is being handled.

        Resets the moment the bottle leaves the frame OR the person leaves
        — passing through view should not arm the medication rule.
        """
        bottle_handled = detection.person_present and ("bottle" in detection.objects)
        if bottle_handled:
            self._gate_state.consecutive_bottle_frames += 1
        else:
            self._gate_state.consecutive_bottle_frames = 0

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _has_open_window(state, scenario: str) -> bool:
        return any(w.scenario == scenario for w in state.open_risk_windows)

    def _record_and_emit(self, event: dict) -> None:
        """Apply event to ContextManager AND emit to audit trail.

        Single helper so the gate cannot accidentally update state without
        logging it (the contract that makes replay correct).
        """
        self._context.record_event(event)
        self._emit(event)
