"""Map recognizer observations into deterministic exhibition events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from demo.orchestrator import DemoOrchestrator
from triggers.rules import DetectionResult


@dataclass
class ObservationAdapter:
    """Small debouncer between per-frame recognition and demo state."""

    seen_names: set[str] = field(default_factory=set)

    def apply(
        self,
        demo: DemoOrchestrator,
        detection: DetectionResult,
    ) -> list[dict]:
        messages: list[dict] = []
        labels = set(detection.objects)
        now = detection.timestamp

        if "medicine_bottle" in labels:
            demo.record_medicine_bottle_seen(now=now)
        else:
            messages.extend(demo.check_medicine_reminder(now=now))

        if "stove" in labels:
            messages.extend(demo.record_stove_interaction(now=now))
        messages.extend(demo.check_stove_timers(now=now))

        for name in detection.recognized_names:
            if name in self.seen_names:
                continue
            self.seen_names.add(name)
            messages.extend(demo.trigger_face_cue(name))

        return messages
