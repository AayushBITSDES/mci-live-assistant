"""Map recognizer observations into deterministic exhibition events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from demo.orchestrator import DemoOrchestrator
from triggers.rules import DetectionResult


@dataclass
class ObservationAdapter:
    """Small debouncer between per-frame recognition and demo state.

    Faces re-fire their cue once the same person has been unseen for at
    least ``face_cue_rearm_seconds``. That keeps the cue from spamming on
    every frame while still letting a returning visitor (or a new round
    of the same visitor) re-trigger the welcome.
    """

    face_cue_rearm_seconds: float = 60.0
    _face_last_seen_at: dict[str, datetime] = field(default_factory=dict)

    def reset(self) -> None:
        """Wipe the per-name dedup so all cues will fire again.

        The operator should call this between demo rounds.
        """
        self._face_last_seen_at.clear()

    def apply(
        self,
        demo: DemoOrchestrator,
        detection: DetectionResult,
    ) -> list[dict]:
        messages: list[dict] = []
        labels = set(detection.objects)
        now = detection.timestamp
        rearm = timedelta(seconds=self.face_cue_rearm_seconds)

        if "medicine_bottle" in labels:
            demo.record_medicine_bottle_seen(now=now)
        else:
            messages.extend(demo.check_medicine_reminder(now=now))

        if "stove" in labels:
            messages.extend(demo.record_stove_interaction(now=now))
        else:
            demo.record_stove_absent()
        messages.extend(demo.check_stove_timers(now=now))

        for name in detection.recognized_names:
            last = self._face_last_seen_at.get(name)
            if last is not None and now - last < rearm:
                # Still inside the cooldown window; just refresh the
                # last-seen time so the cooldown is measured from the
                # latest sighting, not the first.
                self._face_last_seen_at[name] = now
                continue
            self._face_last_seen_at[name] = now
            messages.extend(demo.trigger_face_cue(name, now=now))

        return messages
