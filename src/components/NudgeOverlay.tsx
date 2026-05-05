// Nudge overlay: one full sentence, fades in/out, auto-dismisses.
// Props are STABLE — Figma export drops in here as a replacement.

import { useEffect, useState } from "react";
import type { NudgeMessage } from "../lib/types";

export interface NudgeOverlayProps {
  nudge: NudgeMessage | null;
  onDismiss: () => void;
}

export function NudgeOverlay({ nudge, onDismiss }: NudgeOverlayProps) {
  const [visible, setVisible] = useState(false);
  const [currentId, setCurrentId] = useState<string | null>(null);

  useEffect(() => {
    if (!nudge) {
      setVisible(false);
      return;
    }
    if (nudge.nudge_id === currentId) {
      return;
    }
    setCurrentId(nudge.nudge_id);
    setVisible(true);

    const timeout = window.setTimeout(() => {
      setVisible(false);
      onDismiss();
    }, (nudge.auto_dismiss_seconds || 10) * 1000);

    return () => window.clearTimeout(timeout);
  }, [nudge, currentId, onDismiss]);

  if (!nudge || !visible) return null;

  const accent = nudge.priority === "safety" ? "#ff6b6b" : "#ffd166";

  return (
    <div className="nudge-overlay" role="status" aria-live="polite">
      <div className="nudge-card" style={{ borderLeftColor: accent }}>
        <span className="nudge-icon">🪞</span>
        <span className="nudge-text">{nudge.sentence}</span>
      </div>
    </div>
  );
}
