// Nudge overlay: one full sentence, fades in/out, auto-dismisses.
// Props are STABLE — Figma export drops in here as a replacement.

import { useEffect, useRef, useState } from "react";
import type { NudgeMessage } from "../lib/types";

export interface NudgeOverlayProps {
  nudge: NudgeMessage | null;
  onDismiss: () => void;
}

export function NudgeOverlay({ nudge, onDismiss }: NudgeOverlayProps) {
  const [visible, setVisible] = useState(false);
  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  const nudgeId = nudge?.nudge_id;
  const dismissSec = nudge?.auto_dismiss_seconds ?? 10;

  useEffect(() => {
    if (!nudgeId) {
      setVisible(false);
      return;
    }
    setVisible(true);

    const timeout = window.setTimeout(() => {
      setVisible(false);
      onDismissRef.current();
    }, dismissSec * 1000);

    return () => window.clearTimeout(timeout);
  }, [nudgeId, dismissSec]);

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
