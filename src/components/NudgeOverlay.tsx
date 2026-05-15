// Nudge overlay: one full sentence, fades in/out, auto-dismisses.
// Props are STABLE — Figma export drops in here as a replacement.

import { useEffect, useRef, useState } from "react";
import type { NudgeMessage } from "../lib/types";

export interface NudgeOverlayProps {
  nudge: NudgeMessage | null;
  onDismiss: (action: "nudge_closed" | "nudge_auto_dismiss") => void;
}

export function NudgeOverlay({ nudge, onDismiss }: NudgeOverlayProps) {
  const [visible, setVisible] = useState(false);
  const onDismissRef = useRef(onDismiss);
  const timeoutRef = useRef<number | null>(null);
  onDismissRef.current = onDismiss;

  const nudgeId = nudge?.nudge_id;
  const dismissSec = nudge?.auto_dismiss_seconds ?? 10;

  useEffect(() => {
    if (!nudgeId) {
      setVisible(false);
      return;
    }
    setVisible(true);

    timeoutRef.current = window.setTimeout(() => {
      timeoutRef.current = null;
      setVisible(false);
      onDismissRef.current("nudge_auto_dismiss");
    }, dismissSec * 1000);

    return () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [nudgeId, dismissSec]);

  if (!nudge || !visible) return null;

  const accent = nudge.priority === "safety" ? "#ff6b6b" : "#ffd166";

  return (
    <div className="nudge-overlay" role="status" aria-live="polite">
      <div className="nudge-card" style={{ borderLeftColor: accent }}>
        <span className="nudge-icon">🪞</span>
        <span className="nudge-text">{nudge.sentence}</span>
        <button
          className="nudge-close"
          onClick={() => {
            if (timeoutRef.current !== null) {
              window.clearTimeout(timeoutRef.current);
              timeoutRef.current = null;
            }
            setVisible(false);
            onDismissRef.current("nudge_closed");
          }}
          type="button"
          aria-label="Close nudge"
        >
          Close
        </button>
      </div>
    </div>
  );
}
