// Surface switcher: mobile / quest / desktop.

import type { SurfaceMode } from "../lib/types";

const SURFACES: { value: SurfaceMode; label: string; icon: string }[] = [
  { value: "desktop", label: "Desktop", icon: "🖥️" },
  { value: "mobile", label: "Mobile", icon: "📱" },
  { value: "quest", label: "Quest", icon: "🥽" },
];

export interface SurfaceSwitcherProps {
  surface: SurfaceMode;
  setSurface: (next: SurfaceMode) => void;
  disabled?: boolean;
}

export function SurfaceSwitcher({
  surface,
  setSurface,
  disabled,
}: SurfaceSwitcherProps) {
  return (
    <div className="surface-switcher" role="radiogroup" aria-label="Surface">
      {SURFACES.map((s) => (
        <button
          key={s.value}
          type="button"
          className={surface === s.value ? "surface-pill active" : "surface-pill"}
          onClick={() => setSurface(s.value)}
          disabled={disabled}
          role="radio"
          aria-checked={surface === s.value}
        >
          <span className="surface-icon">{s.icon}</span>
          {s.label}
        </button>
      ))}
    </div>
  );
}
