// Persistent indicator that camera/mic is active.
// Maps to Design Decision #6: small green dot, always visible while
// the camera is on.

export interface PrivacyDotProps {
  active: boolean;
}

export function PrivacyDot({ active }: PrivacyDotProps) {
  return (
    <div className="privacy-dot-wrap" aria-label={active ? "Camera active" : "Camera off"}>
      <span className={`privacy-dot ${active ? "active" : "inactive"}`} />
      <span className="privacy-label">{active ? "live" : "off"}</span>
    </div>
  );
}
