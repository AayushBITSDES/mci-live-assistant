// MCI Edge Client — thin client that streams camera frames to the home
// server and renders nudges it sends back. No on-device ML, no scenarios,
// no engine: this is the dumb end of the rope.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ConnectionBar } from "./components/ConnectionBar";
import { NudgeOverlay } from "./components/NudgeOverlay";
import { PrivacyDot } from "./components/PrivacyDot";
import { SurfaceSwitcher } from "./components/SurfaceSwitcher";
import { playChimeAndAudioB64, playChimeAndSpeak } from "./lib/audio";
import { captureFrameJpeg, startCamera, stopCamera } from "./lib/camera";
import type { SurfaceMode } from "./lib/types";
import { useWebSocket } from "./lib/useWebSocket";

const DEFAULT_WS_URL = "ws://localhost:8000/ws/stream";
const TARGET_FPS = 5;
const DEVICE_ID = "edge-" + Math.random().toString(36).slice(2, 10);

function App() {
  const [surface, setSurface] = useState<SurfaceMode>(() => detectSurface());
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [cameraActive, setCameraActive] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastSpokenIdRef = useRef<string | null>(null);

  const { state, connect, disconnect, send } = useWebSocket();
  const isConnected = state.status === "connected";

  // Start/stop camera in lockstep with the connection.
  useEffect(() => {
    if (!isConnected) {
      stopCamera(streamRef.current, videoRef.current);
      streamRef.current = null;
      setCameraActive(false);
      return;
    }
    if (!videoRef.current) return;

    let cancelled = false;
    startCamera(videoRef.current)
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setCameraActive(true);
      })
      .catch((err) => {
        console.error("Camera error", err);
        setCameraActive(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isConnected]);

  // Frame loop — captures a JPEG every 1/FPS seconds while connected.
  useEffect(() => {
    if (!isConnected || !cameraActive) return;
    const interval = window.setInterval(() => {
      if (!videoRef.current) return;
      const frame = captureFrameJpeg(videoRef.current);
      if (!frame) return;
      send({
        type: "frame",
        device_id: DEVICE_ID,
        surface,
        timestamp: new Date().toISOString(),
        image_b64: frame.base64,
        width: frame.width,
        height: frame.height,
      });
    }, 1000 / TARGET_FPS);
    return () => window.clearInterval(interval);
  }, [isConnected, cameraActive, surface, send]);

  // Play audio whenever a new nudge arrives.
  useEffect(() => {
    if (!audioEnabled || !state.lastNudge) return;
    if (state.lastNudge.nudge_id === lastSpokenIdRef.current) return;
    lastSpokenIdRef.current = state.lastNudge.nudge_id;

    if (state.lastNudge.audio_b64) {
      playChimeAndAudioB64(state.lastNudge.audio_b64);
    } else {
      playChimeAndSpeak(state.lastNudge.sentence);
    }
  }, [state.lastNudge, audioEnabled]);

  const handleConnect = useCallback(() => {
    connect(wsUrl, DEVICE_ID, surface);
  }, [connect, wsUrl, surface]);

  const handleDisconnect = useCallback(() => {
    disconnect();
  }, [disconnect]);

  const handleNudgeDismiss = useCallback(() => {
    // Phase 9 leaves dismissal as a no-op; voice command handler lands
    // when ASR is wired in (Phase 5 deployment).
  }, []);

  const surfaceClass = useMemo(() => `app surface-${surface}`, [surface]);

  return (
    <div className={surfaceClass}>
      <header className="app-header">
        <h1>🪞 MCI Edge</h1>
        <SurfaceSwitcher
          surface={surface}
          setSurface={setSurface}
          disabled={isConnected}
        />
      </header>

      <ConnectionBar
        url={wsUrl}
        setUrl={setWsUrl}
        state={state}
        onConnect={handleConnect}
        onDisconnect={handleDisconnect}
      />

      <main className="stage">
        <video
          ref={videoRef}
          className={`camera ${isConnected ? "live" : "off"}`}
          autoPlay
          muted
          playsInline
        />

        <div className="hud">
          <PrivacyDot active={cameraActive} />
          <button
            className="audio-toggle"
            onClick={() => setAudioEnabled((v) => !v)}
            type="button"
          >
            {audioEnabled ? "🔊 audio on" : "🔇 audio off"}
          </button>
        </div>

        <NudgeOverlay nudge={state.lastNudge} onDismiss={handleNudgeDismiss} />
      </main>

      {!isConnected && (
        <footer className="hint">
          Enter your home-server WebSocket URL above and press Connect.
          Camera turns on automatically.
        </footer>
      )}
    </div>
  );
}

function detectSurface(): SurfaceMode {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes("oculus") || ua.includes("quest")) return "quest";
  if (ua.includes("mobile") || ua.includes("iphone") || ua.includes("android")) return "mobile";
  return "desktop";
}

export default App;
