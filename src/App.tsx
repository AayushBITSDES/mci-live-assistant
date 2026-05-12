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
import { startMicCapture, type MicCaptureHandle } from "./lib/mic";
import type { SurfaceMode } from "./lib/types";
import { useWebSocket } from "./lib/useWebSocket";

const DEFAULT_WS_URL = "ws://localhost:8000/ws/stream";
const TARGET_FPS = 5;

// Stable per-device ID, persisted in localStorage so a page reload
// keeps the same identity. Future phases (face embeddings, session
// continuity, per-device event history) will key on this.
const DEVICE_ID = (() => {
  const KEY = "mci_device_id";
  try {
    const existing = window.localStorage.getItem(KEY);
    if (existing) return existing;
    const fresh = "edge-" + Math.random().toString(36).slice(2, 10);
    window.localStorage.setItem(KEY, fresh);
    return fresh;
  } catch {
    // localStorage can throw in private mode / SSR; fall back to a
    // session-scoped ID so the app still works.
    return "edge-" + Math.random().toString(36).slice(2, 10);
  }
})();

function App() {
  const [surface, setSurface] = useState<SurfaceMode>(() => detectSurface());
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [visitorName, setVisitorName] = useState(() => {
    try {
      return window.localStorage.getItem("mci_visitor_name") ?? "";
    } catch {
      return "";
    }
  });
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [micActive, setMicActive] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastSpokenIdRef = useRef<string | null>(null);
  const micHandleRef = useRef<MicCaptureHandle | null>(null);

  const { state, connect, disconnect, send } = useWebSocket();
  const isConnected = state.status === "connected";

  // The mic capture loop fires `onChunk` from inside a setTimeout
  // callback that closes over its props at start time. Park `send` in a
  // ref so the latest closure is always reachable without rerunning the
  // start-mic effect on every render.
  const sendRef = useRef(send);
  sendRef.current = send;

  // Start/stop camera in lockstep with the connection.
  useEffect(() => {
    if (!isConnected || !cameraEnabled) {
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
  }, [isConnected, cameraEnabled]);

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

  // Speak natural assistant replies (voice conversation / face cues).
  useEffect(() => {
    if (!audioEnabled || !state.lastAssistantReply) return;
    playChimeAndSpeak(state.lastAssistantReply.sentence);
  }, [state.lastAssistantReply, audioEnabled]);

  // Backend voice tools request actual media toggles on the browser edge.
  useEffect(() => {
    const control = state.lastControl;
    if (!control) return;
    const enabled = control.action === "on";
    if (control.target === "mic") {
      setMicError(null);
      setMicActive(enabled);
    } else if (control.target === "camera") {
      setCameraEnabled(enabled);
    } else if (control.target === "audio") {
      setAudioEnabled(enabled);
    }
  }, [state.lastControl]);

  // Mic capture loop. When (connected && micActive), start rolling
  // utterance capture; each completed ~3-second blob is sent as an
  // AudioChunkMessage. Disconnect or toggle-off tears it down cleanly.
  useEffect(() => {
    if (!isConnected || !micActive) {
      const existing = micHandleRef.current;
      if (existing) {
        existing.stop();
        micHandleRef.current = null;
      }
      return;
    }

    let cancelled = false;
    let handleSampleRate = 16000;

    startMicCapture(
      (audioB64, durationMs, _mimeType) => {
        // Identity guard: if a stale capture finishes after toggle-off,
        // ignore its trailing chunk so we don't post stale audio.
        if (cancelled) return;
        sendRef.current({
          type: "audio",
          device_id: DEVICE_ID,
          timestamp: new Date().toISOString(),
          audio_b64: audioB64,
          sample_rate: handleSampleRate,
          duration_ms: Math.round(durationMs),
        });
      },
      { chunkMs: 1500, minMs: 350 },
    )
      .then((handle) => {
        if (cancelled) {
          handle.stop();
          return;
        }
        handleSampleRate = handle.sampleRate;
        micHandleRef.current = handle;
        setMicError(null);
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error ? err.message : "Could not access microphone";
        console.warn("[mic] startMicCapture failed:", err);
        setMicError(message);
        setMicActive(false);
      });

    return () => {
      cancelled = true;
      const existing = micHandleRef.current;
      if (existing) {
        existing.stop();
        micHandleRef.current = null;
      }
    };
  }, [isConnected, micActive]);

  // Push a status update whenever mic/camera state actually changes so
  // the server can audit-log toggles (and future client-controls can
  // react to mic_active without polling).
  useEffect(() => {
    if (!isConnected) return;
    sendRef.current({
      type: "status",
      device_id: DEVICE_ID,
      mic_active: micActive,
      camera_active: cameraActive,
    });
  }, [isConnected, micActive, cameraActive]);

  const handleConnect = useCallback(() => {
    connect(wsUrl, DEVICE_ID, surface);
    if (visitorName.trim()) {
      try {
        window.localStorage.setItem("mci_visitor_name", visitorName.trim());
      } catch {
        // ignore localStorage failures
      }
      void postDemoEvent(wsUrl, { event: "visitor_name", name: visitorName.trim() })
        .catch((err) => console.warn("[demo] visitor_name failed:", err));
    }
  }, [connect, wsUrl, surface, visitorName]);

  const handleDisconnect = useCallback(() => {
    disconnect();
  }, [disconnect]);

  const handleNudgeDismiss = useCallback((action: "nudge_closed" | "nudge_auto_dismiss") => {
    if (!state.lastNudge) return;
    send({
      type: "demo_action",
      device_id: DEVICE_ID,
      action,
      nudge_id: state.lastNudge.nudge_id,
      scenario: state.lastNudge.scenario,
    });
  }, [send, state.lastNudge]);

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

      <form
        className="onboarding-strip"
        onSubmit={(event) => {
          event.preventDefault();
          const cleaned = visitorName.trim();
          if (!cleaned) return;
          try {
            window.localStorage.setItem("mci_visitor_name", cleaned);
          } catch {
            // ignore localStorage failures
          }
          void postDemoEvent(wsUrl, { event: "visitor_name", name: cleaned })
            .catch((err) => console.warn("[demo] visitor_name failed:", err));
        }}
      >
        <label htmlFor="visitor-name">Visitor name</label>
        <input
          id="visitor-name"
          value={visitorName}
          onChange={(event) => setVisitorName(event.target.value)}
          placeholder="What should I call you?"
        />
        <button type="submit">Save</button>
      </form>

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
          <span className={`media-pill ${cameraEnabled ? "on" : "off"}`}>
            camera {cameraEnabled ? "on" : "off"}
          </span>
          <button
            className={`audio-toggle ${audioEnabled ? "on" : "off"}`}
            onClick={() => setAudioEnabled((v) => !v)}
            type="button"
            aria-pressed={audioEnabled}
          >
            {audioEnabled ? "🔊 audio on" : "🔇 audio off"}
          </button>
          <button
            className={`mic-toggle ${micActive ? "on" : "off"}`}
            onClick={() => {
              setMicError(null);
              setMicActive((v) => !v);
            }}
            disabled={!isConnected}
            title={
              isConnected
                ? "Toggle voice commands (mic streams 3-second utterances)"
                : "Connect first"
            }
            type="button"
            aria-pressed={micActive}
          >
            {micActive ? "🎙️ mic on" : "🎤 mic off"}
          </button>
          {micError && <span className="mic-error">{micError}</span>}
        </div>

        {state.lastAssistantReply && (
          <div className="assistant-reply" role="status" aria-live="polite">
            {state.lastAssistantReply.sentence}
          </div>
        )}

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

function postDemoEvent(wsUrl: string, payload: Record<string, unknown>): Promise<void> {
  const url = httpUrlFor(wsUrl, "/demo/operator/event");
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then((response) => {
    if (!response.ok) {
      throw new Error(`Demo event failed: ${response.status}`);
    }
  });
}

function httpUrlFor(wsUrl: string, path: string): string {
  const url = new URL(wsUrl);
  url.protocol = url.protocol === "wss:" ? "https:" : "http:";
  url.pathname = path;
  url.search = "";
  return url.toString();
}

export default App;
