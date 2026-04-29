import { useEffect, useMemo, useRef, useState } from "react";
import { autoDismiss, closeForever, dismissTemporarily, evaluateNudge, flagWrong, markDone } from "./engine/nudgeEngine";
import { defaultScenarioId, scenarios } from "./engine/scenarios";
import type { AssistantSettings, NudgeDecision, ScenarioId, ScenarioRuntime, SurfaceMode } from "./engine/types";
import { playChimeAndSpeak } from "./lib/audio";
import { defaultRuntime, loadRuntime, loadSettings, saveRuntime, saveSettings } from "./lib/storage";

const scenarioList = Object.values(scenarios);

function App() {
  const [surface, setSurface] = useState<SurfaceMode>(() => getInitialSurface());
  const [settings, setSettings] = useState<AssistantSettings>(() => loadSettings());
  const [runtime, setRuntime] = useState<ScenarioRuntime>(() => {
    const stored = loadRuntime();
    return scenarios[stored.scenarioId] ? stored : defaultRuntime;
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [voiceCommand, setVoiceCommand] = useState("");
  const [now, setNow] = useState(Date.now());
  const [xrSupported, setXrSupported] = useState(false);
  const [xrStatus, setXrStatus] = useState("Quest AR is available over HTTPS in the Quest browser.");
  const spokenNudgeKey = useRef("");

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => saveSettings(settings), [settings]);
  useEffect(() => saveRuntime(runtime), [runtime]);

  useEffect(() => {
    if (!navigator.xr) {
      return;
    }

    navigator.xr
      .isSessionSupported("immersive-ar")
      .then(setXrSupported)
      .catch(() => setXrSupported(false));
  }, []);

  const scenario = scenarios[runtime.scenarioId];
  const step = scenario.steps[runtime.stepIndex];
  const decision = useMemo(() => evaluateNudge(runtime, now), [runtime, now]);
  const nudgeVisible = settings.visualEnabled && decision.shouldNudge;

  useEffect(() => {
    if (!decision.shouldNudge || !settings.audioEnabled) {
      return;
    }

    const nudgeKey = `${runtime.scenarioId}:${runtime.stepIndex}:${runtime.snoozedUntil ?? 0}`;
    if (spokenNudgeKey.current === nudgeKey) {
      return;
    }

    spokenNudgeKey.current = nudgeKey;
    playChimeAndSpeak(decision.sentence);
  }, [decision, runtime, settings.audioEnabled]);

  useEffect(() => {
    if (!nudgeVisible) {
      return;
    }

    const timeout = window.setTimeout(() => {
      setRuntime((current) => autoDismiss(current));
    }, settings.nudgeDurationSeconds * 1000);

    return () => window.clearTimeout(timeout);
  }, [nudgeVisible, settings.nudgeDurationSeconds, runtime.scenarioId, runtime.stepIndex, runtime.snoozedUntil]);

  function updateSetting<Key extends keyof AssistantSettings>(key: Key, value: AssistantSettings[Key]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function resetScenario(scenarioId: ScenarioId = runtime.scenarioId) {
    setRuntime({
      ...defaultRuntime,
      scenarioId
    });
    spokenNudgeKey.current = "";
  }

  function setScenario(scenarioId: ScenarioId) {
    resetScenario(scenarioId);
    setSurface(scenarios[scenarioId].targetSurface);
  }

  function nextStep() {
    setRuntime((current) => ({
      ...current,
      stepIndex: Math.min(scenarios[current.scenarioId].steps.length - 1, current.stepIndex + 1),
      snoozedUntil: null
    }));
  }

  function previousStep() {
    setRuntime((current) => ({
      ...current,
      stepIndex: Math.max(0, current.stepIndex - 1),
      snoozedUntil: null
    }));
  }

  function applyVoiceCommand(rawCommand: string) {
    const command = rawCommand.trim().toLowerCase();
    if (!command) {
      return;
    }

    if (["done", "completed", "i did it", "it is done"].includes(command)) {
      setRuntime((current) => markDone(current));
    } else if (command.includes("remind")) {
      setRuntime((current) => dismissTemporarily(current));
    } else if (command.includes("close") || command.includes("forever")) {
      setRuntime((current) => closeForever(current));
    } else if (command.includes("wrong") || command.includes("flag")) {
      setRuntime((current) => flagWrong(current));
    } else if (command.includes("settings")) {
      setSettingsOpen(true);
    }

    setVoiceCommand("");
  }

  async function startQuestAr() {
    if (!navigator.xr) {
      setXrStatus("WebXR is not exposed in this browser.");
      return;
    }

    try {
      await navigator.xr.requestSession("immersive-ar", {
        optionalFeatures: ["dom-overlay"],
        domOverlay: { root: document.body }
      });
      setXrStatus("Quest AR session started. The overlay remains anchored to the browser DOM.");
    } catch (error) {
      setXrStatus(error instanceof Error ? error.message : "Quest AR session could not start.");
    }
  }

  return (
    <main className={`app surface-${surface}`}>
      <SurfaceView surface={surface} settings={settings} xrSupported={xrSupported} xrStatus={xrStatus} onStartQuestAr={startQuestAr} />

      <div className="hud-layer" aria-live="polite">
        <PrivacyDot cameraActive={settings.cameraActive} micActive={settings.micActive} />

        {nudgeVisible ? (
          <NudgeOverlay
            decision={decision}
            onDone={() => setRuntime((current) => markDone(current))}
            onSnooze={() => setRuntime((current) => dismissTemporarily(current))}
            onCloseForever={() => setRuntime((current) => closeForever(current))}
            onFlagWrong={() => setRuntime((current) => flagWrong(current))}
          />
        ) : (
          <AmbientIndicator />
        )}

        {settingsOpen ? <SettingsPanel settings={settings} onChange={updateSetting} onClose={() => setSettingsOpen(false)} /> : null}
      </div>

      <aside className="control-dock" aria-label="Prototype controls">
        <div className="dock-row">
          {(["desktop", "mobile", "quest"] as SurfaceMode[]).map((mode) => (
            <button key={mode} className={surface === mode ? "selected" : ""} onClick={() => setSurface(mode)}>
              {mode}
            </button>
          ))}
        </div>

        <label className="field-label">
          Scenario
          <select value={runtime.scenarioId} onChange={(event) => setScenario(event.target.value as ScenarioId)}>
            {scenarioList.map((item) => (
              <option key={item.id} value={item.id}>
                {item.code} - {item.title}
              </option>
            ))}
          </select>
        </label>

        <section className="scenario-card">
          <div className="scenario-header">
            <span>{scenario.code}</span>
            <strong>{scenario.persona}</strong>
          </div>
          <h1>{scenario.title}</h1>
          <p>{scenario.summary}</p>
          <div className="step-meter">
            {scenario.steps.map((item, index) => (
              <button
                key={item.id}
                className={index === runtime.stepIndex ? "active" : ""}
                aria-label={`Go to step ${index + 1}: ${item.title}`}
                onClick={() => setRuntime((current) => ({ ...current, stepIndex: index, snoozedUntil: null }))}
              />
            ))}
          </div>
        </section>

        <section className="signal-card">
          <span className="eyebrow">Current signal</span>
          <h2>{step.title}</h2>
          <p>{step.ambientDetail}</p>
          <div className="risk-line">
            <span>Risk</span>
            <meter min={0} max={1} value={step.riskScore} />
            <strong>{Math.round(step.riskScore * 100)}%</strong>
          </div>
          <p className="reason">{decision.reason}</p>
        </section>

        <div className="dock-row">
          <button onClick={previousStep}>Back</button>
          <button onClick={nextStep}>Advance</button>
          <button onClick={() => resetScenario()}>Reset</button>
        </div>

        <form
          className="voice-form"
          onSubmit={(event) => {
            event.preventDefault();
            applyVoiceCommand(voiceCommand);
          }}
        >
          <label className="field-label">
            Voice-style command
            <input
              value={voiceCommand}
              placeholder="done, remind me later, settings..."
              onChange={(event) => setVoiceCommand(event.target.value)}
            />
          </label>
          <button type="submit">Apply</button>
        </form>

        <button className="settings-button" onClick={() => setSettingsOpen(true)}>
          Settings
        </button>
      </aside>
    </main>
  );
}

function SurfaceView({
  surface,
  settings,
  xrSupported,
  xrStatus,
  onStartQuestAr
}: {
  surface: SurfaceMode;
  settings: AssistantSettings;
  xrSupported: boolean;
  xrStatus: string;
  onStartQuestAr: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    if (surface !== "mobile" || !settings.cameraActive || !navigator.mediaDevices?.getUserMedia) {
      return;
    }

    let stream: MediaStream | null = null;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((nextStream) => {
        stream = nextStream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((error) => setCameraError(error instanceof Error ? error.message : "Camera could not start."));

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [surface, settings.cameraActive]);

  if (surface === "mobile") {
    return (
      <section className="surface-view mobile-view" aria-label="Mobile camera view">
        {settings.cameraActive ? <video ref={videoRef} autoPlay playsInline muted /> : <div className="camera-off">Camera is off.</div>}
        {cameraError ? <p className="camera-error">{cameraError}</p> : null}
      </section>
    );
  }

  if (surface === "quest") {
    return (
      <section className="surface-view quest-view" aria-label="Quest passthrough simulation">
        <div className="room-horizon" />
        <div className="counter-line" />
        <button className="xr-button" disabled={!xrSupported} onClick={onStartQuestAr}>
          Start Quest AR
        </button>
        <p>{xrStatus}</p>
      </section>
    );
  }

  return (
    <section className="surface-view desktop-view" aria-label="Desktop simulation">
      <div className="calendar-strip">
        <span>09:18</span>
        <strong>Blood test at 10:00am</strong>
      </div>
      <div className="kitchen-scene">
        <div className="window-panel" />
        <div className="tea-cup" />
        <div className="cabinet" />
      </div>
    </section>
  );
}

function NudgeOverlay({
  decision,
  onDone,
  onSnooze,
  onCloseForever,
  onFlagWrong
}: {
  decision: NudgeDecision;
  onDone: () => void;
  onSnooze: () => void;
  onCloseForever: () => void;
  onFlagWrong: () => void;
}) {
  return (
    <section className={`nudge-overlay priority-${decision.priority}`}>
      <p>{decision.sentence}</p>
      <div className="nudge-actions">
        <button onClick={onDone}>Done</button>
        <button onClick={onSnooze}>Later</button>
        <button onClick={onFlagWrong}>Wrong</button>
        <button onClick={onCloseForever}>Close</button>
      </div>
    </section>
  );
}

function PrivacyDot({ cameraActive, micActive }: { cameraActive: boolean; micActive: boolean }) {
  const active = cameraActive || micActive;
  return (
    <div className={`privacy-dot ${active ? "active" : "inactive"}`} title={active ? "Camera or microphone active" : "Camera and microphone off"}>
      <span />
    </div>
  );
}

function AmbientIndicator() {
  return (
    <div className="ambient-indicator" aria-label="Assistant active and silent">
      <span />
    </div>
  );
}

function SettingsPanel({
  settings,
  onChange,
  onClose
}: {
  settings: AssistantSettings;
  onChange: <Key extends keyof AssistantSettings>(key: Key, value: AssistantSettings[Key]) => void;
  onClose: () => void;
}) {
  return (
    <section className="settings-panel">
      <header>
        <h2>Settings</h2>
        <button onClick={onClose}>Close</button>
      </header>

      <label className="toggle-row">
        <span>Visual nudges</span>
        <input type="checkbox" checked={settings.visualEnabled} onChange={(event) => onChange("visualEnabled", event.target.checked)} />
      </label>
      <label className="toggle-row">
        <span>Audio nudges</span>
        <input type="checkbox" checked={settings.audioEnabled} onChange={(event) => onChange("audioEnabled", event.target.checked)} />
      </label>
      <label className="toggle-row">
        <span>Camera</span>
        <input type="checkbox" checked={settings.cameraActive} onChange={(event) => onChange("cameraActive", event.target.checked)} />
      </label>
      <label className="toggle-row">
        <span>Microphone</span>
        <input type="checkbox" checked={settings.micActive} onChange={(event) => onChange("micActive", event.target.checked)} />
      </label>
      <label className="field-label">
        Auto-dismiss seconds
        <input
          type="number"
          min={3}
          max={30}
          value={settings.nudgeDurationSeconds}
          onChange={(event) => onChange("nudgeDurationSeconds", Number(event.target.value))}
        />
      </label>
    </section>
  );
}

function getInitialSurface(): SurfaceMode {
  const requested = new URLSearchParams(window.location.search).get("surface");
  if (requested === "quest" || requested === "mobile" || requested === "desktop") {
    return requested;
  }

  if (/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)) {
    return "mobile";
  }

  return "desktop";
}

export default App;
