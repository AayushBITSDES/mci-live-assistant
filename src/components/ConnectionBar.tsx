// Connection bar: WS URL input + connect/disconnect buttons + status pill.

import type { ConnectionState } from "../lib/useWebSocket";

export interface ConnectionBarProps {
  url: string;
  setUrl: (next: string) => void;
  state: ConnectionState;
  onConnect: () => void;
  onDisconnect: () => void;
}

export function ConnectionBar({
  url,
  setUrl,
  state,
  onConnect,
  onDisconnect,
}: ConnectionBarProps) {
  const isConnected = state.status === "connected";
  const isBusy = state.status === "connecting";

  return (
    <div className="connection-bar">
      <input
        type="text"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="ws://your-home-server:8000/ws/stream"
        disabled={isConnected || isBusy}
      />
      {!isConnected ? (
        <button onClick={onConnect} disabled={isBusy}>
          {isBusy ? "Connecting…" : "Connect"}
        </button>
      ) : (
        <button onClick={onDisconnect}>Disconnect</button>
      )}
      <StatusPill status={state.status} framesSent={state.framesSent} />
    </div>
  );
}

function StatusPill({
  status,
  framesSent,
}: {
  status: ConnectionState["status"];
  framesSent: number;
}) {
  const cls =
    status === "connected" ? "live" : status === "error" ? "fail" : "idle";
  const label =
    status === "connected"
      ? `connected · ${framesSent} frames`
      : status === "connecting"
        ? "connecting…"
        : status === "error"
          ? "error"
          : "disconnected";
  return <span className={`status-pill ${cls}`}>{label}</span>;
}
