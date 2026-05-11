// Custom hook: connect/disconnect, send messages, expose latest nudge.

import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientMessage, NudgeMessage, ServerMessage, SurfaceMode } from "./types";

export interface ConnectionState {
  status: "disconnected" | "connecting" | "connected" | "error";
  lastNudge: NudgeMessage | null;
  framesSent: number;
}

export interface WebSocketHandle {
  state: ConnectionState;
  connect: (url: string, deviceId: string, surface: SurfaceMode) => void;
  disconnect: () => void;
  send: (msg: ClientMessage) => void;
}

export function useWebSocket(): WebSocketHandle {
  const wsRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<ConnectionState>({
    status: "disconnected",
    lastNudge: null,
    framesSent: 0,
  });

  const connect = useCallback((url: string, deviceId: string, surface: SurfaceMode) => {
    // Skip only if a usable socket exists. A socket in CLOSING/OPEN/CONNECTING
    // is "live"; CLOSED means we're free to start fresh. Without the CLOSING
    // check, rapid Disconnect → Connect silently no-ops while the old socket
    // is still tearing down.
    const existing = wsRef.current;
    if (
      existing &&
      existing.readyState !== WebSocket.CLOSED &&
      existing.readyState !== WebSocket.CLOSING
    ) {
      return;
    }

    const sep = url.includes("?") ? "&" : "?";
    const fullUrl = `${url}${sep}device_id=${encodeURIComponent(deviceId)}&surface=${surface}`;

    setState((s) => ({ ...s, status: "connecting" }));
    const ws = new WebSocket(fullUrl);
    wsRef.current = ws;

    // Each handler captures `ws` and only mutates shared state when it
    // is still the active socket. Without these identity guards, the old
    // socket's late-firing `close` event would clobber the new socket's
    // `connecting` / `connected` state during a rapid reconnect.
    ws.addEventListener("open", () => {
      if (wsRef.current !== ws) return;
      setState((s) => ({ ...s, status: "connected", framesSent: 0 }));
    });

    ws.addEventListener("close", () => {
      if (wsRef.current !== ws) return;
      setState((s) => ({ ...s, status: "disconnected" }));
      wsRef.current = null;
    });

    ws.addEventListener("error", () => {
      if (wsRef.current !== ws) return;
      setState((s) => ({ ...s, status: "error" }));
    });

    ws.addEventListener("message", (ev) => {
      if (wsRef.current !== ws) return;
      try {
        const payload = JSON.parse(ev.data) as ServerMessage;
        if (payload.type === "nudge") {
          setState((s) => ({ ...s, lastNudge: payload }));
        }
      } catch {
        // ignore malformed messages
      }
    });
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  const send = useCallback((msg: ClientMessage) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(msg));
    if (msg.type === "frame") {
      setState((s) => ({ ...s, framesSent: s.framesSent + 1 }));
    }
  }, []);

  useEffect(
    () => () => {
      if (wsRef.current) wsRef.current.close();
    },
    [],
  );

  return { state, connect, disconnect, send };
}
