// Wire-format types mirroring home-server/app/models.py.
// Keep this file in sync when the Pydantic models change.

export type SurfaceMode = "mobile" | "quest" | "desktop";

export type NudgePriority = "safety" | "quality";

export interface NudgeMessage {
  type: "nudge";
  nudge_id: string;
  sentence: string;
  priority: NudgePriority;
  audio_b64?: string | null;
  auto_dismiss_seconds: number;
}

export interface AckMessage {
  type: "ack";
  message: string;
  server_time: string;
}

export type ServerMessage = NudgeMessage | AckMessage;

export interface FrameMessage {
  type: "frame";
  device_id: string;
  surface: SurfaceMode;
  timestamp: string;
  image_b64: string;
  width: number;
  height: number;
}

export interface StatusMessage {
  type: "status";
  device_id: string;
  mic_active: boolean;
  camera_active: boolean;
}

export interface AudioChunkMessage {
  type: "audio";
  device_id: string;
  timestamp: string;
  audio_b64: string;        // base64-encoded webm/opus from MediaRecorder
  sample_rate: number;      // browser default is 48000 for webm/opus
  duration_ms: number;
}

export type ClientMessage =
  | FrameMessage
  | StatusMessage
  | AudioChunkMessage
  | { type: "ping" };
