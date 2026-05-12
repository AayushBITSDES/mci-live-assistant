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
  scenario?: string;
}

export interface AckMessage {
  type: "ack";
  message: string;
  server_time: string;
}

export interface VoiceCommandMessage {
  type: "voice_command";
  transcript: string;
  tool?: string | null;
  raw: string;
  provider: string;
}

export interface AssistantReplyMessage {
  type: "assistant_reply";
  sentence: string;
  ts?: number;
}

export interface EdgeControlMessage {
  type: "edge_control";
  target: "mic" | "camera" | "audio";
  action: "on" | "off";
  reason?: string;
}

export type ServerMessage =
  | NudgeMessage
  | AckMessage
  | VoiceCommandMessage
  | AssistantReplyMessage
  | EdgeControlMessage;

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
  audio_b64: string;        // base64-encoded MediaRecorder output
  sample_rate: number;      // browser default is 48000 for webm/opus
  duration_ms: number;
  // The actual MIME the recorder produced, e.g. "audio/webm;codecs=opus"
  // or "audio/mp4". The server forwards this to the ASR adapter so the
  // upload Content-Type matches the bytes; omitted only if unknown.
  mime_type?: string;
}

export interface DemoActionMessage {
  type: "demo_action";
  device_id: string;
  action: "nudge_closed" | "nudge_auto_dismiss";
  nudge_id?: string;
  scenario?: string;
}

export type ClientMessage =
  | FrameMessage
  | StatusMessage
  | AudioChunkMessage
  | DemoActionMessage
  | { type: "ping" };
