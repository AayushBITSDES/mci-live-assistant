// Continuous-mic capture for voice commands.
//
// Pattern: rolling short utterances. We start a MediaRecorder, let it
// run for N ms, stop it, and start a fresh one — each completed recorder
// produces a STANDALONE webm blob (with its own header) that the server
// can decode in isolation. The cost is a brief ~50ms gap between
// utterances; the upside is each chunk is independently transcribable.
//
// Why not `start(timeslice)` ? MediaRecorder.start(N) emits a
// `dataavailable` event every N ms, but those events are SLICES of one
// continuous stream — chunk #2 has no webm header and can't be decoded
// alone. The rolling-restart approach gives standalone blobs at the
// cost of a tiny gap.
//
// Why not VAD ? Browser VAD is doable (Web Audio analyser + RMS
// threshold) but adds tuning risk. For the exhibition, short fixed
// windows keep command latency predictable.

export type MicChunkHandler = (
  audioB64: string,
  durationMs: number,
  mimeType: string,
) => void;

export interface MicCaptureOptions {
  /** Length of each utterance window in ms (default 1500). */
  chunkMs?: number;
  /** Skip blobs shorter than this many ms (default 500). */
  minMs?: number;
  /** Skip blobs smaller than this many bytes (default 800).
   *  A 200ms webm blob is ~3KB; 800 is a safe "empty header" cutoff. */
  minBytes?: number;
}

export interface MicCaptureHandle {
  stop: () => void;
  /** The browser-chosen mime type; useful for logging. */
  mimeType: string;
  /** The MediaStream's nominal sample rate (usually 48000 for webm/opus). */
  sampleRate: number;
}

/**
 * Start rolling-utterance capture. Returns a handle once the mic
 * permission is granted and the first recorder is running.
 *
 * Throws if `navigator.mediaDevices.getUserMedia` rejects (permission
 * denied, no mic, secure-context required). Callers should catch and
 * surface a UI hint.
 */
export async function startMicCapture(
  onChunk: MicChunkHandler,
  opts: MicCaptureOptions = {},
): Promise<MicCaptureHandle> {
  const chunkMs = opts.chunkMs ?? 1500;
  const minMs = opts.minMs ?? 500;
  const minBytes = opts.minBytes ?? 800;

  // 16k mono is what Whisper wants; the browser may not honour these
  // hints (Chrome ignores sampleRate for webm/opus) but they're free to
  // ask for and they help on platforms that DO honour them.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  const mimeType = pickMimeType();
  const sampleRate = stream.getAudioTracks()[0]?.getSettings().sampleRate ?? 16000;

  let stopped = false;
  let currentRecorder: MediaRecorder | null = null;
  let cycleTimer: number | null = null;

  const startRecorder = (): void => {
    if (stopped) return;

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      );
    } catch (err) {
      // Unsupported mimeType on this browser, or stream already closed
      // (race vs. handle.stop()). Either way, give up cleanly.
      console.warn("[mic] MediaRecorder construction failed:", err);
      return;
    }
    currentRecorder = recorder;

    const chunks: Blob[] = [];
    const startedAt = Date.now();

    recorder.addEventListener("dataavailable", (event: BlobEvent) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    });

    recorder.addEventListener("stop", () => {
      if (currentRecorder === recorder) currentRecorder = null;
      const durationMs = Date.now() - startedAt;
      const blob = new Blob(chunks, { type: recorder.mimeType });

      // Drop blobs that are too short or too small — they're usually
      // just the webm header on a chunk the user cancelled mid-flight.
      if (durationMs >= minMs && blob.size >= minBytes) {
        void blobToBase64(blob)
          .then((b64) => onChunk(b64, durationMs, recorder.mimeType))
          .catch((err) => console.warn("[mic] base64 encode failed:", err));
      }

      if (!stopped) startRecorder();
    });

    try {
      recorder.start();
    } catch (err) {
      console.warn("[mic] recorder.start() failed:", err);
      return;
    }

    cycleTimer = window.setTimeout(() => {
      if (recorder.state === "recording") {
        try {
          recorder.stop();
        } catch {
          // Recorder may already be stopped; the "stop" event still fires.
        }
      }
    }, chunkMs);
  };

  startRecorder();

  return {
    mimeType,
    sampleRate,
    stop: () => {
      if (stopped) return;
      stopped = true;
      if (cycleTimer !== null) {
        window.clearTimeout(cycleTimer);
        cycleTimer = null;
      }
      const recorder = currentRecorder;
      if (recorder && recorder.state === "recording") {
        try {
          recorder.stop();      // flushes final chunk via the "stop" handler
        } catch {
          // ignored
        }
      }
      stream.getTracks().forEach((t) => t.stop());
    },
  };
}

// --- Helpers --------------------------------------------------------------

/**
 * Pick the best supported mime type. Order matters: opus is smaller and
 * universally decodable on the server; mp4/aac is a Safari fallback.
 * Empty string means "let the browser choose" — last-resort fallback.
 */
function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const candidate of candidates) {
    if (MediaRecorder.isTypeSupported(candidate)) return candidate;
  }
  return "";
}

/** Convert a Blob to base64 (sans the `data:...;base64,` prefix). */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("FileReader did not produce a string"));
        return;
      }
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error ?? new Error("FileReader error"));
    reader.readAsDataURL(blob);
  });
}
