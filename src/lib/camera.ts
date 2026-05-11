// Camera helpers: getUserMedia + JPEG frame capture.

export interface CaptureResult {
  base64: string;
  width: number;
  height: number;
}

export async function startCamera(
  videoEl: HTMLVideoElement,
  constraints: MediaStreamConstraints = { video: { width: 640, height: 480 }, audio: false },
): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  videoEl.srcObject = stream;
  await videoEl.play().catch(() => {});
  return stream;
}

export function stopCamera(stream: MediaStream | null, videoEl: HTMLVideoElement | null): void {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
  if (videoEl) {
    videoEl.srcObject = null;
  }
}

export function captureFrameJpeg(
  videoEl: HTMLVideoElement,
  quality = 0.6,
): CaptureResult | null {
  if (!videoEl.videoWidth) {
    return null;
  }
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  return {
    base64: dataUrl.split(",")[1] ?? "",
    width: canvas.width,
    height: canvas.height,
  };
}
