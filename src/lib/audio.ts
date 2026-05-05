// Browser audio: chime + TTS playback.
// Two paths:
//   1. playChimeAndSpeak(text)  — fallback when no server WAV is provided
//   2. playChimeAndAudioB64(b64) — preferred when the server returns Piper TTS

export function playChimeAndSpeak(sentence: string): void {
  playChime();

  if (!("speechSynthesis" in window)) {
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(sentence);
  utterance.rate = 0.92;
  utterance.pitch = 1;
  utterance.volume = 0.9;
  window.setTimeout(() => window.speechSynthesis.speak(utterance), 260);
}

export function playChimeAndAudioB64(b64: string): void {
  playChime();

  // Server returns base64-encoded WAV bytes. Decode -> Blob -> ObjectURL -> Audio.
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);

  const audio = new Audio(url);
  audio.volume = 0.9;
  audio.addEventListener("ended", () => URL.revokeObjectURL(url));
  window.setTimeout(() => {
    audio.play().catch(() => URL.revokeObjectURL(url));
  }, 260);
}

/** One shared context per page — creating a new AudioContext per chime hits browser limits (~6). */
let sharedChimeContext: AudioContext | null = null;

function getSharedChimeContext(): AudioContext | null {
  const AudioContextClass =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextClass) {
    return null;
  }
  if (!sharedChimeContext || sharedChimeContext.state === "closed") {
    sharedChimeContext = new AudioContextClass();
  }
  return sharedChimeContext;
}

function playChime(): void {
  const audioContext = getSharedChimeContext();
  if (!audioContext) {
    return;
  }

  const run = (): void => {
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(660, audioContext.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(880, audioContext.currentTime + 0.18);

    gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.18, audioContext.currentTime + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.22);

    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.24);
  };

  if (audioContext.state === "suspended") {
    void audioContext.resume().then(run);
  } else {
    run();
  }
}
