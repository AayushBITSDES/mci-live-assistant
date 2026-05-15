"""Quick smoke test for the Sarvam TTS + STT round trip.

Reads the same .env the server uses, synthesizes a short sentence,
plays it through the system audio (macOS `afplay`), then sends the
generated WAV back through STT and prints the transcript. Latencies
are reported in milliseconds for each stage.

Run from home-server/:

    python3 -m scripts.sarvam_smoke
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Make sure the home-server package root is on sys.path when invoked
# directly from anywhere.
_HOME = Path(__file__).resolve().parent.parent
if str(_HOME) not in sys.path:
    sys.path.insert(0, str(_HOME))

from app.config import settings
from audio.sarvam_asr import SarvamTranscriber
from audio.sarvam_tts import SarvamSynthesizer

TEST_SENTENCE = (
    "Hello, this is Sarvam speaking. If you can hear me, the audio "
    "pipeline is alive."
)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def main() -> int:
    if not settings.sarvam_api_key:
        print("ERROR: SARVAM_API_KEY missing in .env", file=sys.stderr)
        return 1

    print("Sarvam smoke test")
    print(f"  key length     : {len(settings.sarvam_api_key)}")
    print(f"  tts model      : {settings.sarvam_tts_model}")
    print(f"  tts speaker    : {settings.sarvam_tts_speaker}")
    print(f"  stt model      : {settings.sarvam_stt_model}")
    print(f"  language code  : {settings.sarvam_language_code}")
    print()

    # --- TTS: synthesize a short sentence ---------------------------------
    tts = SarvamSynthesizer(
        api_key=settings.sarvam_api_key,
        model=settings.sarvam_tts_model,
        speaker=settings.sarvam_tts_speaker,
        language_code=settings.sarvam_language_code,
    )
    print(f"[1/3] TTS  synthesize  : sending {len(TEST_SENTENCE)} chars ...")
    t0 = time.perf_counter()
    try:
        wav_bytes = tts.synthesize(TEST_SENTENCE)
    except Exception as exc:
        print(f"      FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        tts.close()
        return 2
    tts_ms = time.perf_counter() - t0
    tts.close()

    if not wav_bytes:
        print("      FAILED: empty audio returned", file=sys.stderr)
        return 3
    print(f"      OK in {_format_ms(tts_ms)} | {len(wav_bytes)} bytes")
    print()

    # --- Playback: write to a temp file and play through the OS ----------
    out_path = Path(tempfile.gettempdir()) / "sarvam_smoke.wav"
    out_path.write_bytes(wav_bytes)
    print(f"[2/3] PLAY back        : {out_path}")
    player = shutil.which("afplay") or shutil.which("aplay") or shutil.which(
        "ffplay"
    )
    if not player:
        print("      WARN: no audio player found (afplay/aplay/ffplay)")
    else:
        play_args = [player, str(out_path)]
        if player.endswith("ffplay"):
            play_args = [player, "-autoexit", "-nodisp", str(out_path)]
        try:
            t0 = time.perf_counter()
            subprocess.run(play_args, check=False)
            print(f"      played in {_format_ms(time.perf_counter() - t0)}")
        except Exception as exc:
            print(f"      WARN: playback failed: {exc}")
    print()

    # --- STT: transcribe the WAV we just generated -----------------------
    asr = SarvamTranscriber(
        api_key=settings.sarvam_api_key,
        model=settings.sarvam_stt_model,
        language_code=settings.sarvam_language_code,
    )
    print(f"[3/3] STT  transcribe  : sending {len(wav_bytes)} bytes ...")
    t0 = time.perf_counter()
    try:
        transcript = asr.transcribe(wav_bytes, content_type="audio/wav")
    except Exception as exc:
        print(f"      FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        asr.close()
        return 4
    stt_ms = time.perf_counter() - t0
    asr.close()

    print(f"      OK in {_format_ms(stt_ms)}")
    print(f"      transcript: {transcript!r}")
    print()

    # --- Summary ----------------------------------------------------------
    print("Summary")
    print(f"  TTS latency : {_format_ms(tts_ms)}")
    print(f"  STT latency : {_format_ms(stt_ms)}")
    print(f"  Round trip  : {_format_ms(tts_ms + stt_ms)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
