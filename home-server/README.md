# Home Server – MCI Contextual Assistant (Hybrid)

This is the **always-on home brain** that runs on your powerful machine (4070 Ti Super, M2/M4 Pro, etc.).

It receives continuous low-resolution video + audio streams from edge devices (phone, Quest 3, Mac), runs high-quality local models (YOLOv10s + InsightFace + faster-whisper), maintains temporal memory, gates triggers, and calls Grok-4.3 only when needed for the final one-sentence nudge.

## Quick Start (Mac or PC)

```bash
cd home-server
python -m venv .venv
source .venv/bin/activate          # Mac / Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Run the API + WebSocket server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Streamlit dashboard (separate terminal)
```bash
streamlit run dashboard/app.py --server.port 8501
```

Open:
- Main API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Live monitoring page (lightweight): http://localhost:8000/live

## Audio Architecture (Server-side for quality)

- **Microphone**: Edge devices stream raw audio chunks over WebSocket (binary).
- **ASR**: `faster-whisper` on home server (small or medium model) for accurate transcription even in noisy rooms.
- **TTS**: `pyttsx3` (offline, decent) or Piper (higher quality – install separately if desired) generates the spoken nudge on server and streams audio back.
- **Voice commands**: “done”, “okay”, “remind me later”, “wrong”, “settings” are transcribed server-side and routed to the dismissal logic.
- Kill switch: Available in Streamlit dashboard and will be exposed in edge UI.

This matches the Notion Design Decisions exactly (mic always-on, TTS primary output, voice primary dismissal).

## Environment Variables (.env)
```
XAI_API_KEY=your_xai_key_here
HOME_SERVER_URL=ws://your-pc-ip:8000/ws/stream
WHISPER_MODEL=small          # tiny, small, medium, large
TTS_VOICE=en_US-amy-medium   # for Piper if used
```

## Known Relatives Face DB
Drop photos in `storage/faces/<Person Name>/` (multiple angles, lighting, with/without glasses recommended – 5–10 photos per person is enough).

The system will auto-embed on startup or via the Streamlit “Re-embed” button.

## Current Status (May 2026)
- Video + audio streaming skeleton ready
- YOLOv10s + InsightFace integration in progress
- Trigger gating + temporal event log (JSONL) planned
- Grok-4.3 integration planned
- Streamlit dashboard (face management + live view) in progress
- Fast local path for safety-critical nudges planned

See the full plan in the Notion page “Prototyping Technical Decisions”.

## Quest 3 Note
The Quest 3 browser currently requires an experimental flag for WebXR camera access in passthrough. A 30-minute spike is recommended before full testing.

## Next
Run the server, connect a simple browser client, and watch the logs. We will iterate gap-by-gap.
