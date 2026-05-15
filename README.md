# MCI Live Assistant

Ambient proactive-assistant prototype for smart-glasses and headset scenarios.
The current implementation focuses on an exhibition flow: onboarding a visitor,
remembering medicine, escalating an ignored stove reminder to a caregiver app,
and giving face-based conversation cues.

## Run locally

```bash
npm install --cache .npm-cache
npm run dev
```

In a second terminal:

```bash
cd home-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Desktop simulation: http://localhost:5173/?surface=desktop
- Mobile camera view: http://localhost:5173/?surface=mobile
- Quest-style overlay: http://localhost:5173/?surface=quest

The home-server API and WebSocket run at http://localhost:8000.

## Prototype behavior

- Medicine, stove, and face-cue demo beats can be triggered reliably for the
  exhibition while the live camera/audio path remains available.
- Audio nudges use a chime plus browser TTS when server TTS is unavailable.
- Stove reminders escalate to the caregiver app after repeated ignored replies.
- Voice-style commands support `done`, `remind me later`, `close this forever`,
  `wrong`, `mic off/on`, and `camera off/on`.
- A persistent privacy dot remains visible when camera or microphone state is on.
