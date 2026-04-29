# MCI Live Assistant

Ambient proactive-assistant prototype for smart-glasses scenarios. The current
implementation focuses on **B2: Shanta has a 10am blood test and forgets the
appointment during her morning routine**. **B1: Sanjay needs to send a client
deck before a 3pm meeting** is included as a shared scenario shell.

## Run locally

```bash
npm install --cache .npm-cache
npm run api
npm run dev
```

Open:

- Desktop simulation: http://localhost:5173/?surface=desktop
- Mobile camera view: http://localhost:5173/?surface=mobile
- Quest-style overlay: http://localhost:5173/?surface=quest

The API runs at http://localhost:8787.

## Prototype behavior

- One full-sentence visual nudge appears when the B2 risk window becomes active.
- Audio nudges use a chime plus browser TTS when enabled.
- Nudges auto-dismiss after the configured duration and return if unresolved.
- Voice-style commands support `done`, `remind me later`, `close this forever`,
  `wrong`, and `settings`.
- A persistent privacy dot remains visible when camera or microphone state is on.
