"""Runtime settings: LLM provider, FPS, thresholds, fresh start."""
from __future__ import annotations

from datetime import date, datetime, timezone

import streamlit as st

from app.config import settings
from dashboard.persistence import RuntimeSettings, load_runtime_settings, save_runtime_settings

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")
st.caption("These override the .env defaults. The server applies them on next restart.")

settings.ensure_dirs()
path = settings.storage_root / "runtime_settings.json"
runtime = load_runtime_settings(path)

# --- LLM provider ---------------------------------------------------------

st.subheader("LLM provider")
provider = st.radio(
    "Active provider",
    options=["grok", "gemini"],
    index=0 if runtime.active_llm_provider == "grok" else 1,
    horizontal=True,
    captions=[
        "Grok-4.3 via xAI (api.x.ai/v1)",
        "Gemini 3 Flash via Google AI Studio",
    ],
)

# --- FPS ------------------------------------------------------------------

st.markdown("---")
st.subheader("Performance")
fps = st.slider(
    "Target FPS (frame rate from edge)",
    min_value=1, max_value=15, value=runtime.target_fps,
    help="5 FPS is the design default. Higher means more frames to process and more LLM calls.",
)

# --- Thresholds -----------------------------------------------------------

st.markdown("---")
st.subheader("Detection thresholds")
yolo_conf = st.slider(
    "YOLO confidence threshold",
    min_value=0.1, max_value=0.95, value=float(runtime.yolo_confidence_threshold), step=0.05,
    help="Minimum confidence for YOLO to keep a detection. Higher = fewer false positives, more misses.",
)
face_sim = st.slider(
    "Face similarity threshold",
    min_value=0.30, max_value=0.80, value=float(runtime.face_similarity_threshold), step=0.05,
    help="Cosine similarity required to recognise a face. Buffalo_l: 0.45 is a typical starting point.",
)

# --- Demo behaviour -------------------------------------------------------

st.markdown("---")
st.subheader("Demo behaviour")
fresh = st.checkbox(
    "Fresh start on every server restart",
    value=runtime.fresh_start,
    help=(
        "If on, the ContextManager skips JSONL replay on startup. "
        "Every demo session begins with a clean slate. Recommended for short demos."
    ),
)

# --- Wipe ----------------------------------------------------------------

st.markdown("---")
st.subheader("Danger zone")
if st.button("🧹 Wipe all events (today + yesterday)"):
    # Filenames are YYYY-MM-DD.jsonl (see context/event_log.py) — only remove
    # the last two days, not the entire history.
    today = datetime.now(timezone.utc).date()
    yesterday = date.fromordinal(today.toordinal() - 1)
    targets = {f"{today.isoformat()}.jsonl", f"{yesterday.isoformat()}.jsonl"}
    wiped = 0
    for name in targets:
        path = settings.events_dir / name
        if path.is_file():
            path.unlink()
            wiped += 1
    st.success(f"Wiped {wiped} event log file(s).")

# --- Save -----------------------------------------------------------------

st.markdown("---")
if st.button("💾 Save settings"):
    new_settings = RuntimeSettings(
        active_llm_provider=provider,
        yolo_confidence_threshold=yolo_conf,
        face_similarity_threshold=face_sim,
        target_fps=fps,
        fresh_start=fresh,
    )
    save_runtime_settings(new_settings, path)
    st.success("Saved. Restart the FastAPI server to apply.")
