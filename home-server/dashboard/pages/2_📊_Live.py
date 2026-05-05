"""Live view: current ContextManager state + recent events + open risk windows."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from app.config import settings
from context.builder import build_summary
from dashboard.state import fresh_context, read_recent_events

st.set_page_config(page_title="Live", page_icon="📊", layout="wide")
st.title("📊 Live View")
st.caption("Read from the JSONL audit trail. Refresh to see the latest.")

settings.ensure_dirs()

if st.button("🔄 Refresh"):
    st.rerun()

cm = fresh_context(settings.events_dir, lookback_hours=24.0)
state = cm.state

# --- Top status row -------------------------------------------------------

c1, c2, c3 = st.columns(3)
c1.metric("Current activity", state.current_activity or "—")
c2.metric("Open risk windows", len(state.open_risk_windows))
last_med = state.last_medication_time
c3.metric(
    "Last medication",
    last_med.strftime("%H:%M") if last_med else "—",
    delta=_humanize_age(last_med) if last_med else None,
)

st.markdown("---")

# --- Recent context summary (the LLM-prompt block) ------------------------

st.subheader("Recent Context Summary")
st.caption("This is the exact block injected into every LLM prompt.")
st.code(build_summary(cm), language="text")

# --- Open risk windows ----------------------------------------------------

if state.open_risk_windows:
    st.subheader("Open risk windows")
    for w in state.open_risk_windows:
        st.warning(f"**{w.scenario}** — opened {_humanize_age(w.start_time)}")

# --- People in frame ------------------------------------------------------

if state.last_known_persons:
    st.subheader("Recently in frame")
    st.write(", ".join(state.last_known_persons))

# --- Event log ------------------------------------------------------------

st.markdown("---")
st.subheader("Recent events")
events = read_recent_events(settings.events_dir, limit=50)
if not events:
    st.info("No events recorded yet. The server will start writing here once it sees frames.")
else:
    for event in events:
        ts = event.get("timestamp", "")
        kind = event.get("event_type", "?")
        rest = {k: v for k, v in event.items() if k not in ("timestamp", "event_type")}
        st.markdown(f"`{ts}` &nbsp; **{kind}** &nbsp; `{rest}`")


# --- Helper ---------------------------------------------------------------

def _humanize_age(ts: datetime | None) -> str:
    if ts is None:
        return ""
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    minutes = int((now - ts).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60}h {minutes % 60}m ago"
