"""Face manager: drag-drop photos to register relatives, run re-embed."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import settings
from dashboard.state import list_known_faces

# Lazy-import FaceRecognizer only when re-embed is clicked, so the
# dashboard loads even without insightface installed.

st.set_page_config(page_title="Faces", page_icon="📷", layout="wide")
st.title("📷 Known Faces")
st.caption("Drop 5–10 photos per person. Multiple angles + lighting recommended.")

settings.ensure_dirs()


def _delete_person(faces_dir: Path, name: str) -> None:
    folder = faces_dir / name
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()
        folder.rmdir()
    st.success(f"Removed {name}")


# --- Existing faces overview ---------------------------------------------

known = list_known_faces(settings.faces_dir)

col_a, col_b = st.columns([2, 1])
with col_a:
    if not known:
        st.info("No faces registered yet. Add one below.")
    else:
        st.subheader("Registered relatives")
        for name, count in sorted(known.items()):
            row = st.columns([3, 1, 1])
            row[0].markdown(f"**{name}**")
            row[1].markdown(f"{count} photo{'s' if count != 1 else ''}")
            if row[2].button("🗑️ Delete", key=f"del-{name}"):
                _delete_person(settings.faces_dir, name)
                st.rerun()


# --- Add a new person -----------------------------------------------------

with col_b:
    st.subheader("Add a person")
    new_name = st.text_input("Name", placeholder="e.g. Anjali")
    uploads = st.file_uploader(
        "Photos (5–10 recommended)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "webp"],
    )

    if st.button("Save photos", disabled=not (new_name and uploads)):
        target_dir = settings.faces_dir / new_name.strip()
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(target_dir.glob("*")))
        for i, file in enumerate(uploads, start=existing + 1):
            (target_dir / f"photo_{i:02d}.{file.name.split('.')[-1]}").write_bytes(file.getvalue())
        st.success(f"Saved {len(uploads)} photo(s) for {new_name}")
        st.rerun()


# --- Re-embed -------------------------------------------------------------

st.markdown("---")
st.subheader("Re-embed all photos")
st.caption(
    "Runs InsightFace over every photo and updates `storage/faces/embeddings.json`. "
    "Takes a few seconds per face on a GPU machine."
)

if st.button("🔄 Re-embed now"):
    try:
        from detection.faces import FaceRecognizer

        store_path = settings.faces_dir / "embeddings.json"
        rec = FaceRecognizer(store_path=store_path)

        total = 0
        progress = st.progress(0.0)
        names = list(known.keys())
        for i, name in enumerate(names):
            photos = sorted((settings.faces_dir / name).glob("*"))
            photos = [p for p in photos if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            added = rec.add_face(name, photos)
            total += added
            progress.progress((i + 1) / max(len(names), 1))

        st.success(f"Embedded {total} photos across {len(names)} people.")
    except ImportError:
        st.error("InsightFace is not installed on this machine. Run on the home-server box.")
    except Exception as exc:                     # noqa: BLE001
        st.error(f"Re-embed failed: {exc}")
