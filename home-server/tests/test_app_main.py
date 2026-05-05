from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_create_app_lifespan_initializes_context_manager(monkeypatch, tmp_path: Path) -> None:
    events_dir = tmp_path / "events"

    monkeypatch.setattr("app.main.settings.ensure_dirs", lambda: events_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr("app.main.settings.fresh_start", True)

    app = create_app()
    with TestClient(app) as client:
        assert client.app.state.context_manager is not None
