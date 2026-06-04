"""
Integration tests for the Flask web app (consolidated LangGraph-only API).
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFlaskApp:
    def test_import_app(self):
        from ui.app import app
        assert app is not None

    @pytest.fixture
    def client(self):
        from ui.app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_index(self, client):
        assert client.get("/").status_code == 200

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.get_json().get("status") == "ok"

    def test_config(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        assert "default_model" in r.get_json()

    def test_stream_requires_topic(self, client):
        # Empty topic -> 400 (no LLM call made)
        r = client.post("/api/debate/stream", json={"topic": ""})
        assert r.status_code in (400, 503)

    def test_recent_debates(self, client):
        r = client.get("/api/debates")
        assert r.status_code == 200
        assert "debates" in r.get_json()

    def test_missing_debate_404(self, client):
        assert client.get("/api/debate/deadbeef").status_code == 404
