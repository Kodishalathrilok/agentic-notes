"""API tests via FastAPI TestClient — only routes that don't call the model."""

from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # Every provider the router can report. This list had fallen behind
    # get_active_provider() once already, and only started failing when a
    # deployment actually configured one of the newer ones.
    assert body["provider"] in ("nvidia", "gemini", "ollama")
    assert body["max_text_chars"] > 0


def test_models_listed():
    r = client.get("/api/models")
    assert r.status_code == 200
    assert len(r.json()["models"]) >= 1


def test_generate_rejects_empty_text():
    r = client.post("/api/generate", json={"text": "   "})
    assert r.status_code == 422


def test_generate_rejects_oversized_text():
    r = client.post("/api/generate", json={"text": "x" * 9_999_999})
    assert r.status_code == 422


def test_export_markdown_works_offline():
    r = client.post(
        "/api/export/markdown",
        json={"notes": "**Hi:**\n• point", "quiz": "", "flashcards": ""},
    )
    assert r.status_code == 200
    assert "AI Generated Notes" in r.text


def test_export_flashcards_csv():
    r = client.post(
        "/api/export/flashcards-csv",
        json={"notes": "", "quiz": "", "flashcards": "CARD 1\nFront: Q\nBack: A"},
    )
    assert r.status_code == 200
    assert "Q" in r.text and "A" in r.text
