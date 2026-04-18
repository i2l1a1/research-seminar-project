import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
os.environ.setdefault("MODEL_STEP1", "test-model-step1")
os.environ.setdefault("MODEL_STEP2", "test-model-step2")
os.environ.setdefault("MODEL_STEP3", "test-model-step3")
os.environ.setdefault("MODEL_STEP4", "test-model-step4")

from app.main import app

client = TestClient(app)


def test_get_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_generate_invalid_json_returns_422():
    resp = client.post(
        "/v1/generate",
        content='{"query": ',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_post_generate_success_returns_chain_json():
    mocked = {
        "final": "final",
        "latency_total_sec": 0.05,
        "latency_per_step": {
            "step1_classify_sec": 0.01,
            "step2_sec": 0.02,
            "step3_sec": 0.01,
            "step4_sec": 0.01,
        },
        "error": None,
    }

    with patch(
        "app.api.endpoints.LLMChainOrchestrator.run",
        new_callable=AsyncMock,
        return_value=mocked,
    ):
        resp = client.post("/v1/generate", json={"query": "hi"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "final"
    assert body["metadata"]["latency_total_sec"] == 0.05
    assert body["metadata"]["latency_per_step"]["step1_classify_sec"] == 0.01
    assert body["metadata"]["error"] is None
