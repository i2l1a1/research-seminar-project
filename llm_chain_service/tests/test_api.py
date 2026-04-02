import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("MODEL_STAGE1", "model_stage1")
os.environ.setdefault("MODEL_STAGE2", "model_stage2")
os.environ.setdefault("MODEL_STAGE3", "model_stage3")

from app.main import app  # noqa: E402

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
        "draft": "draft",
        "validated": "validated",
        "final": "final",
        "latency_per_stage": {"stage1_sec": 0.01, "stage2_sec": 0.02, "stage3_sec": 0.03},
    }

    with patch("app.api.endpoints.LLMChainOrchestrator.run", return_value=mocked):
        resp = client.post("/v1/generate", json={"query": "hi"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "final"
    assert body["metadata"]["draft"] == "draft"
    assert body["metadata"]["validated"] == "validated"
    assert "latency_per_stage" in body["metadata"]
