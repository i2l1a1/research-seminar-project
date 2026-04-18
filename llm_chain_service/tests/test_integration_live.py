import asyncio
import os

import pytest


def _live_llm_enabled() -> bool:
    return os.environ.get("RUN_LIVE_LLM") == "1" and bool(
        os.environ.get("OPENROUTER_API_KEY", "").strip()
    )


def _require_settings_env_for_http() -> str | None:
    required = (
        "OPENROUTER_BASE_URL",
        "MODEL_STEP1",
        "MODEL_STEP2",
        "MODEL_STEP3",
        "MODEL_STEP4",
    )
    for key in required:
        if not os.environ.get(key, "").strip():
            return f"missing or empty {key} (needed for Settings when importing app.main)"
    return None


@pytest.mark.integration
def test_live_generate_answer_text_full_pipeline():
    if not _live_llm_enabled():
        pytest.skip("Set RUN_LIVE_LLM=1 and OPENROUTER_API_KEY to run live LLM tests.")

    async def _run():
        from app.services.generation.ai_answer_generate import generate_answer_text

        return await generate_answer_text(
            question_title="Проверка",
            question_text="Ответь одним коротким предложением: сколько будет 2+2?",
        )

    text, latency_per_step = asyncio.run(_run())

    assert isinstance(text, str)
    assert text.strip(), "пустой ответ от модели"
    assert "step1_classify_sec" in latency_per_step
    assert "step2_sec" in latency_per_step
    assert "step3_sec" in latency_per_step
    assert "step4_sec" in latency_per_step
    for k, sec in latency_per_step.items():
        assert isinstance(sec, (int, float)), k
        assert sec >= 0.0, k


@pytest.mark.integration
def test_live_http_v1_generate():
    if not _live_llm_enabled():
        pytest.skip("Set RUN_LIVE_LLM=1 and OPENROUTER_API_KEY to run live LLM tests.")

    skip_http = _require_settings_env_for_http()
    if skip_http:
        pytest.skip(skip_http)

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/v1/generate",
        json={"query": "Ответь одним словом: язык этого вопроса."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer"), body
    assert body["answer"].strip()
    meta = body.get("metadata") or {}
    assert meta.get("latency_total_sec") is not None
    lat = meta.get("latency_per_step")
    assert isinstance(lat, dict), meta
    assert "step1_classify_sec" in lat
