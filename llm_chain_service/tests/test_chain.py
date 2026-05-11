import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from app.services.chain import LLMChainOrchestrator


def make_settings() -> SimpleNamespace:
    return SimpleNamespace()


def _sample_latency() -> dict[str, float]:
    return {
        "step1_classify_sec": 0.01,
        "step2_sec": 0.02,
        "step3_sec": 0.03,
        "step4_sec": 0.04,
    }


@pytest.mark.asyncio
async def test_chain_empty_query_returns_empty_final():
    with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            return_value=("", _sample_latency(), "advice"),
    ):
        orchestrator = LLMChainOrchestrator(settings=make_settings())
        result = await orchestrator.run("")
        assert result["final"] is None
        assert result["latency_per_step"] is not None


@pytest.mark.asyncio
async def test_chain_single_generation_call():
    with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            return_value=("ok", _sample_latency(), "advice"),
    ) as gen:
        orchestrator = LLMChainOrchestrator(settings=make_settings())
        await orchestrator.run("hello")
        assert gen.await_count == 1


@pytest.mark.asyncio
async def test_chain_success_has_expected_fields():
    lat = _sample_latency()
    with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            return_value=("final text", lat, "advice"),
    ):
        orchestrator = LLMChainOrchestrator(settings=make_settings())
        result = await orchestrator.run("hello")
        assert "final" in result
        assert result["final"] == "final text"
        assert "latency_total_sec" in result
        assert result["latency_per_step"] == lat


@pytest.mark.asyncio
async def test_chain_exception_is_handled(caplog):
    caplog.set_level(logging.ERROR)
    with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            side_effect=TimeoutError("LLM timeout"),
    ):
        orchestrator = LLMChainOrchestrator(settings=make_settings())
        result = await orchestrator.run("hello")
        assert result["final"] is None
        assert result["error"] == "generation_failed"
        assert result["latency_per_step"] is None
        assert any("generate_answer_text failed" in rec.message for rec in caplog.records)
