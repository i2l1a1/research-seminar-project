import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.chain import LLMChainOrchestrator


def make_settings() -> SimpleNamespace:
    return SimpleNamespace()


def test_chain_empty_query_returns_empty_final():
    async def _run():
        with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            return_value="",
        ):
            orchestrator = LLMChainOrchestrator(settings=make_settings())
            return await orchestrator.run("")

    result = asyncio.run(_run())
    assert result["final"] is None


def test_chain_single_generation_call():
    async def _run():
        with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            return_value="ok",
        ) as gen:
            orchestrator = LLMChainOrchestrator(settings=make_settings())
            await orchestrator.run("hello")
            return gen

    gen = asyncio.run(_run())
    assert gen.await_count == 1


def test_chain_success_has_expected_fields():
    async def _run():
        with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            return_value="final text",
        ):
            orchestrator = LLMChainOrchestrator(settings=make_settings())
            return await orchestrator.run("hello")

    result = asyncio.run(_run())
    assert "final" in result
    assert result["final"] == "final text"
    assert "latency_total_sec" in result


def test_chain_exception_is_handled(caplog):
    async def _run():
        with patch(
            "app.services.chain.generate_answer_text",
            new_callable=AsyncMock,
            side_effect=TimeoutError("LLM timeout"),
        ):
            orchestrator = LLMChainOrchestrator(settings=make_settings())
            caplog.set_level(logging.ERROR)
            return await orchestrator.run("hello")

    result = asyncio.run(_run())
    assert result["final"] is None
    assert result["error"] == "generation_failed"
    assert any("generate_answer_text failed" in rec.message for rec in caplog.records)
