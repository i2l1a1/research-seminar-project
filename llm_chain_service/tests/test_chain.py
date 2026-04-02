import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.chain import LLMChainOrchestrator


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        model_stage1="model_stage1",
        model_stage2="model_stage2",
        model_stage3="model_stage3",
    )


def make_client() -> Mock:
    client = Mock()
    client.generate_with_usage = Mock()
    return client


def test_chain_empty_query_does_not_crash():
    client = make_client()
    client.generate_with_usage.return_value = (None, None)

    orchestrator = LLMChainOrchestrator(client=client, settings=make_settings())
    result = orchestrator.run("")

    assert result["draft"] is None
    assert result["validated"] is None
    assert result["final"] is None


def test_chain_stage2_stage3_not_called_if_stage1_fails():
    client = make_client()
    client.generate_with_usage.return_value = (None, None)  # stage1 failed -> draft=None

    orchestrator = LLMChainOrchestrator(client=client, settings=make_settings())
    orchestrator.run("hello")

    assert client.generate_with_usage.call_count == 1


def test_chain_success_has_expected_fields():
    client = make_client()
    client.generate_with_usage.side_effect = [
        ("draft text", {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}),
        ("validated text", {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}),
        ("final text", {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}),
    ]

    orchestrator = LLMChainOrchestrator(client=client, settings=make_settings())
    result = orchestrator.run("hello")

    assert set(["draft", "validated", "final"]).issubset(result.keys())
    assert result["draft"] == "draft text"
    assert result["validated"] == "validated text"
    assert result["final"] == "final text"


def test_chain_timeout_in_stage2_is_handled(caplog):
    client = make_client()

    def _side_effect(prompt: str, model: str):
        if model == "model_stage1":
            return ("draft text", {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})
        if model == "model_stage2":
            raise TimeoutError("LLM timeout")
        raise AssertionError("stage3 must not be called when stage2 fails")

    client.generate_with_usage.side_effect = _side_effect

    orchestrator = LLMChainOrchestrator(client=client, settings=make_settings())

    caplog.set_level(logging.ERROR)
    result = orchestrator.run("hello")

    assert result["draft"] == "draft text"
    assert result["validated"] is None
    assert result["final"] is None
    assert any("LLM stage2 failed" in rec.message for rec in caplog.records)
