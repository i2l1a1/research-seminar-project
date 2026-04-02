from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.metrics import (
    llm_chain_estimated_cost,
    llm_chain_latency_seconds,
    llm_chain_requests_total,
    llm_chain_stage_latency_seconds,
)
from app.services.llm_client import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class LLMChainOrchestrator:
    client: OpenRouterClient
    settings: Settings
    _last_stage_usage: dict[str, Any] | None = None

    def _extract_stage_tokens(self) -> dict[str, Any] | None:
        usage = self._last_stage_usage or {}
        token_usage = usage.get("token_usage")
        if not isinstance(token_usage, dict) or not token_usage:
            return None

        prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
        completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
        total_tokens = token_usage.get("total_tokens") or token_usage.get("total")

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": usage.get("cost"),
        }

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _generate_draft(self, query: str) -> str:
        prompt = f"Сгенерируй черновой ответ на запрос пользователя: {query}.\n"
        f"Язык ответа должен полностью соответствовать языку запросу пользователя."
        self._last_stage_usage = None
        result, usage = self.client.generate_with_usage(
            prompt=prompt, model=self.settings.model_stage1
        )
        self._last_stage_usage = usage
        if result is None:
            raise RuntimeError("stage1 returned None")
        return result

    def _validate_and_correct(self, draft: str, query: str) -> str:
        prompt = (
            "Ты валидатор. Проверь черновик ответа на соответствие запросу, "
            "фактические ошибки, стиль, безопасность. Если есть проблемы - исправь. "
            f"Исходный запрос: {query}\n"
            f"Черновик: {draft}\n"
            f"Язык ответа должен полностью соответствовать языку запросу пользователя."
            "Исправленный вариант:"
        )
        self._last_stage_usage = None
        result, usage = self.client.generate_with_usage(
            prompt=prompt, model=self.settings.model_stage2
        )
        self._last_stage_usage = usage
        if result is None:
            raise RuntimeError("stage2 returned None")
        return result

    def _improve(self, corrected: str, query: str) -> str:
        prompt = (
            "Улучши ответ: сделай более структурированным и логичным. "
            f"Исходный запрос: {query}\n"
            f"Если в исходном запросе требовалось дать подробный ответ, сделай итоговый ответ подробным. "
            f"Если короткий, то дай короткий ответ."
            f"Язык ответа должен полностью соответствовать языку запросу пользователя."
            f"Ответ: {corrected}"
        )
        self._last_stage_usage = None
        result, usage = self.client.generate_with_usage(
            prompt=prompt, model=self.settings.model_stage3
        )
        self._last_stage_usage = usage
        if result is None:
            raise RuntimeError("stage3 returned None")
        return result

    def run(self, query: str) -> dict[str, Any]:
        latency: dict[str, float] = {}
        tokens_per_stage: dict[str, Any] = {}
        estimated_cost_total: float | None = None

        draft: str | None = None
        validated: str | None = None
        final: str | None = None

        llm_chain_requests_total.inc()
        t_total0 = time.perf_counter()

        try:
            # stage1
            stage = "stage1"
            model = self.settings.model_stage1
            logger.info("chain_stage_start", extra={"stage": stage, "model": model})
            t0 = time.perf_counter()
            try:
                draft = self._generate_draft(query=query)
                tokens_per_stage[stage] = self._extract_stage_tokens()
            except Exception:
                logger.exception("LLM stage1 failed")
            finally:
                duration = time.perf_counter() - t0
                latency[f"{stage}_sec"] = duration
                llm_chain_stage_latency_seconds.labels(stage=stage).observe(duration)
                logger.info(
                    "chain_stage_end",
                    extra={
                        "stage": stage,
                        "model": model,
                        "duration_sec": duration,
                        "tokens": tokens_per_stage.get(stage),
                    },
                )

            # stage2
            stage = "stage2"
            model = self.settings.model_stage2
            logger.info("chain_stage_start", extra={"stage": stage, "model": model})
            t0 = time.perf_counter()
            try:
                if draft is not None:
                    validated = self._validate_and_correct(draft=draft, query=query)
                    tokens_per_stage[stage] = self._extract_stage_tokens()
            except Exception:
                logger.exception("LLM stage2 failed")
            finally:
                duration = time.perf_counter() - t0
                latency[f"{stage}_sec"] = duration
                llm_chain_stage_latency_seconds.labels(stage=stage).observe(duration)
                logger.info(
                    "chain_stage_end",
                    extra={
                        "stage": stage,
                        "model": model,
                        "duration_sec": duration,
                        "tokens": tokens_per_stage.get(stage),
                    },
                )

            # stage3
            stage = "stage3"
            model = self.settings.model_stage3
            logger.info("chain_stage_start", extra={"stage": stage, "model": model})
            t0 = time.perf_counter()
            try:
                if validated is not None:
                    final = self._improve(corrected=validated, query=query)
                    tokens_per_stage[stage] = self._extract_stage_tokens()
            except Exception:
                logger.exception("LLM stage3 failed")
            finally:
                duration = time.perf_counter() - t0
                latency[f"{stage}_sec"] = duration
                llm_chain_stage_latency_seconds.labels(stage=stage).observe(duration)
                logger.info(
                    "chain_stage_end",
                    extra={
                        "stage": stage,
                        "model": model,
                        "duration_sec": duration,
                        "tokens": tokens_per_stage.get(stage),
                    },
                )
        finally:
            total_duration = time.perf_counter() - t_total0
            llm_chain_latency_seconds.observe(total_duration)

            cost_total = 0.0
            has_cost = False
            for stage_tokens in tokens_per_stage.values():
                if not stage_tokens:
                    continue
                stage_cost = self._to_float(stage_tokens.get("cost"))
                if stage_cost is None:
                    continue
                cost_total += stage_cost
                has_cost = True

            if has_cost:
                estimated_cost_total = cost_total
                llm_chain_estimated_cost.inc(estimated_cost_total)

        return {
            "draft": draft,
            "validated": validated,
            "final": final,
            "latency_per_stage": latency,
            "tokens_per_stage": tokens_per_stage,
            "estimated_cost_total": estimated_cost_total,
        }
