from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.services.llm_client import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class LLMChainOrchestrator:
    client: OpenRouterClient
    settings: Settings

    def _generate_draft(self, query: str) -> str:
        prompt = (
            f"Сгенерируй черновой ответ на запрос пользователя: {query}.\n"
            f"Язык ответа должен полностью соответствовать языку запросу пользователя."
            )
        result = self.client.generate(prompt=prompt, model=self.settings.model_stage1)
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
        result = self.client.generate(prompt=prompt, model=self.settings.model_stage2)
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
        result = self.client.generate(prompt=prompt, model=self.settings.model_stage3)
        if result is None:
            raise RuntimeError("stage3 returned None")
        return result

    def run(self, query: str) -> dict[str, Any]:
        latency: dict[str, float] = {}

        draft: str | None = None
        validated: str | None = None
        final: str | None = None

        t0 = time.perf_counter()
        try:
            draft = self._generate_draft(query=query)
        except Exception:
            logger.exception("LLM stage1 failed")
        latency["stage1_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            if draft is not None:
                validated = self._validate_and_correct(draft=draft, query=query)
            else:
                validated = None
        except Exception:
            logger.exception("LLM stage2 failed")
        latency["stage2_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            if validated is not None:
                final = self._improve(corrected=validated, query=query)
            else:
                final = None
        except Exception:
            logger.exception("LLM stage3 failed")
        latency["stage3_sec"] = time.perf_counter() - t0

        return {
            "draft": draft,
            "validated": validated,
            "final": final,
            "latency_per_stage": latency,
        }
