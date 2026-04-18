from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.metrics import (
    llm_chain_latency_seconds,
    llm_chain_requests_total,
    llm_chain_stage_latency_seconds,
)
from app.services.generation.ai_answer_generate import generate_answer_text

logger = logging.getLogger(__name__)


@dataclass
class LLMChainOrchestrator:
    settings: Settings

    async def run(self, query: str) -> dict[str, Any]:
        llm_chain_requests_total.inc()
        t0 = time.perf_counter()
        final: str | None = None
        error: str | None = None
        latency_per_step: dict[str, float] | None = None

        try:
            text, latency_per_step = await generate_answer_text(question_title=query, question_text=query)
            stripped = (text or "").strip()
            final = stripped or None
            for stage, sec in (latency_per_step or {}).items():
                llm_chain_stage_latency_seconds.labels(stage=stage).observe(sec)
        except Exception:
            logger.exception("generate_answer_text failed")
            error = "generation_failed"
        finally:
            duration = time.perf_counter() - t0
            llm_chain_latency_seconds.observe(duration)

        return {
            "final": final,
            "latency_total_sec": duration,
            "latency_per_step": latency_per_step,
            "error": error,
        }
