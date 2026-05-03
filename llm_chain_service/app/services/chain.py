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
    llm_answer_accuracy,
    llm_answer_relevance,
    llm_answer_completeness,
    llm_answer_conciseness,
    llm_answer_coherence,
    llm_answer_style,
    llm_quality_evaluations_total,
)
from app.services.generation.ai_answer_generate import generate_answer_text
from app.services.quality_metrics import evaluate_answer

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
        quality_scores: dict[str, int] | None = None

        try:
            text, latency_per_step = await generate_answer_text(question_title=query, question_text=query)
            stripped = (text or "").strip()
            final = stripped or None
            for stage, sec in (latency_per_step or {}).items():
                llm_chain_stage_latency_seconds.labels(stage=stage).observe(sec)

            if final:
                scores = await evaluate_answer(query, final, model_step=5)
                if scores:
                    quality_scores = scores
                    llm_answer_accuracy.observe(scores["accuracy"])
                    llm_answer_relevance.observe(scores["relevance"])
                    llm_answer_completeness.observe(scores["completeness"])
                    llm_answer_conciseness.observe(scores["conciseness"])
                    llm_answer_coherence.observe(scores["coherence"])
                    llm_answer_style.observe(scores["style"])
                    llm_quality_evaluations_total.labels(status="success").inc()
                else:
                    llm_quality_evaluations_total.labels(status="failure").inc()
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
            "quality_scores": quality_scores,
        }
