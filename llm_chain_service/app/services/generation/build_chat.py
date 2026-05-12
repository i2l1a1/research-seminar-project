import asyncio
import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

import logging
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")

_STEP_ENV_KEYS: dict[int, list[str]] = {
    1: ["MODEL_STEP1", "OPENROUTER_MODEL"],
    2: ["MODEL_STEP2", "MODEL_STEP1", "OPENROUTER_MODEL"],
    3: ["MODEL_STEP3", "MODEL_STEP2", "MODEL_STEP1", "OPENROUTER_MODEL"],
    4: ["MODEL_STEP4", "MODEL_STEP3", "MODEL_STEP2", "MODEL_STEP1", "OPENROUTER_MODEL"],
}


def _resolve_model_name(step: int) -> str:
    if step not in _STEP_ENV_KEYS:
        raise ValueError(f"step must be 1..4, got {step}")
    for key in _STEP_ENV_KEYS[step]:
        v = os.getenv(key)
        if v:
            return v
    primary = _STEP_ENV_KEYS[step][0]
    raise ValueError(
        f"No model configured for pipeline step {step}. Set {primary} (or OPENROUTER_MODEL as fallback)."
    )


def _build_chat(step: int, max_tokens: int = 512, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=_resolve_model_name(step),
        extra_body={"reasoning": {"enabled": True}},
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def timed_safe_ainvoke(
    step: int,
    messages: List[BaseMessage],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[AIMessage, float]:
    t0 = time.perf_counter()
    chat = _build_chat(step=step, max_tokens=max_tokens, temperature=temperature)
    result = await safe_ainvoke(chat, messages)
    return result, time.perf_counter() - t0


def _is_retryable_error(error_text: str) -> bool:
    text = error_text.lower()
    retryable_markers = (
        "jsondecodeerror",
        "expecting value",
        "timed out",
        "timeout",
        "operation was aborted",
        "code': 429",
        '"code": 429',
        "code': 500",
        '"code": 500',
        "code': 502",
        '"code": 502',
        "code': 503",
        '"code": 503',
        "code': 504",
        '"code": 504',
    )
    return any(marker in text for marker in retryable_markers)


async def safe_ainvoke(chat: ChatOpenAI, messages: List[BaseMessage], retries: int = 3, delay: float = 1.0):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return await chat.ainvoke(messages)
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.error("[error] [safe_ainvoke]")
            logger.error(error_str)
            if _is_retryable_error(error_str) and attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
                continue
            if not _is_retryable_error(error_str):
                raise
    if last_error is not None:
        logger.error("[error] [safe_ainvoke] retries exhausted, returning fallback answer")
    return AIMessage(content="The answer cannot be generated.")
