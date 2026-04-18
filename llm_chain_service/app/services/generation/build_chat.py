import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from typing import List
import asyncio

load_dotenv(dotenv_path=".env")


def _resolve_model_name() -> str:
    # Prefer explicitly configured generation model, then chain stage models.
    model = (
        os.getenv("OPENROUTER_MODEL")
        or os.getenv("MODEL_STAGE1")
        or os.getenv("MODEL_STAGE2")
        or os.getenv("MODEL_STAGE3")
    )
    return model


def _build_chat(max_tokens=512, temperature=0.0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=_resolve_model_name(),
        extra_body={"reasoning": {"enabled": True}},
        max_tokens=max_tokens,
        temperature=temperature
    )


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
            print("[error] [safe_ainvoke]")
            print(error_str)
            if _is_retryable_error(error_str) and attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
                continue
            if not _is_retryable_error(error_str):
                raise
    if last_error is not None:
        print("[error] [safe_ainvoke] retries exhausted, returning fallback answer")
    return AIMessage(content="The answer cannot be generated.")
