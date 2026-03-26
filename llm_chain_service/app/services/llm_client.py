from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.config import Settings


class OpenRouterClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterClient:
    settings: Settings

    def _chat(self, model: str) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            model=model,
            max_tokens=self.settings.max_tokens,
            timeout=self.settings.timeout_sec,
        )

    def generate(self, prompt: str, model: str) -> Optional[str]:
        last_err: Exception | None = None

        for attempt in range(self.settings.retries + 1):
            try:
                llm = self._chat(model=model)
                result = llm.invoke(prompt)
                return getattr(result, "content", None) or None
            except Exception as e:
                last_err = e
                if attempt >= self.settings.retries:
                    break
                time.sleep(1)

        _ = last_err
        return None

