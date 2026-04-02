from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import Settings


class OpenRouterClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterClient:
    settings: Settings

    def _chat(self, model: str) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            model=model,
            max_tokens=self.settings.max_tokens,
            timeout=self.settings.timeout_sec,
        )

    def _extract_usage(self, result: Any) -> dict[str, Any] | None:
        response_metadata = getattr(result, "response_metadata", None) or {}
        if not isinstance(response_metadata, dict):
            response_metadata = {}

        usage = (
            response_metadata.get("token_usage")
            or response_metadata.get("usage")
            or response_metadata.get("usage_metadata")
        )

        cost = response_metadata.get("cost") or response_metadata.get("total_cost")

        if usage is None and cost is None:
            return None

        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            usage = {"raw": usage}

        return {"token_usage": usage, "cost": cost, "raw": response_metadata}

    def generate_with_usage(
        self, prompt: str, model: str
    ) -> tuple[Optional[str], dict[str, Any] | None]:
        last_err: Exception | None = None

        for attempt in range(self.settings.retries + 1):
            try:
                llm = self._chat(model=model)
                result = llm.invoke(prompt)
                text = getattr(result, "content", None) or None
                usage = self._extract_usage(result)
                return text, usage
            except Exception as e:
                last_err = e
                if attempt >= self.settings.retries:
                    break
                time.sleep(min(0.25 * (2**attempt), 2.0))

        _ = last_err
        return None, None

    def generate(self, prompt: str, model: str) -> Optional[str]:
        text, _usage = self.generate_with_usage(prompt=prompt, model=model)
        return text

