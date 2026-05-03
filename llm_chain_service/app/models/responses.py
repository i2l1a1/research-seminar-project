from typing import Any, Optional

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    answer: str
    metadata: dict[str, Any] | None = None
    quality_scores: Optional[dict[str, int]] = None


class ErrorResponse(BaseModel):
    detail: str

