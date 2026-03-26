from typing import Any

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    answer: str
    metadata: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str

