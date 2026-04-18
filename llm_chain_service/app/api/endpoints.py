from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.requests import GenerateRequest
from app.models.responses import GenerateResponse
from app.services.chain import LLMChainOrchestrator


router = APIRouter(prefix="/v1")


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        orchestrator = LLMChainOrchestrator(settings=settings)
        result = await orchestrator.run(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run chain: {e}") from e

    final = result.get("final")
    if not final:
        raise HTTPException(status_code=500, detail="Chain failed: final answer is empty")

    return GenerateResponse(
        answer=final,
        metadata={
            "latency_total_sec": result.get("latency_total_sec"),
            "error": result.get("error"),
        },
    )
