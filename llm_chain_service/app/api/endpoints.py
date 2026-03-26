from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.requests import GenerateRequest
from app.models.responses import GenerateResponse
from app.services.chain import LLMChainOrchestrator
from app.services.llm_client import OpenRouterClient


router = APIRouter(prefix="/v1")


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        client = OpenRouterClient(settings=settings)
        orchestrator = LLMChainOrchestrator(client=client, settings=settings)
        result = orchestrator.run(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run chain: {e}") from e

    final = result.get("final")
    if not final:
        raise HTTPException(status_code=500, detail="Chain failed: final answer is empty")

    return GenerateResponse(
        answer=final,
        metadata={
            "draft": result.get("draft"),
            "validated": result.get("validated"),
            "latency_per_stage": result.get("latency_per_stage"),
        },
    )

