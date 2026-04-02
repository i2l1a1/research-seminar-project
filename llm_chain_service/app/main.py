from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.endpoints import router as api_router
from app.core.logging import setup_logging
from prometheus_fastapi_instrumentator import Instrumentator

def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="LLM Chain REST API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(api_router)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    return app


app = create_app()

