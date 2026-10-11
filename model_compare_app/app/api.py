"""FastAPI endpoints for direct comparison of the three trained models."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .model_manager import GenerationSettings, ModelManager


class GenerateRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    temperature: float = Field(default=0.7, gt=0, le=5)
    max_new_tokens: int = Field(default=100, ge=1, le=1023)
    top_k: int = Field(default=150, ge=1, le=50_257)
    seed: int | None = Field(default=1337)


class ModelResultResponse(BaseModel):
    model: str
    label: str
    text: str
    generated_tokens: int
    duration_ms: float


class GenerateResponse(BaseModel):
    results: list[ModelResultResponse]


@lru_cache(maxsize=1)
def get_manager() -> ModelManager:
    settings = load_settings()
    return ModelManager(
        scratch_checkpoint=settings.scratch_checkpoint,
        continued_checkpoint=settings.continued_checkpoint,
        lora_checkpoint=settings.lora_checkpoint,
        tokenizer_path=settings.tokenizer_path,
        device=settings.device,
        pretrained_model=settings.pretrained_model,
    )


def create_app(manager: ModelManager | None = None) -> FastAPI:
    """Create the API, optionally with an injected manager for tests."""

    app = FastAPI(title="Direct Model Comparison API", version="1.0.0")
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    active_manager = manager

    def resolve_manager() -> ModelManager:
        nonlocal active_manager
        if active_manager is None:
            active_manager = get_manager()
        return active_manager

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "models_loaded": active_manager is not None}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        settings = GenerationSettings(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            seed=request.seed,
        )
        try:
            results = resolve_manager().generate_all(request.message, settings)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return GenerateResponse(
            results=[
                ModelResultResponse(**asdict(result))
                for result in results
            ]
        )

    @app.post("/generate/stream", include_in_schema=True)
    def generate_stream(request: GenerateRequest) -> StreamingResponse:
        settings = GenerationSettings(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            seed=request.seed,
        )

        def events() -> Iterator[str]:
            try:
                for event in resolve_manager().generate_stream(
                    request.message, settings
                ):
                    yield f"data: {json.dumps(asdict(event))}\n\n"
            except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
