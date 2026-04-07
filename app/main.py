import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dependencies import get_config
from routers import transcribe, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title="Whisper ASR microservice")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(transcribe.router)
    app.include_router(health.router)

    return app

_app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info("Temp files: %s", get_config().api_config.temp_dir)
    uvicorn.run("main:_app", host="0.0.0.0", port=8000, reload=True, log_level="info", timeout_keep_alive=300)