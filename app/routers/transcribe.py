import asyncio
import logging
import tempfile
import time
from concurrent.futures.thread import ThreadPoolExecutor

from fastapi import UploadFile, File, Form, APIRouter, HTTPException, Depends, Response
from starlette.responses import JSONResponse

from config import Config
from dependencies import get_whisper_service, get_executor, get_config
from models.transcribe_response import TranscribeResponse
from services.moderation_service import check_text_for_explicit_content
from services.whisper_service import WhisperService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    return_timestamps: bool = Form(False),
    whisper_service: WhisperService = Depends(get_whisper_service),
    executor: ThreadPoolExecutor = Depends(get_executor),
    config: Config = Depends(get_config),
):
    try:
        content = await file.read()

        # File size validation
        if len(content) > config.api_config.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size is too big. Maximum {config.api_config.max_file_size // (1024*1024)} MB",
            )

        # Context manager ensures file is deleted after processing
        with tempfile.NamedTemporaryFile(
            delete=True,
            dir=config.api_config.temp_dir
        ) as tmp:
            tmp.write(content)
            tmp.flush()

            logger.info("File: %s (%.1f MB)", file.filename, len(content) / 1024 / 1024)

            start = time.time()

            # Run whisper in thread pool
            result = await asyncio.get_running_loop().run_in_executor(
                executor,
                whisper_service.run_transcribe,
                tmp.name,
            )

            # Getting full text from segments
            full_text = " ".join(segment.text for segment in result.segments)

            # Run explicit check in thread pool
            is_explicit = await asyncio.get_running_loop().run_in_executor(
                executor,
                check_text_for_explicit_content,
                full_text,
            )

            elapsed = time.time() - start
            logger.info("Done in %.1f s", elapsed)

            return TranscribeResponse(
                text=full_text.strip(),
                segments=result.segments if return_timestamps else None,
                language=result.language,
                is_explicit=is_explicit,
                file=file.filename,
                confidence=round(result.confidence, 4),
                done_in=round(elapsed, 3),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcribe error: %s", e, exc_info=True)
        return JSONResponse(
            content={"err": str(e)},
            status_code=500,
        )