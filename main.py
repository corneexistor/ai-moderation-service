import asyncio
import gc
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import whisper
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def setup_device() -> tuple[str, str, bool]:
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), gpu_memory)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if gpu_memory >= 8:
            model_size = "large"
        elif gpu_memory >= 6:
            model_size = "medium"
        elif gpu_memory >= 4:
            model_size = "small"
        else:
            model_size = "base"

        return "cuda", model_size, True

    if torch.backends.mps.is_available():
        logger.info("Устройство: Apple MPS")
        return "mps", "medium", False

    logger.info("Устройство: CPU")
    return "cpu", "base", False


DEVICE, MODEL_SIZE, USE_FP16 = setup_device()


class Config:
    MODEL_SIZE = MODEL_SIZE
    DEVICE = DEVICE
    USE_FP16 = USE_FP16
    MAX_FILE_SIZE = 500 * 1024 * 1024
    TEMP_DIR = tempfile.mkdtemp()
    DEFAULT_AUDIO_PREPROCESS = "none"
    NUM_WORKERS = 2 if DEVICE == "cuda" else 1

    LANGUAGE_PROMPTS: dict[str, str] = {
        "ru": (
            "Это русская речь. Внимательно слушайте окончания слов, "
            "правильно определяйте падежи и грамматические конструкции."
        ),
        "en": "This is English speech.",
        "es": "Esta es una conversación en español.",
        "fr": "Il s'agit d'un discours en français.",
        "de": "Dies ist eine deutsche Rede.",
        "zh": "这是中文语音。",
        "ja": "これは日本語の音声です。",
        "ko": "이것은 한국어 음성입니다.",
    }

    LANGUAGE_THRESHOLDS: dict[str, dict] = {
        "ru": {"compression_ratio_threshold": 2.2},
    }


config = Config()

logger.info("Загрузка модели %s на %s...", config.MODEL_SIZE.upper(), config.DEVICE.upper())

model = whisper.load_model(config.MODEL_SIZE, device=config.DEVICE)

if config.DEVICE == "cuda":
    if config.USE_FP16:
        model = model.half()
    model.eval()
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()

    logger.info(
        "Модель загружена. GPU память: %.1f / %.1f GB",
        torch.cuda.memory_allocated(0) / 1e9,
        torch.cuda.get_device_properties(0).total_memory / 1e9)

logger.info("Модель загружена.")


def _model_dtype() -> torch.dtype:
    return next(model.parameters()).dtype


def detect_language(file_path: str) -> Optional[str]:
    try:
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio, device=config.DEVICE).to(dtype=_model_dtype())

        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        logger.info("Определён язык: %s (p=%.2f)", lang, probs[lang])
        return lang
    except Exception as e:
        logger.warning("Ошибка определения языка: %s", e)
        return None


def _confidence(segments: list) -> float:
    scores = [np.exp(s["avg_logprob"]) for s in segments if "avg_logprob" in s]
    return float(np.mean(scores)) if scores else 0.0


def _build_transcribe_options(lang: Optional[str], task: str = "transcribe") -> dict:
    options: dict = {
        "task":                        task,
        "temperature":                 0.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold":          -1.0,
        "no_speech_threshold":         0.6,
        "condition_on_previous_text":  True,
        "word_timestamps":             True,
        "fp16":                        config.USE_FP16,
        "verbose":                     False,
    }

    if lang:
        options["language"] = lang
        options["initial_prompt"] = config.LANGUAGE_PROMPTS.get(lang, "")
        options.update(config.LANGUAGE_THRESHOLDS.get(lang, {}))

    return options


def run_transcribe(file_path: str, language: Optional[str]) -> dict:
    if language and language != "auto":
        lang = language
    else:
        lang = detect_language(file_path)

    options = _build_transcribe_options(lang)
    result = model.transcribe(file_path, **options)

    detected_language = result.get("language") or lang
    segments          = result.get("segments", [])

    return {
        "text":       result["text"].strip(),
        "language":   detected_language,
        "segments":   segments,
        "confidence": _confidence(segments)
    }


def format_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def cleanup_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as e:
                logger.warning("Не удалось удалить %s: %s", path, e)
    gc.collect()
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="Whisper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=config.NUM_WORKERS)


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    return_timestamps: bool = Form(False),
):
    try:
        content = await file.read()
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимум {config.MAX_FILE_SIZE // (1024*1024)} МБ",
            )

        ext = os.path.splitext(file.filename or "audio")[-1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=config.TEMP_DIR) as tmp:
            tmp.write(content)
            path_for_transcribe = tmp.name

        logger.info("Файл: %s (%.1f МБ)", file.filename, len(content) / 1024 / 1024)

        logger.info("Распознавание [%s, %s]...", language or "auto")
        start = time.time()

        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            run_transcribe,
            path_for_transcribe,
            language
        )

        elapsed = time.time() - start
        logger.info("Готово за %.1f сек", elapsed)

        dur = result["segments"][-1]["end"] if result["segments"] else 0.0
        payload: dict = {
            "ok": True,
            "text": result["text"],
            "lang": result["language"],
            "file": file.filename,
            "dur": round(float(dur), 3),
            "conf": round(result["confidence"], 4),
            "sec": round(elapsed, 3),
            "meta": {
                "dev": config.DEVICE.upper(),
                "mdl": config.MODEL_SIZE,
                "fp16": config.USE_FP16,
            },
            "at": datetime.now().isoformat(),
        }

        if return_timestamps:
            payload["seg"] = [
                [format_time(s["start"]), format_time(s["end"]), s["text"]]
                for s in result["segments"]
            ]

        return Response(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            media_type="application/json; charset=utf-8",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка транскрипции: %s", e, exc_info=True)
        return Response(
            content=json.dumps({"ok": False, "err": str(e)}, ensure_ascii=False, separators=(",", ":")),
            status_code=500,
            media_type="application/json; charset=utf-8",
        )
    finally:
        background_tasks.add_task(cleanup_files, path_for_transcribe)


@app.get("/health")
async def health():
    gpu_info = {}
    if config.DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(props.total_memory / 1e9, 2),
            "memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_free_gb": round((props.total_memory - torch.cuda.memory_allocated(0)) / 1e9, 2),
        }
    return {
        "status": "healthy",
        "model": config.MODEL_SIZE,
        "device": config.DEVICE.upper(),
        "fp16": config.USE_FP16,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": gpu_info,
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Временные файлы: %s", config.TEMP_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=300)