import os
import tempfile
import whisper
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Literal
import logging
import json
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import gc

# ============ НАСТРОЙКА GPU ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Определяем устройство и оптимизируем
def setup_gpu():
    """Настройка GPU для максимальной производительности"""
    if torch.cuda.is_available():
        # Получаем информацию о GPU
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"GPU обнаружен: {gpu_name}")
        logger.info(f"Количество GPU: {gpu_count}")
        logger.info(f"Видеопамять: {gpu_memory:.1f} GB")
        
        # Оптимизации для GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Выбираем модель в зависимости от памяти
        if gpu_memory >= 8:
            model_size = "large"
            logger.info("Достаточно памяти для модели large")
        elif gpu_memory >= 6:
            model_size = "medium"
            logger.info("Достаточно памяти для модели medium")
        elif gpu_memory >= 4:
            model_size = "small"
            logger.info("Достаточно памяти для модели small")
        else:
            model_size = "base"
            logger.warning("Мало видеопамяти, используется модель base")
        
        device = "cuda"
        use_fp16 = True  # Используем FP16 для скорости
        
    elif torch.backends.mps.is_available():
        # Для Mac M1/M2
        device = "mps"
        model_size = "medium"
        use_fp16 = False
        logger.info("Используется Apple MPS (Metal Performance Shaders)")
        
    else:
        device = "cpu"
        model_size = "base"
        use_fp16 = False
        logger.info("GPU не найден, используется CPU")
    
    return device, model_size, use_fp16

# Настраиваем GPU
DEVICE, MODEL_SIZE, USE_FP16 = setup_gpu()

# ============ КОНФИГУРАЦИЯ ============
class Config:
    MODEL_SIZE = MODEL_SIZE
    DEVICE = DEVICE
    USE_FP16 = USE_FP16
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 МБ
    TEMP_DIR = tempfile.mkdtemp()
    # Предобработка по умолчанию: исходный файл (Whisper сам ресемплит внутри transcribe)
    DEFAULT_AUDIO_PREPROCESS = "none"
    
    # Параметры для GPU
    BATCH_SIZE = 4 if DEVICE == "cuda" else 1
    NUM_WORKERS = 2 if DEVICE == "cuda" else 1
    
    # Языки с высоким качеством
    HIGH_QUALITY_LANGUAGES = ["ru", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "uk", "zh", "ja", "ko"]
    
    # Оптимальные настройки для разных языков
    LANGUAGE_PROMPTS = {
        "ru": "Это русская речь. Обратите особое внимание на окончания слов, падежи и грамматику.",
        "en": "This is English speech. Pay attention to pronunciation and context.",
        "es": "Esta es una conversación en español. Presta atención a la pronunciación y el contexto.",
        "fr": "Il s'agit d'un discours en français. Faites attention à la prononciation.",
        "de": "Dies ist eine deutsche Rede. Achten Sie auf die Aussprache.",
        "zh": "这是中文语音。注意发音和语调。",
        "ja": "これは日本語の音声です。発音に注意してください。",
        "ko": "이것은 한국어 음성입니다. 발음에 주의하세요.",
    }

config = Config()

API_VERSION = "4.0.0"

SERVICE_FEATURES = (
    "GPU ускорение (CUDA)",
    "Максимальное качество распознавания",
    "Автоопределение языка",
    "Опциональная предобработка (heavy) для шумных записей",
    "Временные метки слов",
    "Поддержка 100+ языков",
    "FP16 оптимизация",
)


def log_service_capabilities() -> None:
    """Сводка о сервисе (раньше отдавалась с GET /)."""
    logger.info(
        "Сервис: Whisper GPU Max Quality API | версия: %s | статус: ready",
        API_VERSION,
    )
    if config.DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        used_gb = torch.cuda.memory_allocated(0) / 1e9
        logger.info(
            "GPU память после загрузки модели: всего %.1f GB, занято %.1f GB, свободно %.1f GB",
            total_gb,
            used_gb,
            total_gb - used_gb,
        )
    logger.info("Возможности:")
    for feat in SERVICE_FEATURES:
        logger.info("  - %s", feat)
    logger.info(
        "Языки: поддерживаются коды Whisper (ISO 639-1: ru, en, …); "
        "в POST /transcribe укажите Form-поле language или оставьте auto/пусто для определения моделью."
    )


# ============ ЗАГРУЗКА МОДЕЛИ С ОПТИМИЗАЦИЕЙ ДЛЯ GPU ============
logger.info("\n" + "="*70)
logger.info("ЗАПУСК WHISPER GPU MAX QUALITY")
logger.info("="*70)
logger.info(f"Модель: {config.MODEL_SIZE.upper()}")
logger.info(f"Устройство: {config.DEVICE.upper()}")
if config.DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"FP16: {'Включено' if config.USE_FP16 else 'Выключено'}")
logger.info("="*70 + "\n")

logger.info("Загрузка модели Whisper...")

# Загружаем модель с оптимизациями
model = whisper.load_model(config.MODEL_SIZE, device=config.DEVICE)

# Оптимизации для GPU
if config.DEVICE == "cuda":
    if config.USE_FP16:
        model = model.half()  # Используем FP16 для скорости
        logger.info("FP16 активирован")
    
    # Оптимизация для инференса
    model.eval()
    torch.set_grad_enabled(False)
    
    # Оптимизация памяти
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    logger.info("Оптимизация GPU завершена")

logger.info("Модель загружена успешно!")
log_service_capabilities()
logger.info("\n" + "="*70 + "\n")

# ============ FASTAPI APP ============
app = FastAPI(
    title="Whisper GPU Max Quality API",
    description="Максимально качественное распознавание речи с использованием GPU",
    version=API_VERSION,
)

# CORS для веб-приложений
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пул потоков для обработки (оптимизирован для GPU)
executor = ThreadPoolExecutor(max_workers=config.NUM_WORKERS)


def _compact_json(data: dict, status_code: int = 200) -> Response:
    """JSON без лишних пробелов — меньше тело ответа при тех же данных."""
    return Response(
        content=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )


# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============
def preprocess_audio(file_path: str, mode: Literal["light", "heavy"]) -> str:
    """Готовит WAV для transcribe. Режим none обрабатывается вызывающим кодом.

    none (см. Config): исходник — оптимально для Hi-Res: Whisper сам приводит к 16 kHz
    без узкой полосы и loudnorm.

    light: только 16 kHz моно PCM (без loudnorm и без полосовых фильтров) — удобно для
    экзотических контейнеров/кодеков, почти не режет полезный сигнал осознанно.

    heavy: loudnorm + highpass/lowpass — для сильно зашумлённых/нестабильной громкости;
    для чистого Hi-Res часто хуже, чем none.
    """
    base, _ = os.path.splitext(file_path)
    output_path = f"{base}_{mode}.wav"
    try:
        if mode == "light":
            # light: только приведение к формату, который Whisper всё равно получит
            # внутри (16 kHz моно); без loudnorm и без полосовых фильтров — меньше
            # искажений по сравнению с heavy.
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", file_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                "-y", output_path,
            ]
        else:
            # heavy: громкость по EBU R128 (loudnorm: интегральная громкость I=-16 LUFS,
            # громкостный диапазон LRA, true peak TP), затем полоса 200 Hz–3 kHz.
            # Речь в основном <4 kHz, но срез на 3 kHz убирает шипение/шум сверху и
            # часть согласных — компромисс «чуть проще сигнал для модели» vs потеря деталей;
            # для Hi-Res/музыки это часто хуже, чем подавать исходник (режим none).
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", file_path,
                "-af", "loudnorm=I=-16:LRA=11:TP=-1.5,highpass=f=200,lowpass=f=3000",
                "-ac", "1",
                "-ar", "16000",
                "-y", output_path,
            ]
        # capture_output=True — stderr попадает в исключение при ошибке; check=True —
        # ненулевой код ffmpeg превращается в CalledProcessError и уходит в except ниже.
        subprocess.run(cmd, capture_output=True, check=True)
        # Успех: дальше распознавание идёт по output_path; временный WAV потом удаляет
        # cleanup_files вместе с исходной загрузкой (см. background_tasks в /transcribe).
        return output_path
    except Exception as e:
        logger.warning(f"Предобработка ({mode}) не удалась: {e}")
        return file_path

def transcribe_with_advanced_settings(file_path: str, language: Optional[str], task: str) -> dict:
    """Транскрипция с максимальными настройками качества и GPU оптимизациями"""
    
    # Определяем язык для оптимальных настроек
    if language and language != "auto":
        lang_code = language
    else:
        # Быстрое определение языка с использованием GPU
        try:
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)
            
            # Перемещаем на GPU если доступно
            if config.DEVICE == "cuda":
                audio_tensor = torch.from_numpy(audio).float()
                audio_tensor = audio_tensor.to(config.DEVICE)
                mel = whisper.log_mel_spectrogram(audio_tensor)
            else:
                mel = whisper.log_mel_spectrogram(audio)
            
            _, probs = model.detect_language(mel)
            lang_code = max(probs, key=probs.get)
            logger.info(f"Определен язык: {lang_code} (вероятность: {probs[lang_code]:.2f})")
        except Exception as e:
            logger.warning(f"Ошибка определения языка: {e}")
            lang_code = None
    
    # Оптимальные настройки для разных языков
    best_settings = {
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "verbose": False,
        "word_timestamps": True,
        "fp16": config.USE_FP16  # Используем FP16 для скорости на GPU
    }
    
    # Добавляем язык
    if lang_code:
        best_settings["language"] = lang_code
        best_settings["initial_prompt"] = config.LANGUAGE_PROMPTS.get(lang_code, "")
    
    # Добавляем задачу
    best_settings["task"] = task
    
    # Для русского языка дополнительные улучшения
    if lang_code == "ru":
        best_settings["initial_prompt"] = (
            "Это русская речь. Внимательно слушайте окончания слов, "
            "правильно определяйте падежи и грамматические конструкции. "
            "Записывайте текст грамотно, с правильными окончаниями."
        )
        best_settings["compression_ratio_threshold"] = 2.2
    
    # Выполняем транскрипцию
    result = model.transcribe(file_path, **best_settings)
    
    # Пост-обработка текста
    text = result["text"].strip()
    
    return {
        "text": text,
        "language": result.get("language", lang_code),
        "segments": result.get("segments", []),
        "confidence": get_confidence_score(result.get("segments", []))
    }

def get_confidence_score(segments: List) -> float:
    """Вычисление уверенности распознавания"""
    if not segments:
        return 0.0
    
    confidences = []
    for segment in segments:
        if "avg_logprob" in segment:
            confidences.append(np.exp(segment["avg_logprob"]))
    
    return np.mean(confidences) if confidences else 0.0

def format_time(seconds: float) -> str:
    """Форматирование времени"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def cleanup_files(*file_paths):
    """Удаление временных файлов и очистка GPU памяти"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass
    
    # Очистка GPU памяти после обработки
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

# ============ API ENDPOINTS ============

@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Аудиофайл (любой формат)"),
    language: Optional[str] = Form(None, description="Код языка (ru, en, es...) или auto"),
    task: str = Form("transcribe", description="transcribe/translate"),
    return_timestamps: bool = Form(False, description="Вернуть временные метки"),
    audio_preprocess: str = Form(config.DEFAULT_AUDIO_PREPROCESS),
):
    """
    Максимально качественное распознавание речи с использованием GPU
    
    - Использует GPU для ускорения (CUDA)
    - Поддерживаются все аудиоформаты
    - Предобработка опциональна (по умолчанию исходный файл — см. audio_preprocess)
    - Оптимизация для каждого языка
    - Высокая точность даже при сложных условиях
    - Временные метки слов

    Успешный ответ — компактные ключи: ok, text, lang, task, prep, file, dur, conf, sec,
    meta {dev, mdl, fp16}, at; при return_timestamps=true — seg: [[start,end,text], ...].
    """
    
    temp_file = None
    processed_file = None
    
    try:
        if audio_preprocess not in ("none", "light", "heavy"):
            raise HTTPException(
                status_code=400,
                detail="audio_preprocess должен быть одним из: none, light, heavy",
            )

        # Сохраняем файл
        ext = file.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", dir=config.TEMP_DIR) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        # Проверка размера
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE / (1024 * 1024):
            raise HTTPException(status_code=400, detail=f"Файл слишком большой. Максимум {config.MAX_FILE_SIZE / (1024 * 1024):.0f} МБ")
        
        logger.info(f"Обработка файла: {file.filename} ({file_size_mb:.1f} МБ)")
        if audio_preprocess == "none":
            path_for_transcribe = temp_file
            logger.info("Предобработка: none (исходный файл для Whisper)")
        else:
            processed_file = preprocess_audio(temp_file, audio_preprocess)
            path_for_transcribe = processed_file
            logger.info(f"Предобработка: {audio_preprocess} -> {processed_file}")
        
        # Распознавание
        logger.info(f"Начало распознавания... Язык: {language or 'auto'}")
        logger.info(f"Устройство: {config.DEVICE.upper()} | FP16: {config.USE_FP16}")
        start_time = time.time()
        
        # asyncio: обработчик объявлен как async, поэтому выполняется в event loop uvicorn.
        # Синхронный вызов Whisper/PyTorch внутри этой корутины занял бы поток цикла
        # событий надолго и заблокировал бы остальные HTTP-запросы на этом worker'е.
        # run_in_executor переносит вызов transcribe_with_advanced_settings в пул потоков
        # executor (ThreadPoolExecutor): тяжёлая работа идёт в отдельном потоке.
        # await привязывается к Future от executor'а и отдаёт управление циклу событий,
        # пока поток не вернёт результат — так loop остаётся отзывчивым для других клиентов.
        # get_event_loop() — текущий цикл, в контексте которого выполняется эта корутина
        # (в Python 3.10+ в async-функциях предпочтителен asyncio.get_running_loop()).
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            transcribe_with_advanced_settings,
            path_for_transcribe,
            language,
            task
        )
        
        elapsed_time = time.time() - start_time
        
        # Логируем использование GPU
        if config.DEVICE == "cuda":
            gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Использовано GPU памяти: {gpu_memory:.2f} GB")
        
        logger.info(f"Распознавание завершено за {elapsed_time:.1f} сек")
        
        dur = result["segments"][-1]["end"] if result["segments"] else 0.0
        payload: dict = {
            "ok": True,
            "text": result["text"],
            "lang": result["language"],
            "task": task,
            "prep": audio_preprocess,
            "file": file.filename,
            "dur": round(float(dur), 3),
            "conf": round(float(result["confidence"]), 4),
            "sec": round(float(elapsed_time), 3),
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
        
        return _compact_json(payload)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return _compact_json({"ok": False, "err": str(e)}, status_code=500)
    
    finally:
        # FastAPI BackgroundTasks: задача выполняется после отправки ответа клиенту,
        # чтобы удаление временных файлов не увеличивало время до первого байта ответа.
        background_tasks.add_task(cleanup_files, temp_file, processed_file)

@app.get("/health")
async def health():
    """Проверка здоровья сервиса с информацией о GPU"""
    gpu_info = {}
    if config.DEVICE == "cuda":
        gpu_info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "memory_used_gb": torch.cuda.memory_allocated(0) / 1e9,
            "memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        }
    
    return {
        "status": "healthy",
        "version": API_VERSION,
        "model": config.MODEL_SIZE,
        "device": config.DEVICE.upper(),
        "fp16": config.USE_FP16,
        "gpu": gpu_info,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }

# ============ ЗАПУСК ============
if __name__ == "__main__":
    import uvicorn
    
    logger.info("\n" + "="*70)
    logger.info("HTTP-сервер: см. логи выше (сводка сервиса после загрузки модели)")
    logger.info(f"Временные файлы: {config.TEMP_DIR}")
    logger.info("="*70)
    logger.info("Документация: http://localhost:8000/docs")
    logger.info("Проверка: http://localhost:8000/health")
    logger.info("Рекомендации для максимальной производительности:")
    logger.info("  - Убедитесь, что драйверы NVIDIA обновлены")
    logger.info("  - Для максимальной скорости используйте CUDA 11.8+")
    logger.info("  - Модель large требует 8+ GB видеопамяти")
    logger.info("  - Используйте FP16 для ускорения в 2 раза")
    logger.info("Нажмите CTRL+C для остановки сервера")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=300
    )