import tempfile
from concurrent.futures import ThreadPoolExecutor

from config import Config, ApiConfig, TranscribeConfig, ModelConfig
from services.whisper_service import WhisperService
from utils.device import setup_device

model_config = ModelConfig(*setup_device())
transcribe_config = TranscribeConfig()

num_workers = 2 if model_config.device == "gpu" else 1
api_config = ApiConfig(temp_dir=tempfile.mkdtemp(), num_workers=num_workers)

config = Config(model_config, transcribe_config, api_config)

def get_config() -> Config:
    return config

whisper_service = WhisperService(config.model_config, config.transcribe_config)
executor = ThreadPoolExecutor(max_workers=config.api_config.num_workers)

def get_whisper_service() -> WhisperService:
    return whisper_service

def get_executor() -> ThreadPoolExecutor:
    return executor