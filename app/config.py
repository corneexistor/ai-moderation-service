import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ModelConfig:
    device: str
    model_size: str
    use_fp16: bool
    in_memory: bool = True

@dataclass(frozen=True)
class TranscribeConfig:
    # implementation details
    task: str = "transcribe"
    # sampling-related options
    temperature: Tuple[float, ...] = (0.0,)
    best_of: int = 3
    beam_size: int = 3
    patience: float = 2.0
    # hallucination options
    compression_ratio_threshold: float = 2.0
    log_prob_threshold: float = -0.8
    no_speech_threshold: float = 0.15
    hallucination_silence_threshold: float = 2.0
    condition_on_previous_text: bool = True
    word_timestamps: bool = True
    append_punctuations: str = "\"'.。,，!！?？:：)]}、"
    # VAD settings
    vad_filter: bool = True
    vad_parameters: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass(frozen=True)
class ApiConfig:
    temp_dir: str
    num_workers: int
    max_file_size: int = 500 * 1024 * 1024

@dataclass(frozen=True)
class Config:
    model_config: ModelConfig
    transcribe_config: TranscribeConfig
    api_config: ApiConfig
