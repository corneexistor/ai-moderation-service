import logging
from dataclasses import dataclass
from typing import Tuple

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
    temperature: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    best_of: int = 10  # number of independent sample trajectories, if t > 0
    beam_size: int = 5  # number of beams in beam search
    patience: float = 2.0  # patience in beam search

    # hallucination options
    compression_ratio_threshold: float = 2.6
    log_prob_threshold: float = -1.5
    no_speech_threshold: float = 0.6
    hallucination_silence_threshold: float = 2.0
    condition_on_previous_text: bool = False

    word_timestamps: bool = True
    append_punctuations: str =  "\"'.。,，!！?？:：)]}、"


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
