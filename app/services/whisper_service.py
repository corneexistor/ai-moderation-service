import logging
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import whisper

from config import ModelConfig, TranscribeConfig

logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self, config: ModelConfig, transcribe_config: TranscribeConfig):
        self.model_config = config
        self.transcribe_config = transcribe_config
        self.model = self._load_model()

    def _load_model(self):
        logger.info(
            "Loading model %s on %s...",
            self.model_config.model_size.upper(),
            self.model_config.device.upper()
        )

        model = whisper.load_model(
            self.model_config.model_size,
            device=self.model_config.device,
            in_memory=self.model_config.in_memory,
        )

        if self.model_config.device == "cuda":
            if self.model_config.use_fp16:
                model = model.half()
            model.eval()
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()

            logger.info(
                "Model loaded. GPU memory: %.1f / %.1f GB",
                torch.cuda.memory_allocated(0) / 1e9,
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )

        return model

    def detect_language(self, file_path: str) -> Optional[str]:
        try:
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(
                audio,
                n_mels=128 if self.model_config.model_size == "large" else 80,
                device=self.model_config.device,
            ).to(self.model_config.device)

            _, probs = self.model.detect_language(mel=mel)
            lang = max(probs, key=probs.get)
            logger.info("Detected language: %s (p=%.2f)", lang, probs[lang])
            return lang
        except Exception as e:
            logger.warning("Language detection failed: %s", e)
            return None

    def run_transcribe(self, file_path: str) -> dict:
        lang = self.detect_language(file_path)

        # Converting transcribe config object into dictionary to pass into then function
        options = asdict(self.transcribe_config)
        options["language"] = lang

        result = self.model.transcribe(
            audio=file_path,
            **options,
        )

        detected_language = result.get("language") or lang
        segments = result.get("segments", [])

        return {
            "text": result["text"].strip(),
            "language": detected_language,
            "segments": segments,
            "confidence": self._confidence(segments)
        }

    @staticmethod
    def _confidence(segments: list) -> float:
        # Calculating response confidence by calculating arithmetic mean of all scores
        # Exponential function used to convert logarithmic probability back to normal value between 0 and 1
        scores = [np.exp(s["avg_logprob"]) for s in segments if "avg_logprob" in s]
        return float(np.mean(scores)) if scores else 0.0
