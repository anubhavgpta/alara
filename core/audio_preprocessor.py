"""Audio preprocessing utilities for robust STT input quality."""

from __future__ import annotations

import numpy as np
from loguru import logger

import librosa
import noisereduce as nr


class AudioPreprocessor:
    """Cleans audio with denoising, normalization, and silence trimming."""

    def __init__(self, target_peak: float = 0.95, trim_top_db: int = 20):
        self.target_peak = float(target_peak)
        self.trim_top_db = int(trim_top_db)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess raw audio array.

        Steps:
        1. Noise reduction using first 0.5s as noise profile.
        2. Peak normalization to target amplitude.
        3. Leading/trailing silence trim.
        """
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)

        try:
            cleaned = self._to_mono_float32(audio)
            cleaned = self._reduce_noise(cleaned, sample_rate)
            cleaned = self._normalize(cleaned)
            cleaned = self._trim_silence(cleaned)
            return cleaned.astype(np.float32)
        except Exception as exc:
            logger.warning(f"Audio preprocessing failed, returning raw audio: {exc}")
            return self._to_mono_float32(audio).astype(np.float32)

    def _to_mono_float32(self, audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        if arr.dtype != np.float32:
            # Common path for int16 PCM.
            if np.issubdtype(arr.dtype, np.integer):
                max_val = float(np.iinfo(arr.dtype).max)
                arr = arr.astype(np.float32) / max_val
            else:
                arr = arr.astype(np.float32)
        return arr

    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        noise_samples = max(1, int(0.5 * sample_rate))
        noise_clip = audio[:noise_samples] if len(audio) >= noise_samples else audio
        return nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sample_rate)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak <= 1e-9:
            return audio
        return audio * (self.target_peak / peak)

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_top_db)
        return trimmed if len(trimmed) else audio

