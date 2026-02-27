"""
alara/core/transcriber.py

Speech-to-text backend for ALARA.
Supports local faster-whisper and Deepgram.
"""

import io
import os
import wave

import numpy as np
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from faster_whisper import WhisperModel
from loguru import logger


class Transcriber:
    """
    Wraps faster-whisper for local speech-to-text.
    """

    def __init__(self):
        self.backend = os.getenv("TRANSCRIBER_BACKEND", "whisper").strip().lower()
        self.model = None
        self.beam_size = 1
        if self.backend == "deepgram":
            api_key = os.getenv("DEEPGRAM_API_KEY", "")
            self.dg_client = DeepgramClient(api_key) if api_key else None
            if self.dg_client is None:
                logger.warning("DEEPGRAM_API_KEY missing. Deepgram backend unavailable; using whisper fallback.")
            else:
                logger.success("Deepgram transcriber backend initialized")
        else:
            self.dg_client = None
            self._ensure_whisper_model()

    def _ensure_whisper_model(self):
        """Lazy-load whisper so it can be used as a fallback backend."""
        if self.model is not None:
            return

        model_size = os.getenv("WHISPER_MODEL", "tiny.en")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        logger.info(
            f"Loading Whisper model '{model_size}' "
            f"on {device.upper()} ({compute_type})..."
        )
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self.beam_size = 1 if device == "cpu" else 5
        logger.success(f"Whisper ready model={model_size} device={device}")

    def _transcribe_with_whisper(self, wav_bytes: bytes) -> str:
        self._ensure_whisper_model()
        audio_np = self._wav_to_float32(wav_bytes)
        segments, _ = self.model.transcribe(
            audio_np,
            language="en",
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        if transcript:
            logger.info(f"Transcription: '{transcript}'")
        else:
            logger.warning("Whisper returned empty transcript")
        return transcript

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV bytes to text."""
        if not wav_bytes:
            return ""

        try:
            if self.backend == "deepgram":
                try:
                    if self.dg_client is None:
                        raise RuntimeError("Deepgram client is not initialized")

                    payload: FileSource = {"buffer": wav_bytes}
                    options = PrerecordedOptions(
                        model="nova-2",
                        language="en",
                        punctuate=True,
                    )
                    response = self.dg_client.listen.rest.v("1").transcribe_file(payload, options)
                    if hasattr(response, "to_dict"):
                        response_data = response.to_dict()
                    else:
                        response_data = response

                    transcript = (
                        response_data.get("results", {})
                        .get("channels", [{}])[0]
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                        .strip()
                    )
                    if transcript:
                        logger.info(f"Transcription: '{transcript}'")
                        return transcript

                    logger.warning("Deepgram returned empty transcript; falling back to whisper")
                    return self._transcribe_with_whisper(wav_bytes)
                except Exception as deepgram_error:
                    logger.error(f"Deepgram transcription failed: {deepgram_error}. Falling back to whisper.")
                    return self._transcribe_with_whisper(wav_bytes)

            return self._transcribe_with_whisper(wav_bytes)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def _wav_to_float32(self, wav_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes to float32 numpy array in [-1, 1]."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio_int16 = np.frombuffer(raw, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0
