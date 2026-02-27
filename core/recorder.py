"""
alara/core/recorder.py

Captures microphone audio after wake-word detection and stops on silence.
Returns WAV bytes for the transcriber.
"""

import io
import os
import threading
import wave

import numpy as np
import sounddevice as sd
from loguru import logger


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500
MIN_RECORDING_MS = 300


class AudioRecorder:
    """
    Records one voice command after wake trigger.
    """

    def __init__(self):
        self.silence_timeout_ms = int(os.getenv("SILENCE_TIMEOUT_MS", 1500))
        self.max_duration_s = 15
        self._stop_event = threading.Event()

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        return rms < SILENCE_THRESHOLD

    def record(self) -> bytes:
        """Record until silence is detected, then return WAV bytes."""
        logger.info("Recording command...")
        self._stop_event.clear()
        frames = []
        silent_chunks = 0
        total_chunks = 0

        silence_chunks_needed = int(
            (self.silence_timeout_ms / 1000) * SAMPLE_RATE / CHUNK_SIZE
        )
        max_chunks = int(self.max_duration_s * SAMPLE_RATE / CHUNK_SIZE)
        min_chunks = int((MIN_RECORDING_MS / 1000) * SAMPLE_RATE / CHUNK_SIZE)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        ) as stream:
            while total_chunks < max_chunks:
                if self._stop_event.is_set():
                    logger.info("Recording stopped early")
                    break
                chunk, _ = stream.read(CHUNK_SIZE)
                audio_np = np.frombuffer(chunk, dtype=np.int16)
                frames.append(audio_np.copy())
                total_chunks += 1

                if total_chunks < min_chunks:
                    continue

                if self._is_silent(audio_np):
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed:
                        logger.debug(
                            f"Silence detected after {total_chunks} chunks "
                            f"({total_chunks * CHUNK_SIZE / SAMPLE_RATE:.1f}s)"
                        )
                        break
                else:
                    silent_chunks = 0

        if not frames:
            logger.warning("No audio captured")
            return b""

        return self._to_wav_bytes(np.concatenate(frames, axis=0))

    def _to_wav_bytes(self, audio_np: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_np.tobytes())
        buf.seek(0)
        return buf.read()

    def stop(self):
        """Request current recording loop to stop."""
        self._stop_event.set()
