"""
alara/core/wake_word.py

Continuously listens for wake activation.
Primary path uses OpenWakeWord; fallback path uses volume-based triggering.
"""

import os
import threading

import numpy as np
import sounddevice as sd
from loguru import logger
from openwakeword.model import Model


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
CHANNELS = 1


class WakeWordDetector:
    """
    Background wake detector that calls `on_detected` on trigger.
    """

    def __init__(self, on_detected: callable, threshold: float = None):
        self.on_detected = on_detected
        self.threshold = threshold or float(os.getenv("WAKE_WORD_THRESHOLD", 0.5))

        # Derive both underscore and display variants from the same env var.
        raw_wake = os.getenv("WAKE_WORD", "hey_jarvis").strip()
        self.wake_word_model_name = raw_wake.replace(" ", "_")
        self.wake_word_display = raw_wake.replace("_", " ")
        # Backwards-compatible attribute if anything else inspects it.
        self.wake_word = self.wake_word_display
        self._running = False
        self._thread = None
        self._model = None
        self._use_volume_fallback = False

    def _load_model(self):
        logger.info(f"Loading wake word model: {self.wake_word_display}")

        # OpenWakeWord only ships a fixed set of pre-trained models. If the
        # configured wake word doesn't correspond to a known model, fall back
        # to the simpler volume-based detector but still keep the display name.
        known_models = {
            "alexa",
            "hey_mycroft",
            "hey_jarvis",
            "hey_rhasspy",
            "current_weather",
            "timers",
        }
        if self.wake_word_model_name.lower() not in known_models:
            logger.warning(
                "No built-in OpenWakeWord model for '%s'; using volume-based wake detection instead.",
                self.wake_word_display,
            )
            self._use_volume_fallback = True
            return

        try:
            self._model = Model(
                wakeword_models=[self.wake_word_model_name],
            )
            self._use_volume_fallback = False
            logger.success(f"Wake word model loaded (OpenWakeWord): {self.wake_word_display}")
        except Exception as e:
            logger.warning(
                "OpenWakeWord unavailable (%s: %s) for model '%s', switching to volume-based fallback",
                type(e).__name__,
                str(e),
                self.wake_word_display,
            )
            self._use_volume_fallback = True

    def _listen_loop(self):
        if self._use_volume_fallback:
            self._listen_loop_volume_based()
        else:
            self._listen_loop_ml_based()

    def _listen_loop_ml_based(self):
        logger.info(f"Wake detector active (threshold={self.threshold:.2f})")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        ) as stream:
            while self._running:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                predictions = self._model.predict(audio_np)

                for model_name, confidence in predictions.items():
                    if confidence >= self.threshold:
                        logger.info(f"Wake word detected [{model_name}] confidence={confidence:.2f}")
                        self._model.reset()
                        threading.Thread(target=self.on_detected, daemon=True).start()
                        break

    def _listen_loop_volume_based(self):
        """
        Fallback detector when OpenWakeWord is unavailable.
        Trigger on sustained speech-level RMS.
        """
        logger.info("Wake detector active (volume-based fallback)")

        volume_threshold = 1000
        min_duration_chunks = 2
        silence_limit = 10

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        ) as stream:
            silent_chunks = 0
            speech_chunks = 0

            while self._running:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(float)
                rms = np.sqrt(np.mean(audio_np ** 2))

                if rms > volume_threshold:
                    speech_chunks += 1
                    silent_chunks = 0
                    if speech_chunks >= min_duration_chunks:
                        logger.info(f"Voice trigger detected (RMS={rms:.0f})")
                        speech_chunks = 0
                        threading.Thread(target=self.on_detected, daemon=True).start()
                else:
                    silent_chunks += 1
                    if silent_chunks > silence_limit:
                        speech_chunks = 0

    def start(self):
        """Start listening in a background thread."""
        if self._running:
            logger.warning("WakeWordDetector already running")
            return

        self._load_model()
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.success("WakeWordDetector started")

    def stop(self):
        """Stop the listening thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("WakeWordDetector stopped")
