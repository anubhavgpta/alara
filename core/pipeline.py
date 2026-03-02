"""
alara/core/pipeline.py

Main ALARA orchestration pipeline:
WakeWord -> Recorder -> Transcriber -> IntentEngine -> Executor
"""

import time
from typing import Callable, Optional

from loguru import logger

from alara.core.executor import Executor
from alara.core.intent_engine import IntentEngine
from alara.core.recorder import AudioRecorder
from alara.core.transcriber import Transcriber
from alara.core.wake_word import WakeWordDetector


class AlaraPipeline:
    """
    Orchestrates the full voice-command flow.

    Flow per command:
    1. Wake word is detected.
    2. Audio is recorded until silence.
    3. Audio is transcribed with faster-whisper.
    4. Transcription is parsed to action JSON via Ollama.
    5. Action is dispatched to the executor.
    """

    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        transcriber: Optional[Transcriber] = None,
        intent_engine: Optional[IntentEngine] = None,
        executor: Optional[Executor] = None,
        on_wake_event: Optional[Callable[[], None]] = None,
    ):
        logger.info("Initializing ALARA pipeline...")
        self.recorder = recorder or AudioRecorder()
        self.transcriber = transcriber or Transcriber()
        self.intent_engine = intent_engine or IntentEngine()
        self.executor = executor or Executor()
        self.on_wake_event = on_wake_event
        self._is_listening = False

        self.wake_detector = WakeWordDetector(on_detected=self._on_wake_word)
        logger.success("ALARA pipeline initialized")

    def _on_wake_word(self):
        """Called by WakeWordDetector when a wake event is triggered."""
        if self.on_wake_event:
            try:
                self.on_wake_event()
            except Exception as e:
                logger.debug(f"Wake event callback failed: {e}")

        if self._is_listening:
            logger.debug("Already processing a command, ignoring wake trigger")
            return

        self._is_listening = True
        try:
            self._process_command()
        except Exception:
            logger.exception("Command processing failed")
        finally:
            self._is_listening = False

    def _process_command(self):
        """Run the full pipeline for one command."""
        t0 = time.perf_counter()

        wav_bytes = self.recorder.record()
        if not wav_bytes:
            logger.warning("No audio captured, aborting")
            return
        t1 = time.perf_counter()

        transcription = self.transcriber.transcribe(wav_bytes)
        if not transcription:
            logger.warning("Empty transcription, aborting")
            return
        t2 = time.perf_counter()

        action = self.intent_engine.parse(transcription)
        t3 = time.perf_counter()

        result = self.executor.execute(action)
        t4 = time.perf_counter()

        logger.info(
            "Pipeline timing: "
            f"record={t1 - t0:.2f}s | "
            f"transcribe={t2 - t1:.2f}s | "
            f"intent={t3 - t2:.2f}s | "
            f"execute={t4 - t3:.2f}s | "
            f"total={t4 - t0:.2f}s"
        )
        logger.info(f"Result: {result}")

    def start(self):
        """Start the pipeline and block until interrupted."""
        logger.success("ALARA is active. Say the wake word.")
        logger.info("Press Ctrl+C to stop.")

        try:
            self.wake_detector.start()
        except Exception as e:
            logger.warning(f"Wake word unavailable ({e}), using direct listening mode")
            self._direct_listening_loop()
            return

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down ALARA...")
            self.wake_detector.stop()

    def _direct_listening_loop(self):
        """
        Fallback mode when wake word initialization fails.
        Continuously records and processes commands.
        """
        try:
            while True:
                logger.info("Listening for command...")
                self._on_wake_word()
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down ALARA...")
            logger.success("ALARA stopped cleanly.")
