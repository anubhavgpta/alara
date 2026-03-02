"""Production orchestration loop for Alara voice assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from .audio_preprocessor import AudioPreprocessor
from .intent_engine import Action, IntentEngine
from .transcriber import Transcriber
from .voice_profile import VoiceProfile

load_dotenv()


@dataclass
class SessionStats:
    total_commands: int = 0
    successful_commands: int = 0
    reprompts: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_commands == 0:
            return 0.0
        return self.successful_commands / self.total_commands

    @property
    def reprompt_rate(self) -> float:
        if self.total_commands == 0:
            return 0.0
        return self.reprompts / self.total_commands


class AlaraAssistant:
    """Main assistant loop with robust transcription and confidence gates."""

    def __init__(self, user_id: str = "default", vocabulary: Optional[list[str]] = None):
        self.audio_preprocessor = AudioPreprocessor()
        self.transcriber = Transcriber(vocabulary=vocabulary)
        self.intent_engine = IntentEngine()
        self.voice_profile = VoiceProfile(user_id=user_id)
        self.stats = SessionStats()
        self._tts_engine = None

        logger.info(f"Assistant initialized for user_id='{user_id}'")

    def _say(self, text: str) -> None:
        logger.info(f"TTS> {text}")
        try:
            if self._tts_engine is None:
                import pyttsx3

                self._tts_engine = pyttsx3.init()
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception as exc:
            logger.debug(f"TTS unavailable, using logs only: {exc}")

    def _listen_yes_no(self) -> bool:
        transcript, confidence = self.transcriber.record_and_transcribe(duration=3.5)
        text = transcript.lower().strip()
        logger.info(f"Confirmation heard: '{transcript}' (conf={confidence:.2f})")
        yes_tokens = {"yes", "yeah", "yep", "correct", "right", "do it"}
        no_tokens = {"no", "nope", "wrong", "cancel", "stop"}
        if any(tok in text for tok in yes_tokens):
            return True
        if any(tok in text for tok in no_tokens):
            return False
        return False

    def _transcribe_and_parse(self) -> tuple[str, str, float, Action]:
        raw_transcript, confidence = self.transcriber.record_and_transcribe()
        corrected = self.voice_profile.apply(raw_transcript)
        action = self.intent_engine.parse(corrected)
        logger.info(
            f"Transcription raw='{raw_transcript}' corrected='{corrected}' "
            f"stt_conf={confidence:.2f} action={action.action} intent_conf={action.confidence:.2f}"
        )
        return raw_transcript, corrected, confidence, action

    def execute(self, action: Action) -> bool:
        """Execute action stub grouped by domain."""
        logger.info(f"Executing action: {action.action} | params={action.params}")

        # App management: real implementation should map to OS app/window controls.
        if action.action in {
            "open_app",
            "close_app",
            "switch_app",
            "minimize_window",
            "maximize_window",
            "close_window",
        }:
            logger.info(f"[APP MANAGEMENT] {action.action} -> {action.params}")
            return True

        # File system: real implementation should perform open/search/screenshot actions.
        if action.action in {"open_file", "open_folder", "search_files", "take_screenshot"}:
            logger.info(f"[FILE SYSTEM] {action.action} -> {action.params}")
            return True

        # Browser: real implementation should control tabs/navigation/search.
        if action.action in {
            "browser_new_tab",
            "browser_navigate",
            "browser_search",
            "browser_close_tab",
        }:
            logger.info(f"[BROWSER] {action.action} -> {action.params}")
            return True

        # VS Code: real implementation should drive editor commands.
        if action.action in {"vscode_open_file", "vscode_new_terminal", "vscode_search"}:
            logger.info(f"[VS CODE] {action.action} -> {action.params}")
            return True

        # System controls: real implementation should handle OS-level controls.
        if action.action in {"volume_up", "volume_down", "volume_mute", "lock_screen"}:
            logger.info(f"[SYSTEM CONTROLS] {action.action} -> {action.params}")
            return True

        logger.warning(f"Unknown action: {action.action}")
        return False

    def run_once(self) -> None:
        raw_text, corrected_text, stt_confidence, action = self._transcribe_and_parse()
        if not corrected_text.strip():
            logger.warning("No speech detected")
            return

        self.stats.total_commands += 1

        # Confidence gate step 1: silent one-time retry
        if stt_confidence < 0.6:
            self.stats.reprompts += 1
            logger.warning(
                f"Low STT confidence ({stt_confidence:.2f}). Re-recording once silently."
            )
            raw2, corrected2, conf2, action2 = self._transcribe_and_parse()
            if conf2 >= stt_confidence:
                raw_text, corrected_text, stt_confidence, action = raw2, corrected2, conf2, action2

        # Confidence gate step 2: explicit confirmation
        if stt_confidence < 0.45:
            self.stats.reprompts += 1
            self._say(f"Did you mean: {action.action}?")
            if self._listen_yes_no():
                executed = self.execute(action)
                if executed:
                    corrected_for_profile = corrected_text or action.action.replace("_", " ")
                    self.voice_profile.record_correction(raw_text, corrected_for_profile)
                    self.stats.successful_commands += 1
                return
            self._say("Sorry, say your command again")
            return

        executed = self.execute(action)
        if executed:
            self.stats.successful_commands += 1

    def run(self) -> None:
        logger.success("ALARA assistant loop started. Press Ctrl+C to stop.")
        try:
            while True:
                self.run_once()
        except KeyboardInterrupt:
            logger.info("Shutting down ALARA assistant")
            logger.info(
                "Session summary | total_commands={} success_rate={:.1%} reprompt_rate={:.1%}",
                self.stats.total_commands,
                self.stats.success_rate,
                self.stats.reprompt_rate,
            )


if __name__ == "__main__":
    assistant = AlaraAssistant()
    assistant.run()

