"""Robust multi-backend transcription with consensus and arbitration."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import wave
from typing import Iterable, Optional, Sequence, Tuple, Union

import google.generativeai as genai
import librosa
import numpy as np
from deepgram import DeepgramClient, LiveOptions, PrerecordedOptions
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from loguru import logger
from rapidfuzz import fuzz

from .audio_preprocessor import AudioPreprocessor
from .action_registry import ACTION_REGISTRY, ActionRegistry
from .recorder import AudioRecorder

load_dotenv()


class Transcriber:
    """Deepgram + Whisper consensus transcriber."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vocabulary: Optional[Sequence[str]] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY") or os.getenv("DG_API_KEY")
        # Allow overriding via argument or env; default to nova-2.
        self.deepgram_model = model or os.getenv("DEEPGRAM_MODEL", "nova-2")
        self.registry: ActionRegistry = ACTION_REGISTRY
        self.vocabulary = self._build_vocabulary(vocabulary)
        self.preprocessor = AudioPreprocessor()
        self.recorder = AudioRecorder()

        self.whisper_model_name = os.getenv("WHISPER_MODEL", "large-v3")
        self.whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self._whisper_model: WhisperModel | None = None

        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._gemini_model = None

        self.deepgram_client: DeepgramClient | None = None
        if self.api_key:
            self.deepgram_client = DeepgramClient(api_key=self.api_key)
        else:
            logger.warning("Deepgram key missing, transcription will run Whisper-only")

        logger.info(
            f"Transcriber initialized | deepgram_model={self.deepgram_model}, "
            f"whisper_model={self.whisper_model_name}, vocab={len(self.vocabulary)}"
        )

    def _build_vocabulary(self, custom_vocabulary: Optional[Sequence[str]]) -> list[str]:
        if custom_vocabulary:
            return sorted({term.strip().lower() for term in custom_vocabulary if term.strip()})

        generated: set[str] = set()
        for action in self.registry.all_actions():
            generated.add(action.name.replace("_", " ").lower())
            # Include parameter names.
            for key in (action.params_schema.get("properties") or {}).keys():
                generated.add(str(key).replace("_", " ").lower())
            # Include per-action keywords.
            for kw in action.keywords:
                generated.add(str(kw).lower())
        return sorted(generated)

    def _technical_terms(self) -> set[str]:
        terms: set[str] = set()
        for action_name in self.registry.action_names():
            terms.add(action_name.replace("_", " ").lower())
        return terms

    def _deepgram_keywords(self) -> list[str]:
        technical = self._technical_terms()
        boosted = []
        for term in self.vocabulary:
            weight = 2.0 if term in technical else 1.5
            boosted.append(f"{term}:{weight:.1f}")
        return boosted

    def _whisper_prompt(self) -> str:
        vocab_hint = ", ".join(self.vocabulary[:80])
        return (
            "This is desktop assistant command audio. "
            "Prioritize app names and operating-system actions. "
            f"Relevant vocabulary: {vocab_hint}"
        )

    def _ensure_whisper(self) -> WhisperModel:
        if self._whisper_model is not None:
            return self._whisper_model

        # First attempt: use configured device/compute_type from env.
        try:
            self._whisper_model = WhisperModel(
                self.whisper_model_name,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
            logger.info(
                "Loaded Whisper model: %s (device=%s, compute_type=%s)",
                self.whisper_model_name,
                self.whisper_device,
                self.whisper_compute_type,
            )
            return self._whisper_model
        except ValueError as exc:
            msg = str(exc)
            # Common on Windows when float16 is requested but not supported.
            if "float16" not in msg:
                raise
            logger.warning(
                "Whisper model init failed with compute_type=%s on device=%s (%s). "
                "Retrying with safe CPU int8 settings.",
                self.whisper_compute_type,
                self.whisper_device,
                msg,
            )

        # Fallback: always-supported CPU int8 configuration.
        self.whisper_device = "cpu"
        self.whisper_compute_type = "int8"
        self._whisper_model = WhisperModel(
            self.whisper_model_name,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
        )
        logger.info(
            "Loaded Whisper model with fallback settings: %s (device=%s, compute_type=%s)",
            self.whisper_model_name,
            self.whisper_device,
            self.whisper_compute_type,
        )
        return self._whisper_model

    def _ensure_gemini(self):
        if self._gemini_model is not None:
            return self._gemini_model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        self._gemini_model = genai.GenerativeModel(self.gemini_model_name)
        return self._gemini_model

    def _wav_bytes_to_array(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            sampwidth = wf.getsampwidth()

        if sampwidth == 2:
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        return arr, sample_rate

    def _array_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def _extract_deepgram_text(self, response) -> str:
        best = ""
        try:
            channels = response.results.channels if response and response.results else []
            for channel in channels:
                for alt in getattr(channel, "alternatives", []) or []:
                    transcript = (getattr(alt, "transcript", "") or "").strip()
                    if len(transcript) > len(best):
                        best = transcript
        except Exception:
            return ""
        return best.strip()

    def _get_prerecorded_client(self):
        if self.deepgram_client is None:
            raise RuntimeError("Deepgram is not configured")
        if hasattr(self.deepgram_client.listen, "rest"):
            rest = self.deepgram_client.listen.rest
            return rest.v("1") if hasattr(rest, "v") else rest
        prerecorded = self.deepgram_client.listen.prerecorded
        return prerecorded.v("1") if hasattr(prerecorded, "v") else prerecorded

    def _transcribe_deepgram_sync(self, wav_bytes: bytes) -> str:
        if self.deepgram_client is None:
            return ""
        options = PrerecordedOptions(
            model=self.deepgram_model,
            filler_words=True,
            smart_format=True,
            punctuate=False,
            keywords=self._deepgram_keywords(),
            language="en",
        )
        try:
            client = self._get_prerecorded_client()
            response = client.transcribe_file({"buffer": wav_bytes}, options)
            text = self._extract_deepgram_text(response)
            logger.debug(f"Deepgram transcript: {text}")
            return text
        except Exception as exc:
            msg = str(exc)
            # If the project/key does not have access to the requested model (403/401),
            # disable Deepgram for the rest of this session and fall back to Whisper-only.
            if "Status: 403" in msg or "Status: 401" in msg or "does not have access" in msg:
                logger.warning(
                    "Deepgram authorization/model error ({}). "
                    "Disabling Deepgram for this session and falling back to Whisper-only.",
                    msg,
                )
                self.deepgram_client = None
            else:
                logger.warning(f"Deepgram transcription failed: {exc}")
            return ""

    def _transcribe_whisper_sync(self, wav_bytes: bytes) -> str:
        model = self._ensure_whisper()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp.flush()
                tmp_path = tmp.name
            segments, _ = model.transcribe(
                tmp_path,
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4],
                initial_prompt=self._whisper_prompt(),
                vad_filter=True,
                condition_on_previous_text=False,
                language="en",
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
            logger.debug(f"Whisper transcript: {text}")
            return text
        except Exception as exc:
            logger.warning(f"Whisper transcription failed: {exc}")
            return ""
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def _transcribe_deepgram(self, wav_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_deepgram_sync, wav_bytes)

    async def _transcribe_whisper(self, wav_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_whisper_sync, wav_bytes)

    def _llm_arbitrate(self, a: str, b: str) -> str:
        model = self._ensure_gemini()
        if model is None:
            return a
        prompt = (
            "Two speech transcription systems disagreed on a desktop voice command.\n"
            "Pick the transcript that is more likely to be a valid command.\n"
            "Return only the chosen text, no extra words.\n\n"
            f"A: {a}\n"
            f"B: {b}\n"
            "Chosen:"
        )
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
            )
            choice = (resp.text or "").strip()
            if not choice:
                return a
            return choice
        except Exception as exc:
            logger.warning(f"Gemini arbitration failed, defaulting to Deepgram: {exc}")
            return a

    async def _consensus_transcribe(self, wav_bytes: bytes) -> tuple[str, float]:
        deepgram_task = asyncio.create_task(self._transcribe_deepgram(wav_bytes))
        whisper_task = asyncio.create_task(self._transcribe_whisper(wav_bytes))
        deepgram_text, whisper_text = await asyncio.gather(deepgram_task, whisper_task)

        if not deepgram_text and not whisper_text:
            return "", 0.0
        if deepgram_text and not whisper_text:
            return deepgram_text, 0.50
        if whisper_text and not deepgram_text:
            return whisper_text, 0.50

        score = fuzz.token_sort_ratio(deepgram_text, whisper_text)
        logger.info(f"Consensus score={score:.1f} | dg='{deepgram_text}' | wh='{whisper_text}'")

        if score >= 85:
            return deepgram_text, 0.95
        if 60 <= score <= 84:
            chosen = await asyncio.to_thread(self._llm_arbitrate, deepgram_text, whisper_text)
            return chosen, 0.75
        return deepgram_text, 0.40

    def _preprocess_wav_bytes(self, audio_bytes: bytes) -> bytes:
        audio, sample_rate = self._wav_bytes_to_array(audio_bytes)
        cleaned = self.preprocessor.process(audio, sample_rate)
        return self._array_to_wav_bytes(cleaned, sample_rate)

    def transcribe_bytes(self, audio_bytes: bytes) -> tuple[str, float]:
        """Transcribe in-memory WAV bytes after preprocessing."""
        if not audio_bytes:
            return "", 0.0
        processed = self._preprocess_wav_bytes(audio_bytes)
        transcript, confidence = asyncio.run(self._consensus_transcribe(processed))
        return transcript.strip(), confidence

    def transcribe_file(self, path: str) -> tuple[str, float]:
        """Transcribe audio from a prerecorded file path."""
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
        cleaned = self.preprocessor.process(audio, sample_rate)
        wav_bytes = self._array_to_wav_bytes(cleaned, sample_rate)
        transcript, confidence = asyncio.run(self._consensus_transcribe(wav_bytes))
        return transcript.strip(), confidence

    def record_and_transcribe(self, duration: Optional[float] = None) -> tuple[str, float]:
        """Record from microphone until silence, then transcribe with consensus."""
        prev_max = self.recorder.max_duration_s
        if duration is not None:
            self.recorder.max_duration_s = float(duration)
        try:
            wav_bytes = self.recorder.record()
            if not wav_bytes:
                return "", 0.0
            return self.transcribe_bytes(wav_bytes)
        finally:
            self.recorder.max_duration_s = prev_max

    # Backward-compatible helpers used by existing pipeline code.
    def transcribe(self, audio_input: Union[str, bytes]) -> str:
        if isinstance(audio_input, bytes):
            text, _ = self.transcribe_bytes(audio_input)
            return text
        text, _ = self.transcribe_file(audio_input)
        return text

    def transcribe_stream(self, audio_stream: Iterable[bytes]) -> str:
        """Streaming helper for compatibility: buffers stream then transcribes."""
        payload = b"".join(audio_stream)
        text, _ = self.transcribe_bytes(payload)
        return text

    def deepgram_live_options(self) -> LiveOptions:
        """Expose live options config for integrations."""
        return LiveOptions(
            model=self.deepgram_model,
            filler_words=True,
            smart_format=True,
            punctuate=False,
            keywords=self._deepgram_keywords(),
            language="en",
            endpointing=500,
            utterance_end_ms=1500,
            interim_results=True,
        )

