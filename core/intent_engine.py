"""Production-ready Gemini-based intent engine for Alara."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from .action_registry import ACTION_REGISTRY, ActionRegistry
from .normalizer import ActionNormalizer
from .prompt_builder import PromptBuilder

load_dotenv()


class Action(BaseModel):
    """Structured action response from the intent engine."""

    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str
    normalized: bool = False
    fallback: bool = False

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, val))


class JSONExtractionError(Exception):
    """Raised when the LLM response cannot be coerced into JSON."""


class IntentEngine:
    """Classifies transcripts into structured actions using Gemini + registry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        registry: Optional[ActionRegistry] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        normalizer: Optional[ActionNormalizer] = None,
    ):
        self.registry: ActionRegistry = registry or ACTION_REGISTRY
        self.prompt_builder: PromptBuilder = prompt_builder or PromptBuilder(
            self.registry
        )
        self.normalizer: ActionNormalizer = normalizer or ActionNormalizer(
            self.registry
        )

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or passed in")

        model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        logger.info(
            "IntentEngine initialized | model={} | actions={}",
            self.model_name,
            len(self.registry.actions_by_name),
        )

    # Public API -------------------------------------------------------------

    def parse(self, transcript: str, max_retries: int = 4) -> Action:
        """Main entry point: transcript → Action."""
        transcript = (transcript or "").strip()
        if not transcript:
            logger.warning("Empty transcript passed to IntentEngine.parse")
            return self._fallback_unknown(transcript)

        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            prompt = self.prompt_builder.build(transcript)
            try:
                raw_text = self._call_llm(prompt, transcript)
                logger.bind(stage="llm_raw", attempt=attempt + 1).debug(raw_text)

                payload = self._extract_action_payload(raw_text)
                logger.bind(stage="parsed_json", attempt=attempt + 1).debug(payload)

                action = self._build_action_from_payload(payload, transcript)

                logger.bind(
                    stage="normalized_action",
                    action=action.action,
                    confidence=action.confidence,
                    normalized=action.normalized,
                    fallback=action.fallback,
                ).info("IntentEngine parse succeeded")

                return action

            except JSONExtractionError as exc:
                last_exception = exc
                logger.warning(
                    "JSON parse failure on attempt {}: {}",
                    attempt + 1,
                    exc,
                )
                # Fixed short wait on JSON parse issues.
                time.sleep(0.5)
                continue
            except Exception as exc:
                last_exception = exc
                msg = str(exc)
                is_5xx = any(code in msg for code in ("500", "502", "503"))
                if is_5xx and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "5xx error from Gemini on attempt {} (will retry in {}s): {}",
                        attempt + 1,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                    continue

                if attempt < max_retries - 1:
                    logger.warning(
                        "Non-retriable error from Gemini on attempt {}: {}",
                        attempt + 1,
                        exc,
                    )
                    break

        logger.warning(
            "All Gemini attempts exhausted; using deterministic fallback. last_error={}",
            last_exception,
        )
        return self._fallback_unknown(transcript)

    # Internal helpers -------------------------------------------------------

    def _call_llm(self, system_prompt: str, transcript: str) -> str:
        """Invoke Gemini with the constructed prompt."""
        response = self.model.generate_content(
            system_prompt,
            generation_config={"temperature": 0.0},
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Empty response from Gemini")
        return text

    def _extract_action_payload(self, text: str) -> Dict[str, Any]:
        """Robustly extract JSON (object or list) from LLM output."""
        body = text.strip()

        # Handle markdown fences.
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", body, flags=re.DOTALL)
        if fence_match:
            body = fence_match.group(1).strip()

        # Trim leading commentary before first JSON-like structure.
        first_brace = body.find("{")
        first_bracket = body.find("[")
        candidates = [i for i in (first_brace, first_bracket) if i != -1]
        if candidates:
            start = min(candidates)
            body = body[start:]

        sanitized = self._sanitize_json_like(body)

        try:
            obj = json.loads(sanitized)
        except json.JSONDecodeError:
            # Last resort: try Python-style dict/list via ast.literal_eval.
            import ast

            try:
                obj = ast.literal_eval(sanitized)
            except Exception as exc:
                raise JSONExtractionError(f"Unable to parse JSON from LLM: {exc}") from exc

        # If a list is returned, prefer the first element.
        if isinstance(obj, list):
            if not obj:
                raise JSONExtractionError("LLM returned an empty list")
            obj = obj[0]

        # Some models wrap in {"actions": [...]}.
        if isinstance(obj, dict) and "action" not in obj and "actions" in obj:
            actions = obj.get("actions") or []
            if actions:
                obj = actions[0]

        if not isinstance(obj, dict):
            raise JSONExtractionError(f"LLM payload is not an object: {type(obj)}")

        return obj

    def _sanitize_json_like(self, text: str) -> str:
        """Best-effort cleanup of quasi-JSON into valid JSON."""
        cleaned = text.strip()

        # Normalize curly quotes.
        cleaned = (
            cleaned.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )

        # Remove trailing commas before } or ].
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        # If keys are unquoted but values look quoted, try a simple key quoting.
        # This is intentionally conservative.
        cleaned = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*:",
            r'"\1":',
            cleaned,
        )

        return cleaned

    def _build_action_from_payload(
        self, payload: Dict[str, Any], transcript: str
    ) -> Action:
        raw_action_name = str(payload.get("action", "unknown"))
        raw_params: Dict[str, Any] = payload.get("params") or {}
        raw_confidence = payload.get("confidence", 0.0)

        normalized_action, normalized_params = self.normalizer.normalize(
            raw_action_name, raw_params, transcript
        )
        normalized_flag = (
            normalized_action != raw_action_name or normalized_params != raw_params
        )

        action = Action(
            action=normalized_action,
            params=normalized_params,
            confidence=raw_confidence,
            raw_text=transcript,
            normalized=normalized_flag,
            fallback=False,
        )
        return action

    def _fallback_unknown(self, transcript: str) -> Action:
        """Deterministic fallback path when LLM is unavailable or fails."""
        action_name, params = self.normalizer.normalize("unknown", {}, transcript)
        logger.bind(
            stage="fallback",
            action=action_name,
        ).warning("Using deterministic fallback for transcript")

        return Action(
            action=action_name,
            params=params,
            confidence=0.0,
            raw_text=transcript,
            normalized=True,
            fallback=True,
        )

    # Backwards-compatible alias used by pipeline/UI code.
    def classify(self, transcription: str, max_retries: int = 4) -> Action:
        return self.parse(transcription, max_retries=max_retries)

