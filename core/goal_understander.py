"""Goal understanding module for converting raw goal text into structured context."""

from __future__ import annotations

import json
import os

import google.generativeai as genai
from loguru import logger

from alara.schemas.goal import GoalContext


class GoalUnderstander:
    """Extract structured goal context from a raw user-provided goal string."""

    def __init__(self) -> None:
        self.model_name = "gemini-2.5-flash"
        self.system_prompt = self._build_system_prompt()
        self._disabled = False
        self._model = None

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set. Goal understanding disabled; falling back to GoalContext.from_raw."
            )
            self._disabled = True
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(
                "GoalUnderstander initialized successfully with model={}",
                self.model_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize GoalUnderstander model: {}. Falling back to from_raw.",
                exc,
            )
            self._disabled = True
            self._model = None

    def understand(self, raw_input: str) -> GoalContext:
        if self._disabled or self._model is None:
            logger.warning(
                "Goal understanding is disabled; returning fallback GoalContext.from_raw."
            )
            return GoalContext.from_raw(raw_input)

        raw_response: str | None = None
        try:
            response = self._generate_content(raw_input)
            raw_response = (getattr(response, "text", None) or "").strip()
            if not raw_response:
                raise ValueError("Gemini returned an empty response")

            payload_text = self._strip_fences(raw_response)
            payload = json.loads(payload_text)

            context_payload = {
                "raw_input": raw_input,
                "goal": payload["goal"],
                "scope": payload["scope"],
                "constraints": payload.get("constraints", []),
                "working_directory": payload.get("working_directory"),
                "estimated_complexity": payload["estimated_complexity"],
            }
            logger.debug("GoalUnderstander parsed payload: {}", context_payload)
            return GoalContext.model_validate(context_payload)
        except Exception as exc:
            logger.warning(
                "Goal understanding failed: {} | raw_response={}",
                exc,
                raw_response,
            )
            return GoalContext.from_raw(raw_input)

    def extract(self, raw_goal: str) -> GoalContext:
        return self.understand(raw_goal)

    def _generate_content(self, raw_input: str):
        try:
            return self._model.generate_content(
                raw_input,
                generation_config={"temperature": 0.1},
                system_instruction=self.system_prompt,
            )
        except TypeError:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=self.system_prompt,
            )
            return model.generate_content(
                raw_input,
                generation_config={"temperature": 0.1},
            )

    def _strip_fences(self, text: str) -> str:
        body = text.strip()
        if body.startswith("```"):
            lines = body.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            body = "\n".join(lines).strip()
        return body

    def _build_system_prompt(self) -> str:
        return (
            "You extract structured goal context for ALARA.\n"
            "Return only raw JSON with this schema:\n"
            "{\n"
            '  "goal": "...",\n'
            '  "scope": "filesystem|cli|app|system|mixed",\n'
            '  "constraints": [],\n'
            '  "working_directory": null,\n'
            '  "estimated_complexity": "simple|moderate|complex"\n'
            "}\n\n"
            "Field rules:\n"
            "- goal: clean normalized restatement of the user's request.\n"
            "- scope:\n"
            "  filesystem = only file/folder operations\n"
            "  cli = only terminal/shell commands\n"
            "  app = controlling a specific application\n"
            "  system = OS-level queries/settings\n"
            "  mixed = spans more than one domain\n"
            "- constraints: explicit constraints in text (empty if none).\n"
            "- working_directory: path string if specified/implied, else null.\n"
            "- estimated_complexity:\n"
            "  simple = 1-2 steps, single operation type\n"
            "  moderate = 3-6 steps, possibly multi-domain\n"
            "  complex = 7+ steps, multi-domain, likely needs reflection\n\n"
            "Do not include markdown, code fences, or explanation text."
        )
