"""Planning module for converting goal context into a task graph via Gemini."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.generativeai as genai
from loguru import logger

from alara.schemas.goal import GoalContext
from alara.schemas.task_graph import Step, TaskGraph


class PlanningError(Exception):
    """Raised when planning fails due to model or schema issues."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class Planner:
    """Generate a task graph from a parsed GoalContext."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com and add it to .env"
            )

        self.model_name = "gemini-2.5-flash"
        self.system_prompt = self._build_system_prompt()
        self.last_raw_response: str | None = None

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logger.info("Planner initialized successfully with model={}", self.model_name)

    def plan(self, goal_context: GoalContext) -> TaskGraph:
        logger.info(
            "Planning started | goal='{}' | complexity={}",
            goal_context.goal,
            goal_context.estimated_complexity,
        )

        user_message = self._build_user_message(goal_context)
        raw_response = self._call_gemini(user_message)
        parsed_steps = self._parse_response(raw_response)

        parsed_ids: set[int] = set()
        for item in parsed_steps:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                parsed_ids.add(item["id"])

        normalized_step_dicts: list[dict[str, Any]] = []
        for raw_step in parsed_steps:
            step_dict = dict(raw_step)
            depends_on = step_dict.get("depends_on", [])
            if isinstance(depends_on, list):
                filtered = [dep for dep in depends_on if dep in parsed_ids]
                removed = [dep for dep in depends_on if dep not in parsed_ids]
                if removed:
                    logger.warning(
                        "Removed invalid depends_on references {} from step id={}",
                        removed,
                        step_dict.get("id"),
                    )
                step_dict["depends_on"] = filtered
            normalized_step_dicts.append(self._normalize_paths(step_dict))

        errors: list[str] = []
        validated_steps: list[Step] = []
        for index, step_dict in enumerate(normalized_step_dicts, start=1):
            try:
                validated_steps.append(Step.model_validate(step_dict))
            except Exception as exc:
                errors.append(f"step[{index}] validation failed: {exc}")

        if errors:
            message = "Invalid steps returned by planner:\n" + "\n".join(errors)
            logger.error(message)
            raise PlanningError(message)

        if len(validated_steps) > 10 and all(not step.depends_on for step in validated_steps):
            logger.warning(
                "Planner produced {} steps with no dependencies - this may indicate a planning error",
                len(validated_steps),
            )

        logger.debug("Parsed planning JSON: {}", normalized_step_dicts)

        task_graph = TaskGraph(
            goal=goal_context.goal,
            goal_context=goal_context,
            steps=validated_steps,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
            results=[],
        )

        logger.info("Planning succeeded | steps={}", len(task_graph.steps))
        logger.debug(
            "TaskGraph details | created_at={} | step_ids={}",
            task_graph.created_at,
            [s.id for s in task_graph.steps],
        )
        return task_graph

    def _call_gemini(self, user_message: str) -> str:
        retry_suffix = (
            "\n\nYour previous response was not valid JSON. Return ONLY the JSON object. "
            "No markdown, no explanation, no code fences."
        )

        first_response_text = self._generate_content(user_message)
        self.last_raw_response = first_response_text
        try:
            self._parse_response(first_response_text)
            return first_response_text
        except PlanningError:
            logger.warning("First parse attempt failed, retrying with stricter instruction.")
            logger.debug("Raw response attempt 1: {}", first_response_text)

        second_message = f"{user_message}{retry_suffix}"
        second_response_text = self._generate_content(second_message)
        self.last_raw_response = second_response_text
        try:
            self._parse_response(second_response_text)
            return second_response_text
        except PlanningError as second_parse_error:
            logger.debug("Raw response attempt 2: {}", second_response_text)
            logger.error("Gemini returned malformed JSON after 2 attempts")
            raise PlanningError(
                "Gemini returned malformed JSON after 2 attempts",
                cause=second_parse_error,
            ) from second_parse_error

    def _generate_content(self, message: str) -> str:
        try:
            response = self.model.generate_content(
                message,
                generation_config={"temperature": 0.2},
                system_instruction=self.system_prompt,
            )
        except TypeError:
            try:
                model = genai.GenerativeModel(
                    self.model_name, system_instruction=self.system_prompt
                )
                response = model.generate_content(
                    message,
                    generation_config={"temperature": 0.2},
                )
            except Exception as exc:
                logger.error("Gemini API call failed: {}", exc)
                raise PlanningError(f"Gemini API call failed: {exc}", cause=exc) from exc
        except Exception as exc:
            logger.error("Gemini API call failed: {}", exc)
            raise PlanningError(f"Gemini API call failed: {exc}", cause=exc) from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            exc = ValueError("Gemini returned an empty response")
            logger.error("Gemini returned an empty response")
            raise PlanningError("Gemini returned an empty response", cause=exc)
        return text

    def _parse_response(self, raw: str) -> list[dict]:
        if raw is None:
            raise PlanningError("Invalid JSON from Gemini: response was None")

        body = raw.strip()
        if not body:
            raise PlanningError("Invalid JSON from Gemini: response was empty or whitespace")

        if body.lower() in {"null", "undefined"}:
            raise PlanningError(f"Invalid JSON from Gemini: response was {body!r}")

        if body.startswith("```"):
            lines = body.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            body = "\n".join(lines).strip()

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Invalid JSON from Gemini: {exc}", cause=exc) from exc

        if isinstance(parsed, list):
            steps = parsed
        elif isinstance(parsed, dict) and "steps" in parsed:
            steps = parsed["steps"]
        else:
            raise PlanningError("Unexpected response shape: expected steps array")

        if not isinstance(steps, list):
            raise PlanningError("Unexpected response shape: expected steps array")
        if not steps:
            raise PlanningError("Gemini returned an empty steps array")

        NORMALISE_KEYS = {"step_type", "preferred_layer"}
        for step in steps:
            for key in NORMALISE_KEYS:
                if key in step and isinstance(step[key], str):
                    step[key] = step[key].lower()

        return steps

    def _build_user_message(self, goal_context: GoalContext) -> str:
        return (
            f"Platform: Windows 10/11\n"
            f"Shell: PowerShell\n"
            f"Package manager: winget\n"
            f"Path separator: \\\n"
            f"Home directory variable: $env:USERPROFILE\n"
            f"\n"
            f"Goal: {goal_context.goal}\n"
            f"Scope: {goal_context.scope}\n"
            f"Constraints: "
            f"{', '.join(goal_context.constraints) or 'none'}\n"
            f"Working directory: "
            f"{goal_context.working_directory or 'not specified'}\n"
            f"Complexity: {goal_context.estimated_complexity}\n"
        )

    def _normalize_paths(self, step_dict: dict) -> dict:
        params = step_dict.get("params")
        if not isinstance(params, dict):
            return step_dict

        path_keys = {"path", "source", "destination", "target_path", "working_dir", "cwd"}
        for key in path_keys:
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    params[key] = Path(value).as_posix()
                except Exception:
                    continue
        return step_dict

    def _build_system_prompt(self) -> str:
        return (
            "You are ALARA's planning engine. Analyze the goal and produce atomic, executable steps only.\n"
            "Atomic means each step does exactly one thing. Do not combine multiple actions in one step.\n\n"
            "Only use these operations:\n"
            "Filesystem:\n"
            "  create_directory  params: { path }\n"
            "  create_file       params: { path, content }\n"
            "  write_file        params: { path, content }\n"
            "  read_file         params: { path }\n"
            "  delete_file       params: { path }\n"
            "  move_file         params: { source, destination }\n"
            "  copy_file         params: { source, destination }\n"
            "  list_directory    params: { path }\n"
            "  search_files      params: { path, pattern }\n"
            "  check_path_exists params: { path }\n"
            "CLI:\n"
            "  run_command       params: { command, working_dir }\n"
            "App:\n"
            "  open_app          params: { app_name, args: [] }\n"
            "  close_app         params: { app_name }\n"
            "  focus_app         params: { app_name }\n"
            "System:\n"
            "  check_process     params: { process_name }\n"
            "  get_env_var       params: { name }\n"
            "  set_env_var       params: { name, value }\n\n"
            "Only use these verification_method values:\n"
            "  check_path_exists\n"
            "  check_exit_code_zero\n"
            "  check_process_running\n"
            "  check_file_contains\n"
            "  check_directory_not_empty\n"
            "  check_port_open\n"
            "  check_output_contains\n"
            "  none\n\n"
            "Path rules:\n"
            "- Always use forward slashes in all path values.\n"
            "- Never use backslashes.\n"
            "- If working_directory is provided, use it as the base for all relative paths and expand to absolute paths.\n\n"
            "Ordering and dependencies:\n"
            "- Steps must be ordered so dependencies always come first.\n"
            "- depends_on must only reference earlier step IDs.\n"
            "- If no dependencies, depends_on must be [].\n\n"
            "Layer selection:\n"
            "- filesystem operations -> preferred_layer: os_api\n"
            "- run_command -> preferred_layer: cli\n"
            "- open_app/close_app/focus_app -> preferred_layer: app_adapter\n"
            "- system operations -> preferred_layer: os_api\n\n"
            "Fallback strategy:\n"
            "- If a CLI step could alternatively be done via filesystem, set fallback_strategy to \"use_filesystem\".\n"
            "- If a step is optional and failure should not block the task, set fallback_strategy to \"skip_optional\".\n"
            "- If no fallback exists, set fallback_strategy to null.\n\n"
            "Respond with raw JSON only. No markdown. No code fences. No explanations.\n"
            "No text before or after the JSON. The response must parse with json.loads directly.\n"
            "Use exactly this schema:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "description": "...",\n'
            '      "step_type": "...",\n'
            '      "preferred_layer": "...",\n'
            '      "operation": "...",\n'
            '      "params": {},\n'
            '      "expected_outcome": "...",\n'
            '      "verification_method": "...",\n'
            '      "depends_on": [],\n'
            '      "fallback_strategy": null\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
