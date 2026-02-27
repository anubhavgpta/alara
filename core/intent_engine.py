"""
alara/core/intent_engine.py

Ollama-first intent parser for ALARA.
No deterministic classifier is used for primary intent selection.
"""

import json
import os
import re
import time
import ast
from typing import Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, validator

load_dotenv()


class Action(BaseModel):
    action: str
    params: dict[str, Any]
    confidence: float
    raw_text: str

    @validator("action")
    def validate_action(cls, v):
        valid_actions = {
            "open_app",
            "close_app",
            "switch_app",
            "minimize_window",
            "maximize_window",
            "close_window",
            "take_screenshot",
            "open_file",
            "open_folder",
            "search_files",
            "run_command",
            "browser_new_tab",
            "browser_navigate",
            "browser_search",
            "browser_close_tab",
            "vscode_open_file",
            "vscode_new_terminal",
            "vscode_search",
            "volume_up",
            "volume_down",
            "volume_mute",
            "lock_screen",
            "unknown",
        }
        if v not in valid_actions:
            logger.warning(f"Invalid action '{v}', defaulting to 'unknown'")
            return "unknown"
        return v

    @validator("confidence")
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))


SYSTEM_PROMPT = """
You are ALARA's intent classifier for Windows voice commands.
Return one JSON object only. No markdown. No prose.

Allowed actions:
open_app, close_app, switch_app,
minimize_window, maximize_window, close_window, take_screenshot,
open_file, open_folder, search_files,
run_command,
browser_new_tab, browser_navigate, browser_search, browser_close_tab,
vscode_open_file, vscode_new_terminal, vscode_search,
volume_up, volume_down, volume_mute, lock_screen,
unknown

Return exactly:
{"action":"<allowed_action>","params":{},"confidence":0.95}

Rules:
- Normalize app names: "vs code" -> "vscode", "chrome" -> "google chrome", "terminal" -> "windows terminal".
- Terminal/dev commands (git, npm, pip, python, pytest, docker, docker-compose, uv, pnpm, yarn) should be run_command.
- Params may include contextual keys when needed. For terminal commands, include cwd when a folder/directory/project location is mentioned. For search actions, refine query based on context clues in the sentence.
- For compound commands, resolve the full intent into one supported action object. Example pattern: "open a terminal in X and run Y" should be one run_command with both command and cwd populated.
- "search for X" defaults to browser_search unless editor/code context indicates vscode_search.
- If command includes a domain/url (e.g. google.com), use browser_navigate with https://.
- If command asks to open a specific file in VS Code, use vscode_open_file.
- If command asks to open a specific file without VS Code context, use open_file.
- Humans speak loosely. Infer the most likely developer intent even when phrasing is informal, abbreviated, or indirect.
- Do not return unknown if a reasonable mapping exists to a supported action.
- Use unknown only when no supported action can be reasonably inferred.
- confidence in [0, 1].

Few-shot examples:
User: "open VS Code"
{"action":"open_app","params":{"app_name":"vscode"},"confidence":0.95}

User: "run git status"
{"action":"run_command","params":{"command":"git status"},"confidence":0.95}

User: "search for Python tutorials"
{"action":"browser_search","params":{"query":"Python tutorials"},"confidence":0.95}

User: "open utils.py in VS Code"
{"action":"vscode_open_file","params":{"query":"utils.py"},"confidence":0.95}

User: "open a terminal in my projects folder and run git status"
{"action":"run_command","params":{"command":"git status","cwd":"~/Desktop/Projects"},"confidence":0.95}

User: "search for README files"
{"action":"search_files","params":{"query":"README*"},"confidence":0.95}

User: "fire up chrome"
{"action":"open_app","params":{"app_name":"google chrome"},"confidence":0.95}

User: "pull the latest code"
{"action":"run_command","params":{"command":"git pull"},"confidence":0.90}

User: "look up react hooks"
{"action":"browser_search","params":{"query":"react hooks"},"confidence":0.95}

User: "what's the weather"
{"action":"unknown","params":{"reason":"weather queries not supported"},"confidence":0.70}
""".strip()


class IntentEngine:
    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            self.client = None
        self.max_retries = 4
        self.retry_delay = 2
        self._check_gemini()

    def _check_gemini(self):
        if not self.api_key:
            logger.error("GEMINI_API_KEY is not set.")
        else:
            logger.success(f"Gemini client initialized model={self.model}")

    def _make_action(self, action: str, params: dict[str, Any], confidence: float, raw_text: str) -> Action:
        return Action(action=action, params=params, confidence=confidence, raw_text=raw_text)

    def _normalize_action(self, parsed: dict, transcription: str) -> Action:
        original_action = str(parsed.get("action", "unknown")).strip()
        action = original_action
        params = parsed.get("params", {})
        if not isinstance(params, dict):
            params = {}
        confidence = parsed.get("confidence", 0.5)
        t = transcription.strip().lower()

        def _folder_alias_to_path(alias: str) -> Optional[str]:
            alias_clean = alias.strip().lower()
            home = os.path.expanduser("~")
            mapping = {
                "projects": os.path.join(home, "Desktop", "Projects"),
                "project": os.path.join(home, "Desktop", "Projects"),
                "downloads": os.path.join(home, "Downloads"),
                "desktop": os.path.join(home, "Desktop"),
                "documents": os.path.join(home, "Documents"),
                "home": home,
                "this project": os.getcwd(),
                "current project": os.getcwd(),
            }
            return mapping.get(alias_clean)

        valid_actions = {
            "open_app",
            "close_app",
            "switch_app",
            "minimize_window",
            "maximize_window",
            "close_window",
            "take_screenshot",
            "open_file",
            "open_folder",
            "search_files",
            "run_command",
            "browser_new_tab",
            "browser_navigate",
            "browser_search",
            "browser_close_tab",
            "vscode_open_file",
            "vscode_new_terminal",
            "vscode_search",
            "volume_up",
            "volume_down",
            "volume_mute",
            "lock_screen",
            "unknown",
        }

        # Handle combined terminal commands, e.g.:
        # "open a terminal in my projects folder and run git status"
        run_match = re.search(r"\brun\s+(.+)$", transcription, flags=re.IGNORECASE)
        if run_match and ("terminal" in t or "powershell" in t):
            run_command = run_match.group(1).strip().strip(".")
            if run_command and (not params.get("command") or action in {"unknown", "open_app"}):
                action = "run_command"
                params = {"command": run_command}

            folder_match = re.search(r"\bin\s+(?:my\s+)?(.+?)\s+folder\b", transcription, flags=re.IGNORECASE)
            if folder_match:
                alias = folder_match.group(1).strip().lower()
                resolved = _folder_alias_to_path(alias)
                if resolved:
                    params["cwd"] = resolved

        # Canonicalize common non-schema actions produced by LLM.
        if action not in valid_actions:
            known_apps = {"vscode", "google chrome", "chrome", "firefox", "notepad", "slack", "terminal", "windows terminal"}
            action_l = action.lower()
            if action_l in known_apps:
                if t.startswith("open ") or t.startswith("launch "):
                    action = "open_app"
                    params = {"app_name": action_l}
                elif t.startswith("close ") or t.startswith("shutdown "):
                    action = "close_app"
                    params = {"app_name": action_l}
                elif t.startswith("switch to "):
                    action = "switch_app"
                    params = {"app_name": action_l}
                else:
                    action = "open_app"
                    params = {"app_name": action_l}
            elif action_l == "vscode_open_url":
                action = "browser_navigate"
            else:
                action = "unknown"

        # Normalize param key aliases.
        if "app" in params and "app_name" not in params:
            params["app_name"] = params.pop("app")
        if "folder_name" in params and "path" not in params:
            params["path"] = params.pop("folder_name")
        if "folder_path" in params and "path" not in params:
            params["path"] = params.pop("folder_path")
        if "file_name" in params and "path" not in params and "query" not in params:
            params["path"] = params["file_name"]
        if "file" in params:
            if action == "open_file" and "path" not in params:
                params["path"] = params.pop("file")
            elif action == "vscode_open_file" and "query" not in params:
                params["query"] = params.pop("file")
            elif "path" not in params and "query" not in params:
                params["path"] = params.pop("file")

        # Fix obvious misroutes with transcription context.
        if action == "run_command" and not params.get("command"):
            target = re.sub(r"^(open|launch|close|shutdown|switch to)\s+", "", t).strip()
            if t.startswith("open ") or t.startswith("launch "):
                action = "open_app"
                params = {"app_name": target}
            elif t.startswith("close ") or t.startswith("shutdown "):
                action = "close_app"
                params = {"app_name": target}
            elif t.startswith("switch to "):
                action = "switch_app"
                params = {"app_name": target}

        if action == "run_command" and str(params.get("command", "")).strip().lower() == "cls":
            params["command"] = "clear"
        if action == "run_command" and str(params.get("command", "")).strip().lower() == "exit" and "terminal" in t:
            action = "close_app"
            params = {"app_name": "windows terminal"}
        if action == "run_command" and str(params.get("command", "")).strip().lower() == "windows terminal" and "open terminal" in t:
            action = "open_app"
            params = {"app_name": "windows terminal"}
        if action == "run_command":
            cmd_val = str(params.get("command", "")).strip()
            if t.startswith("open ") and cmd_val and "." in cmd_val:
                action = "open_file"
                params = {"path": cmd_val}

        if action == "open_file" and "path" not in params:
            m = re.search(r"([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_]+)", transcription)
            if m:
                params["path"] = m.group(1)

        if action == "vscode_open_file" and "query" not in params:
            m = re.search(r"([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_]+)", transcription)
            if m:
                params["query"] = m.group(1)

        if action == "browser_search":
            q = str(params.get("query", "")).lower()
            if "files" in q:
                action = "search_files"
                if "python" in q:
                    params["query"] = "*.py"
                elif "readme" in q:
                    params["query"] = "README*"
            if "search for readme files" in t:
                action = "search_files"
                params["query"] = "README*"
            code_terms = ("function", "class", "method", "definition", "import")
            if any(term in t for term in code_terms):
                action = "vscode_search"
                if t.startswith("find the "):
                    params["query"] = transcription.lower().replace("find the ", "", 1).strip()
                elif t.startswith("find "):
                    params["query"] = transcription.lower().replace("find ", "", 1).strip()
                elif t.startswith("search for "):
                    params["query"] = transcription.replace("search for ", "", 1).strip()

        # Derive app target directly from command phrase when app actions are selected.
        if action in {"open_app", "close_app", "switch_app"}:
            target = ""
            if t.startswith("open ") or t.startswith("launch "):
                target = re.sub(r"^(open|launch)\s+", "", t).strip()
            elif t.startswith("close ") or t.startswith("shutdown "):
                target = re.sub(r"^(close|shutdown)\s+", "", t).strip()
            elif t.startswith("switch to "):
                target = re.sub(r"^switch to\s+", "", t).strip()
            target = re.sub(r"^(a|an|the)\s+", "", target).strip()
            # "open terminal in my projects folder" -> "terminal"
            target = re.sub(r"\s+in\s+.+?\s+folder\b.*$", "", target).strip()
            if "terminal" in target:
                target = "windows terminal"
            target = re.sub(r"\s+app$", "", target).strip()
            if target:
                params["app_name"] = target

        if action == "unknown":
            if t.startswith("find "):
                q = transcription.strip()
                q = re.sub(r"^find\s+(the\s+)?", "", q, flags=re.IGNORECASE).strip()
                return self._make_action("vscode_search", {"query": q}, float(confidence), transcription)
            if t.startswith("switch to "):
                action = "switch_app"
                params = {"app_name": t.replace("switch to ", "", 1).strip()}
            elif t.startswith("open ") or t.startswith("launch "):
                target = re.sub(r"^(open|launch)\s+", "", t).strip()
                target = re.sub(r"^(a|an|the)\s+", "", target).strip()
                target = re.sub(r"\s+in\s+.+?\s+folder\b.*$", "", target).strip()
                if "." in target and ("vs code" in t or "vscode" in t):
                    action = "vscode_open_file"
                    params = {"query": target.split()[0]}
                elif "." in target:
                    action = "open_file"
                    params = {"path": target.split()[0]}
                elif target == "github":
                    action = "browser_navigate"
                    params = {"url": "https://github.com"}
                elif "terminal" in target:
                    action = "open_app"
                    params = {"app_name": "windows terminal"}
                else:
                    action = "open_app"
                    params = {"app_name": target}
            elif t.startswith("close ") or t.startswith("shutdown "):
                target = re.sub(r"^(close|shutdown)\s+", "", t).strip()
                action = "close_app"
                params = {"app_name": target}

        # App name normalization.
        if action in {"open_app", "close_app", "switch_app"}:
            app = str(params.get("app_name", "")).strip().lower()
            aliases = {
                "vs code": "vscode",
                "visual studio code": "vscode",
                "chrome": "google chrome",
                "terminal": "windows terminal",
            }
            if app in aliases:
                params["app_name"] = aliases[app]

        # Correct common misroutes using command context.
        if action == "open_app" and t.startswith("switch to "):
            action = "switch_app"
        if action == "close_window" and "terminal" in t:
            action = "close_app"
            params = {"app_name": "windows terminal"}
        if action == "open_app" and t.startswith("python "):
            action = "run_command"
            params = {"command": transcription.strip()}
        if action == "open_app" and t == "clear terminal":
            action = "run_command"
            params = {"command": "clear"}
        if action == "run_command" and t.startswith("switch to "):
            action = "switch_app"
            params = {"app_name": re.sub(r"^switch to\s+", "", t).strip()}

        # Normalize open_file/open_folder shape after key aliasing.
        if action == "open_file" and "path" not in params and "file_name" in params:
            params["path"] = params["file_name"]
        if action == "open_folder" and "path" not in params:
            if "folder_name" in params:
                params["path"] = params["folder_name"]
            elif "folder_path" in params:
                params["path"] = params["folder_path"]

        if action in {"volume_up", "volume_down"} and "amount" not in params:
            params["amount"] = 10

        if action == "unknown" and "reason" not in params:
            params["reason"] = "unsupported or unclear command"

        return self._make_action(action, params, float(confidence), transcription)

    def _extract_json(self, raw_content: str) -> Optional[dict]:
        def _try_load(text: str) -> Optional[dict]:
            value = text.strip()
            if not value:
                return None

            attempts = [value]
            attempts.append(re.sub(r",\s*([}\]])", r"\1", value))
            attempts.append(re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', value))
            attempts.append(value.replace("'", '"'))

            for attempt in attempts:
                try:
                    parsed = json.loads(attempt)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                return item
                except Exception:
                    pass
                try:
                    parsed = ast.literal_eval(attempt)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                return item
                except Exception:
                    pass
            return None

        cleaned = (raw_content or "").strip()
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        cleaned = cleaned.strip()

        candidates = []
        if cleaned:
            candidates.append(cleaned)

        # Extract candidate JSON-like objects, prioritizing smaller matches.
        for match in re.findall(r"\{[\s\S]*?\}", cleaned):
            snippet = match.strip()
            if snippet:
                candidates.append(snippet)

        for candidate in candidates:
            parsed = _try_load(candidate)
            if parsed is not None:
                return parsed
        return None

    def _query_gemini(self, transcription: str) -> str:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized")

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f'User command: "{transcription}"\n'
            "Return only one valid JSON object."
        )

        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=220,
                response_mime_type="application/json",
            ),
        )

        raw = (getattr(response, "text", None) or "").strip()
        if not raw and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts or []
            raw = "".join(getattr(p, "text", "") or "" for p in parts).strip()

        # Force retry when Gemini truncates a non-JSON response.
        if raw and "{" not in raw and "}" not in raw:
            finish_reason = None
            if getattr(response, "candidates", None):
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
            if str(finish_reason).upper().endswith("MAX_TOKENS"):
                raise RuntimeError("truncated non-json response from gemini")

        return raw

    def parse(self, transcription: str) -> Action:
        if not transcription.strip():
            return self._make_action("unknown", {"reason": "empty transcription"}, 0.0, transcription)

        last_error = "unknown error"
        saw_json_error = False

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Parsing intent with Gemini: attempt {attempt}/{self.max_retries}")
                raw = self._query_gemini(transcription)
                parsed = self._extract_json(raw)
                if parsed is None:
                    last_error = "invalid JSON from LLM"
                    saw_json_error = True
                    time.sleep(0.5)
                    continue

                action = self._normalize_action(parsed, transcription)
                logger.info(
                    f"Intent (gemini): {action.action} | params={action.params} | "
                    f"confidence={action.confidence:.2f}"
                )
                return action

            except Exception as e:
                err_text = str(e).lower()
                status_code = None
                for attr in ("status_code", "code"):
                    if hasattr(e, attr):
                        try:
                            status_code = int(getattr(e, attr))
                        except Exception:
                            status_code = None
                        break

                if status_code is not None:
                    last_error = f"gemini error: {status_code}"
                elif "429" in err_text:
                    last_error = "gemini error: 429"
                else:
                    last_error = str(e)

                if ((status_code is not None and status_code >= 500) or "temporar" in err_text) and attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                    continue
                break

        # Deterministic fallback: run normalization heuristics even when LLM output
        # is missing or malformed, so common commands still execute.
        fallback_confidence = 0.55 if saw_json_error else 0.45
        fallback = self._normalize_action(
            {"action": "unknown", "params": {"reason": last_error}, "confidence": fallback_confidence},
            transcription,
        )
        if saw_json_error and fallback.action != "unknown":
            logger.warning(
                f"LLM JSON invalid after retries; used fallback intent: {fallback.action} "
                f"with confidence={fallback.confidence:.2f}"
            )
        else:
            logger.error(f"Intent engine failed after retries: {last_error}")
        logger.info(
            f"Intent fallback: {fallback.action} | params={fallback.params} | "
            f"confidence={fallback.confidence:.2f}"
        )
        return fallback
