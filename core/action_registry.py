"""Registry schema and base action definitions for Alara."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from . import registry_loader


class NormalizationRule(BaseModel):
    """Param-level normalization rules executed in order.

    - ``param``: name of the parameter to normalize.
    - ``mapping``: case-insensitive value substitutions (e.g. "vs code" -> "vscode").
    - ``regex_extract``: optional regex to extract the value from the raw transcript.
    - ``fallback_value``: value to use if still missing after all rules.
    """

    param: str
    mapping: Dict[str, str] = Field(default_factory=dict)
    regex_extract: Optional[str] = None
    fallback_value: Any = None


class CrossActionCorrectionRule(BaseModel):
    """Cross-action correction rule declared in the registry.

    These are evaluated by the normalizer *after* per-action normalization and default
    injection. They allow patterns like:

    - ``run_command`` with no ``command`` and transcript starting with "open" →
      reroute to ``open_app``.
    """

    source_action: str
    target_action: str
    transcript_regex: Optional[str] = None
    required_missing_params: List[str] = Field(default_factory=list)
    required_params: Dict[str, Any] = Field(default_factory=dict)
    set_params: Dict[str, Any] = Field(default_factory=dict)


class ActionDefinition(BaseModel):
    """Single action declaration in the ACTION_REGISTRY."""

    name: str
    description: str
    params_schema: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Tuple[str, Dict[str, Any]]] = Field(default_factory=list)
    default_params: Dict[str, Any] = Field(default_factory=dict)
    aliases: List[str] = Field(default_factory=list)
    normalization_rules: List[NormalizationRule] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    tags: Set[str] = Field(default_factory=set)
    # Param key aliases, e.g. "app" -> "app_name"
    param_aliases: Dict[str, str] = Field(default_factory=dict)
    # Cross-action rules *originating* from this action.
    cross_action_corrections: List[CrossActionCorrectionRule] = Field(default_factory=list)


class ActionRegistry:
    """Immutable, precomputed registry indexes for O(1) lookups."""

    def __init__(self, definitions: Iterable[ActionDefinition]):
        self.actions_by_name: Dict[str, ActionDefinition] = {a.name: a for a in definitions}
        self.alias_to_name: Dict[str, str] = {}
        self.tag_to_actions: Dict[str, List[str]] = {}
        self.global_keywords: Set[str] = set()
        self.cross_action_rules: List[CrossActionCorrectionRule] = []

        for action in self.actions_by_name.values():
            # Action name itself is a canonical alias.
            self.alias_to_name[action.name] = action.name
            for alias in action.aliases:
                self.alias_to_name[alias] = action.name
            for tag in action.tags:
                self.tag_to_actions.setdefault(tag, []).append(action.name)
            self.global_keywords.update(action.keywords)
            self.cross_action_rules.extend(action.cross_action_corrections)

        for tag, names in self.tag_to_actions.items():
            # Deduplicate and sort for deterministic ordering.
            self.tag_to_actions[tag] = sorted(set(names))

    def resolve_action_name(self, name: str) -> str:
        """Resolve an action alias or canonical name to the canonical name."""
        return self.alias_to_name.get(name, name)

    def get(self, name: str) -> Optional[ActionDefinition]:
        return self.actions_by_name.get(name)

    def all_actions(self) -> List[ActionDefinition]:
        return list(self.actions_by_name.values())

    def action_names(self) -> List[str]:
        return sorted(self.actions_by_name.keys())

    def keywords(self) -> List[str]:
        return sorted(self.global_keywords)


def _base_definitions() -> List[ActionDefinition]:
    return [
        ActionDefinition(
            name="open_app",
            description="Open an application by name.",
            params_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
            examples=[
                ("open chrome", {"action": "open_app", "params": {"app_name": "chrome"}}),
                ("launch vscode", {"action": "open_app", "params": {"app_name": "vscode"}}),
                ("start spotify", {"action": "open_app", "params": {"app_name": "spotify"}}),
            ],
            default_params={},
            aliases=["launch_app", "start_app"],
            normalization_rules=[
                NormalizationRule(
                    param="app_name",
                    mapping={
                        "google chrome": "chrome",
                        "chrome browser": "chrome",
                        "visual studio code": "vscode",
                        "vs code": "vscode",
                        "microsoft edge": "edge",
                        "file explorer": "explorer",
                    },
                    regex_extract=r"(?:open|launch|start)\s+([a-zA-Z0-9 ._-]+)",
                )
            ],
            keywords=[
                "open",
                "launch",
                "start",
                "chrome",
                "edge",
                "firefox",
                "notepad",
                "vscode",
                "spotify",
                "discord",
            ],
            tags={"app", "window"},
            param_aliases={"app": "app_name", "application": "app_name", "program": "app_name"},
            cross_action_corrections=[
                CrossActionCorrectionRule(
                    source_action="run_command",
                    target_action="open_app",
                    transcript_regex=r"^\s*(open|launch|start)\b",
                    required_missing_params=["command"],
                ),
                CrossActionCorrectionRule(
                    source_action="unknown",
                    target_action="open_app",
                    transcript_regex=r"^\s*(open|launch|start)\b",
                ),
            ],
        ),
        ActionDefinition(
            name="close_app",
            description="Close an open application by name.",
            params_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
            examples=[
                ("close chrome", {"action": "close_app", "params": {"app_name": "chrome"}}),
                ("quit spotify", {"action": "close_app", "params": {"app_name": "spotify"}}),
            ],
            aliases=["quit_app", "exit_app"],
            normalization_rules=[
                NormalizationRule(
                    param="app_name",
                    mapping={"visual studio code": "vscode", "google chrome": "chrome"},
                    regex_extract=r"(?:close|quit|exit)\s+([a-zA-Z0-9 ._-]+)",
                )
            ],
            keywords=["close", "quit", "exit", "chrome", "spotify", "discord", "notepad"],
            tags={"app", "window"},
            param_aliases={"app": "app_name", "application": "app_name", "program": "app_name"},
        ),
        ActionDefinition(
            name="switch_app",
            description="Switch focus to an already-open application.",
            params_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
            examples=[
                ("switch to chrome", {"action": "switch_app", "params": {"app_name": "chrome"}}),
                ("focus discord", {"action": "switch_app", "params": {"app_name": "discord"}}),
            ],
            aliases=["focus_app", "activate_app"],
            normalization_rules=[
                NormalizationRule(
                    param="app_name",
                    mapping={"visual studio code": "vscode"},
                    regex_extract=r"(?:switch to|focus|activate)\s+([a-zA-Z0-9 ._-]+)",
                )
            ],
            keywords=["switch", "focus", "activate", "window", "app", "chrome", "discord"],
            tags={"app", "window"},
            param_aliases={"app": "app_name", "application": "app_name", "window": "app_name"},
        ),
        ActionDefinition(
            name="minimize_window",
            description="Minimize current window or a specified application window.",
            params_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": [],
            },
            examples=[
                ("minimize this", {"action": "minimize_window", "params": {"app_name": None}}),
                ("minimize chrome", {"action": "minimize_window", "params": {"app_name": "chrome"}}),
            ],
            aliases=["minimize_app", "hide_window"],
            default_params={"app_name": None},
            normalization_rules=[
                NormalizationRule(
                    param="app_name",
                    mapping={"current": ""},
                    regex_extract=r"minimize\s+([a-zA-Z0-9 ._-]+)",
                    fallback_value=None,
                )
            ],
            keywords=["minimize", "hide", "window"],
            tags={"window", "app"},
            param_aliases={"app": "app_name", "window": "app_name"},
        ),
        ActionDefinition(
            name="maximize_window",
            description="Maximize current window or a specified application window.",
            params_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": [],
            },
            examples=[
                ("maximize this", {"action": "maximize_window", "params": {"app_name": None}}),
                ("maximize edge", {"action": "maximize_window", "params": {"app_name": "edge"}}),
            ],
            aliases=["expand_window", "maximize_app"],
            default_params={"app_name": None},
            normalization_rules=[
                NormalizationRule(
                    param="app_name",
                    regex_extract=r"(?:maximize|expand)\s+([a-zA-Z0-9 ._-]+)",
                    fallback_value=None,
                )
            ],
            keywords=["maximize", "expand", "window"],
            tags={"window", "app"},
            param_aliases={"app": "app_name", "window": "app_name"},
        ),
        ActionDefinition(
            name="close_window",
            description="Close current window or a specific app window.",
            params_schema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": [],
            },
            examples=[
                ("close this window", {"action": "close_window", "params": {"app_name": None}}),
                ("close edge window", {"action": "close_window", "params": {"app_name": "edge"}}),
            ],
            aliases=["shutdown_window", "exit_window"],
            default_params={"app_name": None},
            normalization_rules=[
                NormalizationRule(
                    param="app_name",
                    regex_extract=r"(?:close|exit)\s+([a-zA-Z0-9 ._-]+)\s*(?:window)?",
                    fallback_value=None,
                )
            ],
            keywords=["close window", "exit window", "close this"],
            tags={"window", "app"},
            param_aliases={"app": "app_name", "window": "app_name"},
        ),
        ActionDefinition(
            name="take_screenshot",
            description="Capture a screenshot and optionally copy it to clipboard.",
            params_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": ["string", "null"]},
                    "include_clipboard": {"type": "boolean"},
                },
                "required": [],
            },
            examples=[
                ("take a screenshot", {"action": "take_screenshot", "params": {"filename": None, "include_clipboard": False}}),
                ("screenshot to clipboard", {"action": "take_screenshot", "params": {"filename": None, "include_clipboard": True}}),
            ],
            aliases=["screenshot", "capture_screen"],
            default_params={"filename": None, "include_clipboard": False},
            normalization_rules=[
                NormalizationRule(
                    param="include_clipboard",
                    mapping={"clipboard": "true"},
                    regex_extract=r"\bclipboard\b",
                    fallback_value=False,
                )
            ],
            keywords=["screenshot", "capture", "screen", "clipboard"],
            tags={"system", "file"},
        ),
        ActionDefinition(
            name="open_file",
            description="Open a file path with default app.",
            params_schema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
            examples=[
                ("open file C:\\temp\\notes.txt", {"action": "open_file", "params": {"filepath": "C:\\temp\\notes.txt"}}),
                ("open readme dot md", {"action": "open_file", "params": {"filepath": "readme.md"}}),
            ],
            aliases=["open_document"],
            normalization_rules=[
                NormalizationRule(
                    param="filepath",
                    regex_extract=r"(?:open file|open)\s+(.+\.[a-zA-Z0-9]{1,5})",
                )
            ],
            keywords=["open file", "document", "txt", "pdf", "readme"],
            tags={"file"},
            param_aliases={"file": "filepath", "path": "filepath"},
        ),
        ActionDefinition(
            name="open_folder",
            description="Open a folder path in File Explorer.",
            params_schema={
                "type": "object",
                "properties": {"folder_path": {"type": "string"}},
                "required": ["folder_path"],
            },
            examples=[
                ("open downloads folder", {"action": "open_folder", "params": {"folder_path": "downloads"}}),
                ("open C:\\Users\\Public", {"action": "open_folder", "params": {"folder_path": "C:\\Users\\Public"}}),
            ],
            aliases=["browse_folder", "show_folder"],
            normalization_rules=[
                NormalizationRule(
                    param="folder_path",
                    mapping={"downloads": "C:\\Users\\{username}\\Downloads", "desktop": "C:\\Users\\{username}\\Desktop"},
                    regex_extract=r"(?:open|browse)\s+(.+?)\s*(?:folder)?$",
                )
            ],
            keywords=["open folder", "browse", "directory", "downloads", "desktop", "documents"],
            tags={"file"},
            param_aliases={"folder": "folder_path", "path": "folder_path"},
        ),
        ActionDefinition(
            name="search_files",
            description="Search files by query and optional location.",
            params_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "search_location": {"type": ["string", "null"]}},
                "required": ["query"],
            },
            examples=[
                ("find all pdfs", {"action": "search_files", "params": {"query": "pdf", "search_location": None}}),
                ("search photos in downloads", {"action": "search_files", "params": {"query": "photos", "search_location": "downloads"}}),
            ],
            aliases=["find_files"],
            default_params={"search_location": None},
            normalization_rules=[
                NormalizationRule(param="query", regex_extract=r"(?:find|search(?: for)?)\s+(.+?)(?:\s+in\s+.+)?$"),
                NormalizationRule(param="search_location", regex_extract=r"\bin\s+([a-zA-Z0-9:\\._ -]+)$", fallback_value=None),
            ],
            keywords=["search files", "find", "pdf", "documents", "folder"],
            tags={"file"},
            param_aliases={"find": "query", "search": "query", "location": "search_location"},
        ),
        ActionDefinition(
            name="run_command",
            description="Execute a shell command in terminal.",
            params_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}, "wait": {"type": "boolean"}},
                "required": ["command"],
            },
            examples=[
                ("run pip install requests", {"action": "run_command", "params": {"command": "pip install requests", "wait": True}}),
                ("execute git status", {"action": "run_command", "params": {"command": "git status", "wait": True}}),
            ],
            aliases=["execute_command", "terminal_command"],
            default_params={"wait": True},
            normalization_rules=[
                NormalizationRule(param="command", regex_extract=r"(?:run|execute)\s+(.+)$"),
                NormalizationRule(param="wait", mapping={"background": "false"}, fallback_value=True),
            ],
            keywords=["run", "execute", "terminal", "shell", "command", "powershell"],
            tags={"system", "terminal"},
            param_aliases={"cmd": "command"},
        ),
        ActionDefinition(
            name="browser_new_tab",
            description="Open a new browser tab optionally with URL.",
            params_schema={
                "type": "object",
                "properties": {"browser": {"type": "string"}, "url": {"type": ["string", "null"]}},
                "required": [],
            },
            examples=[
                ("new tab", {"action": "browser_new_tab", "params": {"browser": "default", "url": None}}),
                ("open new tab in chrome", {"action": "browser_new_tab", "params": {"browser": "chrome", "url": None}}),
            ],
            aliases=["new_tab"],
            default_params={"browser": "default", "url": None},
            normalization_rules=[
                NormalizationRule(param="browser", regex_extract=r"\bin\s+(chrome|edge|firefox)\b", fallback_value="default"),
                NormalizationRule(param="url", regex_extract=r"(https?://\S+)", fallback_value=None),
            ],
            keywords=["new tab", "browser tab", "chrome", "edge", "firefox"],
            tags={"browser"},
        ),
        ActionDefinition(
            name="browser_navigate",
            description="Navigate browser to target URL.",
            params_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}, "browser": {"type": "string"}},
                "required": ["url"],
            },
            examples=[
                ("go to github.com", {"action": "browser_navigate", "params": {"url": "github.com", "browser": "default"}}),
                ("open youtube in chrome", {"action": "browser_navigate", "params": {"url": "youtube.com", "browser": "chrome"}}),
            ],
            aliases=["navigate", "go_to_url"],
            default_params={"browser": "default"},
            normalization_rules=[
                NormalizationRule(param="url", regex_extract=r"(?:go to|open|navigate to)\s+([a-zA-Z0-9./:_-]+)"),
                NormalizationRule(param="browser", regex_extract=r"\bin\s+(chrome|edge|firefox)\b", fallback_value="default"),
            ],
            keywords=["navigate", "go to", "website", "url", "open site"],
            tags={"browser"},
        ),
        ActionDefinition(
            name="browser_search",
            description="Search the web for a query.",
            params_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "browser": {"type": "string"}},
                "required": ["query"],
            },
            examples=[
                ("search for weather tomorrow", {"action": "browser_search", "params": {"query": "weather tomorrow", "browser": "default"}}),
                ("google python dataclasses", {"action": "browser_search", "params": {"query": "python dataclasses", "browser": "default"}}),
            ],
            aliases=["web_search"],
            default_params={"browser": "default"},
            normalization_rules=[
                NormalizationRule(param="query", regex_extract=r"(?:search(?: for)?|google|bing)\s+(.+)$"),
                NormalizationRule(param="browser", regex_extract=r"\bin\s+(chrome|edge|firefox)\b", fallback_value="default"),
            ],
            keywords=["search web", "google", "bing", "query", "look up"],
            tags={"browser"},
        ),
        ActionDefinition(
            name="browser_close_tab",
            description="Close current or specific browser tab.",
            params_schema={
                "type": "object",
                "properties": {"browser": {"type": "string"}, "tab_index": {"type": ["integer", "null"]}},
                "required": [],
            },
            examples=[
                ("close tab", {"action": "browser_close_tab", "params": {"browser": "default", "tab_index": None}}),
                ("close tab 3 in chrome", {"action": "browser_close_tab", "params": {"browser": "chrome", "tab_index": 3}}),
            ],
            aliases=["close_browser_tab"],
            default_params={"browser": "default", "tab_index": None},
            normalization_rules=[
                NormalizationRule(param="browser", regex_extract=r"\bin\s+(chrome|edge|firefox)\b", fallback_value="default"),
                NormalizationRule(param="tab_index", regex_extract=r"\btab\s+(\d+)\b", fallback_value=None),
            ],
            keywords=["close tab", "browser", "tab", "chrome"],
            tags={"browser"},
        ),
        ActionDefinition(
            name="vscode_open_file",
            description="Open file in Visual Studio Code.",
            params_schema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
            examples=[
                ("open main.py in vscode", {"action": "vscode_open_file", "params": {"filepath": "main.py"}}),
                ("edit settings json in code", {"action": "vscode_open_file", "params": {"filepath": "settings.json"}}),
            ],
            aliases=["code_open_file"],
            normalization_rules=[
                NormalizationRule(param="filepath", regex_extract=r"(?:open|edit)\s+(.+?)\s+(?:in\s+)?(?:vs\s*code|vscode|code)$")
            ],
            keywords=["vscode", "vs code", "open file", "edit file", "code editor"],
            tags={"vscode", "editor"},
            param_aliases={"file": "filepath", "path": "filepath"},
        ),
        ActionDefinition(
            name="vscode_new_terminal",
            description="Create new VS Code terminal and optionally run command.",
            params_schema={
                "type": "object",
                "properties": {"command": {"type": ["string", "null"]}},
                "required": [],
            },
            examples=[
                ("new terminal in vscode", {"action": "vscode_new_terminal", "params": {"command": None}}),
                ("open vscode terminal and run npm test", {"action": "vscode_new_terminal", "params": {"command": "npm test"}}),
            ],
            aliases=["code_terminal"],
            default_params={"command": None},
            normalization_rules=[
                NormalizationRule(param="command", regex_extract=r"(?:run|execute)\s+(.+)$", fallback_value=None)
            ],
            keywords=["vscode terminal", "new terminal", "integrated terminal", "code terminal"],
            tags={"vscode", "editor"},
        ),
        ActionDefinition(
            name="vscode_search",
            description="Search text across files in VS Code.",
            params_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            examples=[
                ("search for TODO in vscode", {"action": "vscode_search", "params": {"query": "TODO"}}),
                ("find hello world in code", {"action": "vscode_search", "params": {"query": "hello world"}}),
            ],
            aliases=["code_search"],
            normalization_rules=[
                NormalizationRule(param="query", regex_extract=r"(?:search(?: for)?|find)\s+(.+?)(?:\s+in\s+(?:vs\s*code|vscode|code))?$")
            ],
            keywords=["vscode search", "find in files", "search text"],
            tags={"vscode", "editor"},
            param_aliases={"find": "query"},
        ),
        ActionDefinition(
            name="volume_up",
            description="Increase system volume.",
            params_schema={
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": [],
            },
            examples=[
                ("volume up", {"action": "volume_up", "params": {"amount": 10}}),
                ("increase volume by 20", {"action": "volume_up", "params": {"amount": 20}}),
            ],
            aliases=["increase_volume", "louder"],
            default_params={"amount": 10},
            normalization_rules=[NormalizationRule(param="amount", regex_extract=r"\bby\s+(\d+)\b", fallback_value=10)],
            keywords=["volume up", "increase volume", "louder", "sound"],
            tags={"system", "audio"},
        ),
        ActionDefinition(
            name="volume_down",
            description="Decrease system volume.",
            params_schema={
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": [],
            },
            examples=[
                ("volume down", {"action": "volume_down", "params": {"amount": 10}}),
                ("decrease volume by 15", {"action": "volume_down", "params": {"amount": 15}}),
            ],
            aliases=["decrease_volume", "quieter"],
            default_params={"amount": 10},
            normalization_rules=[NormalizationRule(param="amount", regex_extract=r"\bby\s+(\d+)\b", fallback_value=10)],
            keywords=["volume down", "decrease volume", "quieter", "lower sound"],
            tags={"system", "audio"},
        ),
        ActionDefinition(
            name="volume_mute",
            description="Mute or unmute system volume.",
            params_schema={
                "type": "object",
                "properties": {"mute": {"type": "boolean"}},
                "required": [],
            },
            examples=[
                ("mute volume", {"action": "volume_mute", "params": {"mute": True}}),
                ("unmute", {"action": "volume_mute", "params": {"mute": False}}),
            ],
            aliases=["mute", "toggle_mute"],
            default_params={"mute": True},
            normalization_rules=[
                NormalizationRule(param="mute", mapping={"unmute": "false", "mute": "true"}, regex_extract=r"\b(unmute|mute)\b", fallback_value=True)
            ],
            keywords=["mute", "unmute", "volume mute", "audio mute"],
            tags={"system", "audio"},
        ),
        ActionDefinition(
            name="lock_screen",
            description="Lock the Windows screen.",
            params_schema={"type": "object", "properties": {}, "required": []},
            examples=[
                ("lock screen", {"action": "lock_screen", "params": {}}),
                ("lock my computer", {"action": "lock_screen", "params": {}}),
            ],
            aliases=["sleep_screen", "lock_computer"],
            normalization_rules=[],
            keywords=["lock screen", "lock computer", "secure desktop"],
            tags={"system"},
        ),
        ActionDefinition(
            name="unknown",
            description="Unknown or unsupported command.",
            params_schema={
                "type": "object",
                "properties": {"reason": {"type": ["string", "null"]}},
                "required": [],
            },
            examples=[
                ("blabla", {"action": "unknown", "params": {"reason": "unrecognized"}}),
            ],
            default_params={"reason": None},
            aliases=["none", "unsupported"],
            normalization_rules=[
                NormalizationRule(param="reason", fallback_value="could not classify command"),
            ],
            keywords=[],
            tags={"fallback"},
            cross_action_corrections=[
                CrossActionCorrectionRule(
                    source_action="unknown",
                    target_action="browser_search",
                    transcript_regex=r"^\s*(search|google|bing)\b",
                    set_params={},
                ),
                CrossActionCorrectionRule(
                    source_action="unknown",
                    target_action="volume_up",
                    transcript_regex=r"\b(volume up|louder)\b",
                ),
            ],
        ),
    ]


def _load_custom_definitions() -> List[ActionDefinition]:
    """Load and validate custom actions from optional YAML/JSON file.

    Validation errors are logged but do not crash startup.
    """
    raw_defs = registry_loader.load_custom_actions(
        path=os.getenv("ALARA_CUSTOM_ACTIONS_PATH")
    )
    if not raw_defs:
        return []

    custom_defs: List[ActionDefinition] = []
    for idx, raw in enumerate(raw_defs):
        try:
            custom_defs.append(ActionDefinition.model_validate(raw))
        except ValidationError as exc:
            logger.warning(
                "Invalid custom action definition at index {}: {}", idx, exc
            )
    return custom_defs


@lru_cache(maxsize=1)
def get_registry() -> ActionRegistry:
    """Return the merged ACTION_REGISTRY (base + custom) as a singleton.

    Custom actions override base actions on name collisions.
    """
    base_defs = _base_definitions()
    custom_defs = _load_custom_definitions()

    merged: Dict[str, ActionDefinition] = {a.name: a for a in base_defs}
    for custom in custom_defs:
        logger.info("Registering custom action: {}", custom.name)
        merged[custom.name] = custom

    registry = ActionRegistry(merged.values())
    logger.info(
        "ACTION_REGISTRY initialized with {} actions ({} custom)",
        len(registry.actions_by_name),
        len(custom_defs),
    )
    return registry


# Public, importable single source of truth.
ACTION_REGISTRY: ActionRegistry = get_registry()


