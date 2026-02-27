"""
Week 5-6 integration tests.

These tests validate command routing and integration invocation without launching
real apps by mocking subprocess and shell discovery functions.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.intent_engine import IntentEngine
from integrations.terminal import TerminalIntegration
from integrations.windows_os import WindowsOSIntegration


class TestWeek56Integrations(unittest.TestCase):
    def test_parse_falls_back_when_llm_returns_invalid_json(self):
        engine = IntentEngine()
        with patch.object(engine, "_query_gemini", return_value="not-json-response"):
            engine.max_retries = 1
            action = engine.parse("open a terminal in my projects folder and run git status")

        self.assertEqual(action.action, "run_command")
        self.assertEqual(action.params.get("command"), "git status")
        self.assertTrue(action.params.get("cwd", "").lower().endswith("desktop\\projects"))
        self.assertGreaterEqual(action.confidence, 0.5)

    def test_parse_fallback_terminal_open_in_folder_sets_windows_terminal(self):
        engine = IntentEngine()
        with patch.object(engine, "_query_gemini", return_value="not-json-response"):
            engine.max_retries = 1
            action = engine.parse("open a terminal in my projects folder")

        self.assertEqual(action.action, "open_app")
        self.assertEqual(action.params.get("app_name"), "windows terminal")
        self.assertGreaterEqual(action.confidence, 0.5)

    def test_windows_expand_path_maps_desktop_alias(self):
        integration = WindowsOSIntegration()
        path = integration._expand_path("desktop")
        self.assertEqual(path, (Path.home() / "Desktop").resolve())

    def test_windows_expand_path_can_resolve_repo_relative_when_cwd_is_parent(self):
        integration = WindowsOSIntegration()
        repo_parent = Path(__file__).resolve().parents[2]
        with patch("pathlib.Path.cwd", return_value=repo_parent):
            path = integration._expand_path("requirements.txt")
        self.assertEqual(path.name, "requirements.txt")
        self.assertTrue(path.exists())

    def test_terminal_run_command_uses_windows_terminal_when_available(self):
        integration = TerminalIntegration()

        with patch("integrations.terminal.shutil.which", return_value=r"C:\\Windows\\System32\\wt.exe") as which_mock, patch(
            "integrations.terminal.subprocess.Popen"
        ) as popen_mock:
            integration.run_command({"command": "git status", "cwd": "."})

            which_mock.assert_called_once_with("wt")
            popen_mock.assert_called_once()
            args = popen_mock.call_args[0][0]
            self.assertEqual(args[0], r"C:\\Windows\\System32\\wt.exe")
            self.assertIn("git status", args)
            self.assertIn("-d", args)

    def test_terminal_run_command_falls_back_to_powershell(self):
        integration = TerminalIntegration()

        with patch("integrations.terminal.shutil.which", return_value=None), patch(
            "integrations.terminal.subprocess.Popen"
        ) as popen_mock:
            integration.run_command({"command": "git status", "cwd": "."})

            popen_mock.assert_called_once()
            args = popen_mock.call_args[0][0]
            kwargs = popen_mock.call_args[1]
            self.assertEqual(args[:3], ["powershell", "-NoExit", "-Command"])
            self.assertEqual(args[3], "git status")
            self.assertTrue(os.path.isdir(kwargs["cwd"]))

    def test_windows_open_app_resolves_alias(self):
        integration = WindowsOSIntegration()

        with patch.object(integration, "_resolve_executable_path", return_value=r"C:\Tools\Code.exe"), patch(
            "integrations.windows_os.subprocess.Popen"
        ) as popen_mock:
            integration.open_app({"app_name": "vs code"})
            popen_mock.assert_called_once_with([r"C:\Tools\Code.exe"], shell=False)

    def test_windows_open_app_falls_back_to_cmd_start_when_exe_not_found(self):
        integration = WindowsOSIntegration()

        with patch.object(integration, "_resolve_executable_path", return_value=None), patch(
            "integrations.windows_os.subprocess.Popen"
        ) as popen_mock:
            integration.open_app({"app_name": "google chrome"})
            popen_mock.assert_called_once_with(["cmd", "/c", "start", "", "google chrome"], shell=False)

    def test_windows_close_app_uses_taskkill(self):
        integration = WindowsOSIntegration()

        with patch("integrations.windows_os.subprocess.run") as run_mock:
            integration.close_app({"app_name": "chrome"})
            run_mock.assert_called_once()
            args = run_mock.call_args[0][0]
            self.assertEqual(args[:3], ["taskkill", "/IM", "chrome.exe"])

    def test_intent_normalizes_terminal_folder_run_flow(self):
        engine = IntentEngine()
        parsed = {"action": "open_app", "params": {"app_name": "windows terminal"}, "confidence": 0.9}

        action = engine._normalize_action(parsed, "open a terminal in my projects folder and run git status")

        self.assertEqual(action.action, "run_command")
        self.assertEqual(action.params.get("command"), "git status")
        self.assertTrue(action.params.get("cwd", "").lower().endswith("desktop\\projects"))


if __name__ == "__main__":
    unittest.main()
