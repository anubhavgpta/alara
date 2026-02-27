"""
alara/integrations/terminal.py

Terminal integration for Week 5-6 scope.
Executes commands in Windows Terminal when available, with PowerShell fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger


class TerminalIntegration:
    """Run shell commands from structured action params."""

    def _resolve_cwd(self, params: dict) -> str | None:
        raw = str(params.get("cwd") or params.get("path") or params.get("directory") or "").strip()
        if not raw:
            return None
        expanded = os.path.expandvars(os.path.expanduser(raw))
        path = Path(expanded).resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Terminal working directory not found: {path}")
        return str(path)

    def run_command(self, params: dict):
        command = str(params.get("command", "")).strip()
        if not command:
            raise ValueError("Missing terminal command")

        cwd = self._resolve_cwd(params)
        logger.info(f"Running terminal command: {command} (cwd={cwd or 'default'})")

        wt_path = shutil.which("wt")
        if wt_path:
            wt_cmd = [wt_path]
            if cwd:
                wt_cmd.extend(["-d", cwd])
            wt_cmd.extend(["powershell", "-NoExit", "-Command", command])
            subprocess.Popen(wt_cmd, shell=False)
            return

        # Fallback for systems without Windows Terminal.
        subprocess.Popen(
            ["powershell", "-NoExit", "-Command", command],
            cwd=cwd or None,
            shell=False,
        )
