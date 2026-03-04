"""CLI capability for subprocess execution with exit-code verification."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from alara.capabilities.base import BaseCapability, CapabilityResult


class CLICapability(BaseCapability):
    """Execute command-line operations through subprocess orchestration."""

    def __init__(self) -> None:
        timeout_value = os.getenv("STEP_TIMEOUT_S", "30")
        try:
            self.default_timeout_s = int(timeout_value)
        except ValueError:
            self.default_timeout_s = 30

    def execute(self, operation: str, params: dict[str, Any]) -> CapabilityResult:
        logger.debug("CLI operation requested: {} | params={}", operation, params)
        if not self.supports(operation):
            return CapabilityResult.fail(f"Unsupported CLI operation: {operation}")
        if operation != "run_command":
            return CapabilityResult.fail(f"Unhandled CLI operation: {operation}")

        try:
            command = str(params.get("command", "")).strip()
            if not command:
                return CapabilityResult.fail("Missing required parameter: command")

            timeout_s = params.get("timeout_s", self.default_timeout_s)
            if timeout_s is None:
                timeout_s = self.default_timeout_s
            timeout_s = int(timeout_s)

            working_dir_raw = params.get("working_dir")
            resolved_working_dir = self._resolve_dir(working_dir_raw)
            if not resolved_working_dir.exists():
                return CapabilityResult.fail(
                    f"Working directory does not exist: {working_dir_raw or resolved_working_dir}"
                )

            logger.debug(
                "Executing command: {} | cwd={} | timeout_s={}",
                command,
                resolved_working_dir,
                timeout_s,
            )

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(resolved_working_dir),
                timeout=timeout_s,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout if stdout.strip() else (stderr if stderr.strip() else "(no output)")
            metadata = {
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }

            if result.returncode == 0:
                return CapabilityResult.ok(output=output, metadata=metadata)

            logger.warning(
                "Command returned non-zero exit code {}: {}",
                result.returncode,
                command,
            )
            combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
            return CapabilityResult.fail(
                error=combined or output,
                metadata=metadata,
            )
        except subprocess.TimeoutExpired:
            return CapabilityResult.fail(error=f"Command timed out after {timeout_s}s")
        except Exception as exc:
            logger.error("CLI execution exception: {}", exc)
            return CapabilityResult.fail(error=str(exc))

    def supports(self, operation: str) -> bool:
        return operation == "run_command"

    def _resolve_dir(self, path: str | None) -> Path:
        """Resolve directory path with proper environment variable substitution and anchoring."""
        try:
            # Step 1 — Handle None
            if path is None:
                return Path.home()

            # Step 2 — Substitute known environment variable patterns
            path_string = str(path)
            path_string = path_string.replace("$env:USERPROFILE", str(Path.home()))
            path_string = path_string.replace("%USERPROFILE%", str(Path.home()))
            path_string = path_string.replace("$env:HOME", str(Path.home()))
            path_string = path_string.replace("$HOME", str(Path.home()))
            
            # Handle ~ expansion (only if path starts with ~ or ~/)
            if path_string.startswith("~") or path_string.startswith("~/"):
                path_string = str(Path.home()) + path_string[1:] if path_string.startswith("~") else path_string[2:]

            # Step 3 — Expand any remaining ~ using pathlib
            result = Path(path_string).expanduser()

            # Step 4 — Anchor relative paths to home directory
            if result.is_absolute():
                return result
            else:
                return Path.home() / result

        except Exception as exc:
            logger.warning("Directory resolution failed for '{}': {}", path, exc)
            return Path.home()
