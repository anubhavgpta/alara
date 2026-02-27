"""
alara/integrations/windows_os.py

Windows integration for Week 5-6 scope:
- app open/close/switch
- window controls
- basic file operations
- core system controls
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger


class WindowsOSIntegration:
    """Execute Windows OS actions."""

    APP_MAP = {
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "edge": "msedge.exe",
        "firefox": "firefox.exe",
        "vscode": "Code.exe",
        "vs code": "Code.exe",
        "visual studio code": "Code.exe",
        "windows terminal": "wt.exe",
        "terminal": "wt.exe",
        "powershell": "powershell.exe",
        "notepad": "notepad.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "spotify": "Spotify.exe",
        "slack": "slack.exe",
    }

    SW_MINIMIZE = 6
    SW_MAXIMIZE = 3
    WM_CLOSE = 0x0010
    KEYEVENTF_KEYUP = 0x0002
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF
    USER_FOLDER_ALIASES = {
        "desktop": "Desktop",
        "downloads": "Downloads",
        "documents": "Documents",
        "pictures": "Pictures",
        "music": "Music",
        "videos": "Videos",
        "home": "",
    }
    COMMON_EXE_PATHS = {
        "chrome.exe": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            str(Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe"),
        ],
        "msedge.exe": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "Code.exe": [
            str(Path.home() / "AppData/Local/Programs/Microsoft VS Code/Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
            r"C:\Program Files (x86)\Microsoft VS Code\Code.exe",
        ],
        "wt.exe": [
            str(Path.home() / "AppData/Local/Microsoft/WindowsApps/wt.exe"),
        ],
    }

    def _resolve_app(self, app_name: str) -> str:
        name = app_name.strip().lower()
        if not name:
            raise ValueError("Missing app_name")
        if name in self.APP_MAP:
            return self.APP_MAP[name]
        if name.endswith(".exe"):
            return name
        return f"{name}.exe"

    def _active_window(self) -> int:
        return int(ctypes.windll.user32.GetForegroundWindow())

    def _show_window(self, command: int) -> None:
        hwnd = self._active_window()
        if not hwnd:
            raise RuntimeError("No active window found")
        ctypes.windll.user32.ShowWindow(hwnd, command)

    def _press_media_key(self, key_code: int, repeats: int) -> None:
        count = max(1, repeats)
        for _ in range(count):
            ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key_code, 0, self.KEYEVENTF_KEYUP, 0)

    def _expand_path(self, raw_path: str) -> Path:
        raw = raw_path.strip()
        if not raw:
            raise ValueError("Path is empty")

        normalized = raw.replace("\\", "/").strip("/")
        parts = [p for p in normalized.split("/") if p]
        alias_key = parts[0].lower() if parts else ""
        alias_key = alias_key.replace("my ", "").replace(" folder", "").strip()

        home = Path.home()
        if alias_key in self.USER_FOLDER_ALIASES:
            suffix = self.USER_FOLDER_ALIASES[alias_key]
            base = home / suffix if suffix else home
            if len(parts) > 1:
                return (base.joinpath(*parts[1:])).resolve()
            return base.resolve()

        expanded = os.path.expandvars(os.path.expanduser(raw))
        candidate = Path(expanded)
        if candidate.is_absolute():
            return candidate.resolve()

        # Try from current shell CWD first, then repository root, then home.
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path.cwd() / candidate,
            repo_root / candidate,
            home / candidate,
        ]
        for path in candidates:
            if path.exists():
                return path.resolve()
        return (Path.cwd() / candidate).resolve()

    def _resolve_executable_path(self, exe_name: str) -> str | None:
        exe = exe_name.strip()
        if not exe:
            return None
        if Path(exe).is_absolute() and Path(exe).exists():
            return exe
        which_path = shutil.which(exe)
        if which_path:
            return which_path
        for candidate in self.COMMON_EXE_PATHS.get(exe, []):
            if Path(candidate).exists():
                return str(Path(candidate))
        return None

    def open_app(self, params: dict):
        app_name = str(params.get("app_name", "")).strip()
        exe = self._resolve_app(app_name)
        resolved = self._resolve_executable_path(exe)
        logger.info(f"Opening app: {app_name} -> {resolved or exe}")
        if resolved:
            subprocess.Popen([resolved], shell=False)
            return
        # Shell fallback: still allows apps registered via App Paths or aliases.
        start_target = app_name or exe
        subprocess.Popen(["cmd", "/c", "start", "", start_target], shell=False)

    def close_app(self, params: dict):
        app_name = str(params.get("app_name", "")).strip()
        exe = self._resolve_app(app_name)
        logger.info(f"Closing app: {app_name} -> {exe}")
        subprocess.run(["taskkill", "/IM", exe, "/F"], capture_output=True, text=True, check=False)

    def switch_app(self, params: dict):
        app_name = str(params.get("app_name", "")).strip()
        if not app_name:
            raise ValueError("Missing app_name")
        script = (
            "$ws = New-Object -ComObject WScript.Shell; "
            f"$null = $ws.AppActivate('{app_name}')"
        )
        logger.info(f"Switching to app: {app_name}")
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False, capture_output=True, text=True)

    def minimize_window(self, params: dict):
        logger.info("Minimizing active window")
        self._show_window(self.SW_MINIMIZE)

    def maximize_window(self, params: dict):
        logger.info("Maximizing active window")
        self._show_window(self.SW_MAXIMIZE)

    def close_window(self, params: dict):
        logger.info("Closing active window")
        hwnd = self._active_window()
        if not hwnd:
            raise RuntimeError("No active window found")
        ctypes.windll.user32.PostMessageW(hwnd, self.WM_CLOSE, 0, 0)

    def take_screenshot(self, params: dict):
        target_dir = self._expand_path(str(params.get("directory", "~/Pictures/Alara/Screenshots")))
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = params.get("filename", "")
        if not filename:
            from datetime import datetime

            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        target_file = target_dir / filename
        script = (
            "Add-Type -AssemblyName System.Drawing; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$bounds=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bmp=New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; "
            "$g=[System.Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($bounds.Location,[System.Drawing.Point]::Empty,$bounds.Size); "
            f"$bmp.Save('{target_file}',[System.Drawing.Imaging.ImageFormat]::Png); "
            "$g.Dispose(); $bmp.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
        logger.info(f"Saved screenshot to: {target_file}")

    def open_file(self, params: dict):
        raw = str(params.get("path", "")).strip()
        if not raw:
            raise ValueError("Missing file path")
        path = self._expand_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        logger.info(f"Opening file: {path}")
        os.startfile(str(path))

    def open_folder(self, params: dict):
        raw = str(params.get("path", "")).strip()
        if not raw:
            raise ValueError("Missing folder path")
        path = self._expand_path(raw)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Folder not found: {path}")
        logger.info(f"Opening folder: {path}")
        subprocess.Popen(["explorer.exe", str(path)], shell=False)

    def search_files(self, params: dict):
        query = str(params.get("query", "*")).strip() or "*"
        location = str(params.get("location", "~")).strip() or "~"
        location_path = self._expand_path(location)
        if not location_path.exists():
            raise FileNotFoundError(f"Search location not found: {location_path}")
        matches = [str(path) for path in location_path.rglob(query)]
        limit = int(params.get("limit", 20))
        limited = matches[: max(1, limit)]
        logger.info(f"Found {len(matches)} file(s) for query='{query}' in {location_path}")
        for file_path in limited:
            logger.info(f"  {file_path}")
        if not matches:
            logger.warning("No files matched search query")

    def volume_up(self, params: dict):
        amount = int(params.get("amount", 10))
        steps = max(1, amount // 2)
        logger.info(f"Volume up by ~{amount}%")
        self._press_media_key(self.VK_VOLUME_UP, steps)

    def volume_down(self, params: dict):
        amount = int(params.get("amount", 10))
        steps = max(1, amount // 2)
        logger.info(f"Volume down by ~{amount}%")
        self._press_media_key(self.VK_VOLUME_DOWN, steps)

    def volume_mute(self, params: dict):
        logger.info("Toggling mute")
        self._press_media_key(self.VK_VOLUME_MUTE, 1)

    def lock_screen(self, params: dict):
        logger.info("Locking screen")
        ctypes.windll.user32.LockWorkStation()
