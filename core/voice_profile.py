"""Per-user correction profile for transcript personalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from loguru import logger


class VoiceProfile:
    """Stores and applies user-specific transcription corrections."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id.strip() or "default"
        self.profile_dir = Path.home() / ".alara" / "profiles"
        self.profile_path = self.profile_dir / f"{self.user_id}.json"
        self.corrections: Dict[str, Dict[str, object]] = {}
        self._load()

    def apply(self, text: str) -> str:
        """Apply known corrections to text before intent parsing."""
        if not text:
            return text

        corrected = text
        entries = sorted(self.corrections.items(), key=lambda kv: len(kv[0]), reverse=True)
        for misheard, payload in entries:
            replacement = str(payload.get("corrected", "")).strip()
            if not replacement:
                continue
            pattern = re.compile(rf"\b{re.escape(misheard)}\b", flags=re.IGNORECASE)
            corrected = pattern.sub(replacement, corrected)
        return corrected

    def record_correction(self, misheard: str, corrected: str) -> None:
        """Add or update a correction mapping and persist it."""
        src = (misheard or "").strip().lower()
        dst = (corrected or "").strip()
        if not src or not dst:
            return

        payload = self.corrections.get(src, {"corrected": dst, "count": 0})
        payload["corrected"] = dst
        payload["count"] = int(payload.get("count", 0)) + 1
        self.corrections[src] = payload
        self._save()

    def most_common_failures(self, n: int = 10) -> List[dict]:
        """Return top misheard entries by frequency."""
        items = sorted(
            self.corrections.items(),
            key=lambda kv: int(kv[1].get("count", 0)),
            reverse=True,
        )
        top = items[: max(1, n)]
        return [
            {
                "misheard": misheard,
                "corrected": str(payload.get("corrected", "")),
                "count": int(payload.get("count", 0)),
            }
            for misheard, payload in top
        ]

    def _load(self) -> None:
        try:
            if not self.profile_path.exists():
                return
            raw = json.loads(self.profile_path.read_text(encoding="utf-8"))
            data = raw.get("corrections", {})
            if isinstance(data, dict):
                self.corrections = data
            logger.debug(
                f"Loaded voice profile '{self.user_id}' with {len(self.corrections)} corrections"
            )
        except Exception as exc:
            logger.warning(f"Failed to load voice profile '{self.user_id}': {exc}")

    def _save(self) -> None:
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            payload = {"user_id": self.user_id, "corrections": self.corrections}
            self.profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to save voice profile '{self.user_id}': {exc}")

