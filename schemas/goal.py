"""Goal schema definitions for normalized user intent and constraints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GoalContext(BaseModel):
    """Structured representation of a user's high-level goal."""

    raw_input: str
    goal: str
    scope: Literal["filesystem", "cli", "app", "system", "mixed"]
    constraints: list[str] = Field(default_factory=list)
    working_directory: str | None = None
    estimated_complexity: Literal["simple", "moderate", "complex"]

    @classmethod
    def from_raw(cls, raw_input: str) -> GoalContext:
        return cls(
            raw_input=raw_input,
            goal=raw_input.strip(),
            scope="mixed",
            constraints=[],
            working_directory=None,
            estimated_complexity="moderate",
        )


__all__ = ["GoalContext"]
