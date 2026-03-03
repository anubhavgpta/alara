"""Task graph schema definitions for planner and orchestrator coordination."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from alara.schemas.goal import GoalContext


class StepType(str, Enum):
    """Supported execution step categories."""

    FILESYSTEM = "filesystem"
    CLI = "cli"
    APP_ADAPTER = "app_adapter"
    UI_AUTOMATION = "ui_automation"
    VISION = "vision"
    SYSTEM = "system"


class ExecutionLayer(str, Enum):
    """Execution layer preference for a step."""

    OS_API = "os_api"
    APP_ADAPTER = "app_adapter"
    CLI = "cli"
    UI_AUTOMATION = "ui_automation"
    VISION = "vision"


class StepStatus(str, Enum):
    """Lifecycle status of a step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class Step(BaseModel):
    """Single executable step in a task graph."""

    id: int
    description: str
    step_type: StepType
    preferred_layer: ExecutionLayer
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str
    verification_method: str
    depends_on: list[int] = Field(default_factory=list)
    fallback_strategy: str | None = None
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None

    @field_validator(
        "description",
        "operation",
        "expected_outcome",
        "verification_method",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value.strip()


class StepResult(BaseModel):
    """Execution result for one step."""

    step_id: int
    success: bool
    output: str | None = None
    error: str | None = None
    verified: bool
    execution_layer_used: ExecutionLayer
    duration_ms: float
    attempts: int


class TaskGraph(BaseModel):
    """Planner-produced graph of steps for orchestrator execution."""

    goal: str
    goal_context: GoalContext
    steps: list[Step]
    created_at: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    results: list[StepResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_steps(self) -> TaskGraph:
        if not self.steps:
            raise ValueError("TaskGraph must contain at least one step")

        counts = Counter(step.id for step in self.steps)
        has_duplicates = any(count > 1 for count in counts.values())
        if has_duplicates:
            id_to_first_new: dict[int, int] = {}
            for index, step in enumerate(self.steps, start=1):
                if step.id not in id_to_first_new:
                    id_to_first_new[step.id] = index
                step.id = index

            for step in self.steps:
                remapped: list[int] = []
                for dep_id in step.depends_on:
                    if dep_id in id_to_first_new:
                        remapped.append(id_to_first_new[dep_id])
                    else:
                        remapped.append(dep_id)
                step.depends_on = remapped

            logger.warning(
                "Duplicate step IDs detected and renumbered sequentially from 1."
            )

        valid_ids = {step.id for step in self.steps}
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in valid_ids:
                    raise ValueError(
                        f"Step {step.id} depends on non-existent step ID {dep_id}"
                    )

        adjacency = {step.id: step.depends_on for step in self.steps}
        visited: set[int] = set()
        in_stack: set[int] = set()
        path: list[int] = []

        def dfs(node: int) -> None:
            visited.add(node)
            in_stack.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in in_stack:
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    formatted = " -> ".join(str(item) for item in cycle_path)
                    raise ValueError(f"Circular dependency detected: {formatted}")

            in_stack.remove(node)
            path.pop()

        for step in self.steps:
            if step.id not in visited:
                dfs(step.id)

        return self

    def get_step(self, step_id: int) -> Step | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def next_pending_step(self) -> Step | None:
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            dependencies_done = all(
                (self.get_step(dep_id) is not None)
                and (self.get_step(dep_id).status == StepStatus.DONE)
                for dep_id in step.depends_on
            )
            if dependencies_done:
                return step
        return None

    def is_complete(self) -> bool:
        return all(
            step.status in {StepStatus.DONE, StepStatus.SKIPPED} for step in self.steps
        )

    def is_failed(self) -> bool:
        return any(
            step.status == StepStatus.FAILED
            and step.attempts >= 3
            and step.fallback_strategy is None
            for step in self.steps
        )

    def summary(self) -> dict:
        return {
            "total_steps": len(self.steps),
            "completed": sum(1 for step in self.steps if step.status == StepStatus.DONE),
            "failed": sum(1 for step in self.steps if step.status == StepStatus.FAILED),
            "pending": sum(
                1 for step in self.steps if step.status == StepStatus.PENDING
            ),
            "skipped": sum(
                1 for step in self.steps if step.status == StepStatus.SKIPPED
            ),
            "retrying": sum(
                1 for step in self.steps if step.status == StepStatus.RETRYING
            ),
            "overall_status": self.status,
        }


__all__ = [
    "StepType",
    "ExecutionLayer",
    "StepStatus",
    "Step",
    "StepResult",
    "TaskGraph",
]
