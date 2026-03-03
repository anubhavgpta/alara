"""Integration planner test runner.

Run with:
    python -m tests.test_planner
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure `alara` package imports work whether the command is run from
# `.../Alara` or `.../Alara/alara`.
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parents[1]   # .../Alara/alara
_PROJECT_ROOT = _THIS_FILE.parents[2]   # .../Alara
for _candidate in (_PROJECT_ROOT, _PACKAGE_ROOT):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from alara.core.goal_understander import GoalUnderstander
from alara.core.planner import Planner, PlanningError
from alara.schemas.task_graph import ExecutionLayer, StepType, TaskGraph


GOALS = [
    "Create a new Python project called myapp with a virtual environment",
    "Find all PDF files on my desktop and move them to a folder called Documents/PDFs",
    "Delete all .tmp and .log files in my Downloads folder",
    "Set up a FastAPI project with a Postgres database called myapp_db",
    "Create a folder structure for a new React project called dashboard",
    "Find the largest 10 files in my Documents folder and list them",
    "Rename all images in my Downloads folder to include today's date",
    "Install git if not already installed and configure my username as Anubhav",
]

ALLOWED_VERIFICATION_METHODS = {
    "check_path_exists",
    "check_exit_code_zero",
    "check_process_running",
    "check_file_contains",
    "check_directory_not_empty",
    "check_port_open",
    "check_output_contains",
    "none",
}

PATH_PARAM_KEYS = {"path", "source", "destination", "target_path", "working_dir", "cwd"}


def _find_cycle(task_graph: TaskGraph) -> list[int] | None:
    adjacency = {step.id: step.depends_on for step in task_graph.steps}
    visited: set[int] = set()
    stack: list[int] = []
    in_stack: set[int] = set()

    def dfs(node: int) -> list[int] | None:
        visited.add(node)
        in_stack.add(node)
        stack.append(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
            elif neighbor in in_stack:
                start = stack.index(neighbor)
                return stack[start:] + [neighbor]

        stack.pop()
        in_stack.remove(node)
        return None

    for node in adjacency:
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                return cycle
    return None


def _assert_plan(goal: str, task_graph: TaskGraph, goal_index: int) -> list[str]:
    failures: list[str] = []
    step_ids = {step.id for step in task_graph.steps}

    if not task_graph.steps:
        failures.append("task_graph.steps is empty")

    for step in task_graph.steps:
        if not str(step.description).strip():
            failures.append(f"step {step.id}: description is empty")
        if not str(step.operation).strip():
            failures.append(f"step {step.id}: operation is empty")
        if not str(step.expected_outcome).strip():
            failures.append(f"step {step.id}: expected_outcome is empty")
        if not isinstance(step.step_type, StepType):
            failures.append(f"step {step.id}: invalid step_type value {step.step_type!r}")
        if not isinstance(step.preferred_layer, ExecutionLayer):
            failures.append(
                f"step {step.id}: invalid preferred_layer value {step.preferred_layer!r}"
            )
        if not str(step.verification_method).strip():
            failures.append(f"step {step.id}: verification_method is empty")
        if step.verification_method not in ALLOWED_VERIFICATION_METHODS:
            failures.append(
                f"step {step.id}: invalid verification_method {step.verification_method!r}"
            )
        for dep in step.depends_on:
            if dep not in step_ids:
                failures.append(f"step {step.id}: depends_on references unknown id {dep}")
            if dep >= step.id:
                failures.append(
                    f"step {step.id}: depends_on includes non-lower id {dep} (forward-only dependency violated)"
                )

        params = step.params if isinstance(step.params, dict) else {}
        for key in PATH_PARAM_KEYS:
            value = params.get(key)
            if isinstance(value, str) and "\\" in value:
                failures.append(f"step {step.id}: path param '{key}' contains backslashes")

    cycle = _find_cycle(task_graph)
    if cycle:
        failures.append(f"circular dependency detected in task graph: {cycle}")

    try:
        datetime.fromisoformat(task_graph.created_at)
    except Exception as exc:
        failures.append(f"created_at is not valid ISO datetime: {exc}")

    if task_graph.goal != task_graph.goal_context.goal:
        failures.append(
            f"task_graph.goal mismatch: task_graph.goal={task_graph.goal!r} goal_context.goal={task_graph.goal_context.goal!r}"
        )

    status = "FAILED" if failures else "PASSED"
    print(f"\nGoal {goal_index}: {goal}")
    print(f"Status: {status}")
    print("Steps:")
    for step in task_graph.steps:
        print(f"  {step.id}. [{step.step_type.value}] {step.operation} - {step.description}")
        print(f"     Verify: {step.verification_method}")
        print(f"     Deps: {step.depends_on}")
    print(
        "GoalContext: scope={} complexity={}".format(
            task_graph.goal_context.scope,
            task_graph.goal_context.estimated_complexity,
        )
    )

    return failures


def main() -> int:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is required to run tests. Add it to your .env file.")
        return 1

    understander = GoalUnderstander()
    try:
        planner = Planner()
    except EnvironmentError as exc:
        print(str(exc))
        return 1

    total_passed = 0
    failed_details: list[tuple[int, str, list[str]]] = []

    for index, goal in enumerate(GOALS, start=1):
        try:
            goal_context = understander.understand(goal)
            task_graph = planner.plan(goal_context)
            failures = _assert_plan(goal, task_graph, index)
        except PlanningError as exc:
            failures = [f"planning failed: {exc}"]
            print(f"\nGoal {index}: {goal}")
            print("Status: FAILED")
            print(f"Failure: {exc}")
        except Exception as exc:
            failures = [f"unexpected error: {exc}"]
            print(f"\nGoal {index}: {goal}")
            print("Status: FAILED")
            print(f"Failure: {exc}")

        if failures:
            print("Failures:")
            for failure in failures:
                print(f"  - {failure}")
            failed_details.append((index, goal, failures))
        else:
            total_passed += 1

    total_failed = len(GOALS) - total_passed
    print("\nFinal summary:")
    print(f"PASSED: {total_passed}/8")
    print(f"FAILED: {total_failed}/8")

    if failed_details:
        print("Failed goals:")
        for index, goal, failures in failed_details:
            print(f"  Goal {index}: {goal}")
            for failure in failures:
                print(f"    - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
