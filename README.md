<p align="center">
  <img src="./alara-banner.jpg" alt="ALARA Banner" width="100%" />
</p>

# ALARA
**Ambient Language & Reasoning Assistant**

ALARA is an agentic desktop AI platform for Windows that transforms natural language goals into executable tasks on a real machine. It implements a complete autonomous loop: goal understanding, structured planning, capability-routed execution, programmatic verification, and LLM-powered adaptive error recovery — all backed by a persistent three-tier memory layer and exposed through a floating Electron overlay.

**Version:** 0.2.0 &nbsp;|&nbsp; **Platform:** Windows 10/11 &nbsp;|&nbsp; **Python:** 3.11+ &nbsp;|&nbsp; **Model:** Gemini 2.5 Flash

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Pipeline](#core-pipeline)
- [Memory Layer](#memory-layer)
- [Capability Layer](#capability-layer)
- [Overlay UI & WebSocket Server](#overlay-ui--websocket-server)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Logging Standards](#logging-standards)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

ALARA is not a voice assistant, macro recorder, or conversational chatbot. It is an autonomous planning and execution engine for desktop tasks. A user provides a goal in natural language — via the overlay UI or terminal — and ALARA decomposes it into a structured `TaskGraph` of typed, ordered, verifiable steps, executes each step against real system state, validates outcomes programmatically, and recovers from failures using LLM-guided reflection.

The system is designed around three principles:

**Verification-first execution.** Every step has a declared verification method. Execution is not considered successful until the verifier confirms real-world state matches the expected outcome. ALARA does not trust exit codes alone.

**Adaptive recovery.** When verification fails, the Reflector sends full execution context — original goal, complete plan, prior results, failure details — to Gemini and receives corrected actions or alternative paths. Recovery is not hardcoded; it is reasoned.

**Persistent memory.** ALARA learns from every execution. Path aliases, tool preferences, and successful task patterns are stored in a SQLite-backed memory layer and injected into every subsequent planning invocation, making the system progressively more accurate over time.

---

## Architecture

### System Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                     Electron Overlay                         │
│              ws://localhost:8765 (WebSocket)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               WebSocket Server (async)                       │
│         server/websocket_server.py                           │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  Goal Understander    Planner      Memory Manager
  (GoalContext)     (TaskGraph)   (context injection)
         │               │               │
         └───────────────▼───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │        Orchestrator           │
         │   ┌─────────────────────┐     │
         │   │  Execution Router   │     │
         │   │  └► Filesystem      │     │
         │   │  └► CLI             │     │
         │   │  └► System          │     │
         │   │  └► App Adapter     │     │
         │   └──────────┬──────────┘     │
         │              ▼                │
         │          Verifier             │
         │       (real-world check)      │
         │              │                │
         │    ┌─────────▼──────────┐     │
         │    │     Reflector      │     │
         │    │  (on failure only) │     │
         │    └────────────────────┘     │
         └───────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │         Memory Layer          │
         │  Session │ Preferences │ Skills│
         │       SQLite + WAL mode        │
         └───────────────────────────────┘
```

---

## Core Pipeline

### Goal Understander

`core/goal_understander.py`

Parses raw natural language input into a structured `GoalContext` using Gemini 2.5 Flash. Extracts normalized goal intent, operational scope (`filesystem`, `cli`, `mixed`, `system`), explicit constraints, inferred working directory, and estimated complexity (`simple`, `moderate`, `complex`). Implements a guaranteed `from_raw` fallback — the understander never raises, ensuring the pipeline always receives a valid context object.

### Planner

`core/planner.py`

Receives a `GoalContext` and a `MemoryContext` from the memory layer and sends both to Gemini 2.5 Flash via a constrained planning prompt. The prompt enforces atomic step decomposition, absolute path generation, Windows/PowerShell-compatible commands, and a fixed set of whitelisted operations and verification methods. Returns a fully validated `TaskGraph`.

Platform context is injected at planning time — the planner provides Gemini with the machine's absolute paths for all common user directories (`Desktop`, `Documents`, `Downloads`, `Pictures`, `Videos`, `Music`, `AppData`) so all generated paths are fully qualified and never relative to the process working directory.

The planner includes:
- Single JSON retry on malformed responses with a stricter prompt
- Enum casing normalization (`step_type` and `preferred_layer` forced to lowercase before Pydantic validation)
- Path normalization across all path-bearing fields using `pathlib.Path.as_posix()`
- `last_raw_response` property for debug inspection

### TaskGraph Validation

`schemas/task_graph.py`

`TaskGraph` performs structural validation on construction:
1. Rejects empty step lists
2. Detects and renumbers duplicate step IDs, remapping all `depends_on` references
3. Validates all `depends_on` targets exist post-normalization
4. Runs DFS cycle detection, raising `ValueError` with the full cycle path if found

`next_pending_step()` returns the first `PENDING` step whose dependencies are all `DONE`. `SKIPPED`, `FAILED`, and `RUNNING` statuses do not satisfy dependency requirements.

### Orchestrator

`core/orchestrator.py`

Executes a `TaskGraph` through the full orchestration loop with a maximum of 3 retries per step:

```
get next pending step
        ↓
route to capability layer
        ↓
execute step → increment attempts
        ↓
verify real-world outcome
        ↓ (on failure)
retry if attempts < MAX_RETRIES
        ↓ (on max retries)
invoke Reflector → apply modified step / skip / escalate
```

The Orchestrator accepts an optional `progress_callback` invoked after each step state change, used by the WebSocket server to stream real-time progress to the Electron overlay.

### Execution Router

`core/execution_router.py`

Routes each `Step` to the correct capability based on `step_type` and `preferred_layer` in strict priority order: `FilesystemCapability` → `CLICapability` → `SystemCapability` → app adapter fallback → UI automation fallback. Unimplemented layers log a `WARNING` and fall back to CLI where possible. Vision capability returns an explicit `CapabilityResult.fail` with a clear message.

### Verifier

`core/verifier.py`

Validates real-world state after every step execution. Verification is not optional — every step declares a `verification_method` and the Verifier executes the appropriate check:

| Method | What is checked |
|---|---|
| `check_path_exists` | File or directory exists on disk |
| `check_file_contains` | File content includes expected text |
| `check_exit_code_zero` | Command returned exit code 0 |
| `check_process_running` | Named process is active in tasklist |
| `check_port_open` | TCP port accepts connections |
| `check_output_contains` | Command output includes expected content |
| `check_directory_not_empty` | Directory exists and has at least one entry |
| `none` | No verification required — always passes |

Unknown verification methods log a `WARNING` and pass rather than raising, preserving pipeline stability.

### Reflector

`core/reflector.py`

Invoked when a step exhausts its retry budget. Sends full execution context to Gemini 2.5 Flash at `temperature=0.3` and receives one of three actions:

- **retry** — Gemini provides a modified step with corrected operation, parameters, or approach. The Orchestrator applies the modification, resets `attempts` to 0, and retries.
- **skip** — Step is marked `SKIPPED` and execution continues with dependent steps unblocked where possible.
- **escalate** — Step is marked `FAILED`, the failure reason is recorded, and the Orchestrator terminates the task.

The Reflector never raises. On any API or parse failure it returns `action="escalate"` with the error message as the reason.

---

## Memory Layer

`memory/`

A production-ready, thread-safe, SQLite-backed memory system with three tiers. All access is through the `MemoryManager` singleton. The database runs in WAL mode for concurrent read access. Schema versioning is implemented from day one with a migration hook for future schema evolution.

### Session Memory

`memory/session.py`

Tracks every goal execution within and across sessions. Each `SessionEntry` records the original goal, scope, final status (`success`, `partial`, `failed`), step counts, full execution log, and UTC timestamps. Queryable by recency, session ID, and goal text search.

### Preference Memory

`memory/preferences.py`

Persistent key-value store for user preferences, tool choices, and path aliases. Preferences carry confidence scores, usage counts, and provenance (`user_explicit`, `inferred`, `default`). Seeded with platform defaults on first run.

**Automatic inference:** After every successful execution, `infer_from_execution()` extracts path aliases from step parameters (mapping noun phrases in the goal to absolute paths used), tool preferences from CLI commands, and package patterns from `pip install` invocations. Inference is wrapped in `try/except` and never affects the execution flow.

**Path aliases** are the most impactful feature — once ALARA learns that "my projects folder" maps to `C:/Users/Anubhav Gupta/Desktop/Projects`, that mapping is injected into every subsequent planning invocation via the memory context summary.

### Skill Memory

`memory/skills.py`

Stores successful `TaskGraph` executions as reusable templates. Skills are retrieved using word-overlap similarity search with a composite ranking score:

```
score = (overlap × 0.6) + (success_rate × 0.3) + (recency × 0.1)
```

Where `success_rate = success_count / (success_count + failure_count + 1)` and `recency = 1.0` if used within 7 days, `0.5` within 30 days, `0.0` otherwise. Skills with similarity above 0.8 are deduplicated — repeated similar goals update the existing skill's statistics rather than creating a new entry.

### Memory Context Injection

Before every planning invocation, `MemoryManager.build_context()` assembles a `MemoryContext` containing recent goals, relevant skills, relevant preferences, and all known path aliases. This context is serialized as a structured summary string and appended to the Gemini planning prompt, giving the planner awareness of prior work without requiring conversational state.

### Database

`memory/database.py`

Thread-safe singleton `DatabaseManager` with per-call connections (not shared), WAL journal mode, foreign key enforcement, and retry logic on `OperationalError: database is locked` (3 attempts with 100ms backoff). Indexed on `sessions.created_at`, `sessions.session_id`, `skills.scope`, `preferences.category`, and `preferences.key` for sub-100ms `build_context()` performance.

---

## Capability Layer

All capabilities inherit from `BaseCapability` and implement a single entry point:

```python
class BaseCapability(ABC):
    @abstractmethod
    def execute(self, operation: str, params: dict) -> CapabilityResult:
        ...

    def supports(self, operation: str) -> bool:
        return False
```

All operations return `CapabilityResult` and never raise — exceptions are caught internally and returned as `CapabilityResult.fail(error=str(e))`.

### FilesystemCapability

`capabilities/filesystem.py`

Uses `pathlib.Path` exclusively. Supports: `create_directory`, `create_file`, `write_file`, `read_file`, `delete_file`, `delete_directory`, `move_file`, `copy_file`, `list_directory`, `search_files`, `check_path_exists`.

Path resolution via `_resolve(path: str) -> Path` handles all Windows path variants in order:
1. Substitute `$env:USERPROFILE`, `%USERPROFILE%`, `$env:HOME`, `$HOME`
2. Expand `~` via `Path.expanduser()`
3. Anchor remaining relative paths to `Path.home()` — nothing resolves relative to the process working directory

### CLICapability

`capabilities/cli.py`

Executes shell commands via `subprocess.run(shell=True, capture_output=True, text=True)`. Captures stdout, stderr, and returncode. Working directory resolved via `_resolve_dir()` using the same path resolution logic as `FilesystemCapability`. Default timeout read from `STEP_TIMEOUT_S` environment variable (default 30s). Both stdout and stderr included in result metadata for Reflector context.

### SystemCapability

`capabilities/system.py`

Handles `get_env_var` (with Windows-specific fallback chain: `USERPROFILE` → `HOMEDRIVE+HOMEPATH` when `HOME` is requested), `set_env_var`, and `check_process` via `tasklist /FI`.

---

## Overlay UI & WebSocket Server

### WebSocket Server

`server/websocket_server.py`

Async WebSocket server on `ws://localhost:8765` using the `websockets` library. All blocking pipeline operations (understand, plan, orchestrate) run in `asyncio.run_in_executor` to avoid blocking the event loop. The `progress_callback` dispatches step progress messages to the connected client thread-safely via `asyncio.run_coroutine_threadsafe`.

#### Message Protocol

**Client → Server:**

| Message | Description |
|---|---|
| `{ "type": "goal_submit", "goal": "..." }` | Submit a new goal for planning |
| `{ "type": "goal_confirm" }` | Confirm and execute the current plan |
| `{ "type": "goal_cancel" }` | Cancel the current plan |
| `{ "type": "ping" }` | Keepalive |

**Server → Client:**

| Message | Description |
|---|---|
| `{ "type": "status", "message": "..." }` | General status update |
| `{ "type": "plan_ready", "goal", "steps", "scope", "complexity", "step_count" }` | Plan ready for confirmation |
| `{ "type": "execution_started", "total_steps" }` | Orchestration has begun |
| `{ "type": "step_progress", "step_id", "operation", "description", "status", "steps_done", "steps_total", "progress_pct" }` | Per-step progress update |
| `{ "type": "execution_complete", "success", "steps_completed", "steps_failed", "steps_skipped", "total_steps", "message" }` | Task complete |
| `{ "type": "error", "message" }` | Pipeline error |
| `{ "type": "pong" }` | Keepalive response |

### Electron Overlay

`electron/`

A frameless, always-on-top transparent overlay window built with Electron. Triggered globally with `Ctrl+Space` (fallback: `Ctrl+Shift+Space`). WebSocket connection is managed from the main process with automatic reconnection on disconnect (2s backoff).

**Interaction flow:**
1. `Ctrl+Space` → overlay appears, input focused
2. User types goal → presses Enter → `goal_submit` sent
3. Server responds with `plan_ready` → step list rendered, window resizes to fit
4. User clicks Execute → `goal_confirm` sent
5. Determinate progress bar fills as `step_progress` messages arrive
6. `execution_complete` → result shown → auto-dismiss after 3s → overlay hides

**UI states:**

| State | Indicator | Description |
|---|---|---|
| Ready | White | Waiting for input |
| Planning | Purple (#9B59FF) | Decomposing goal into steps |
| Awaiting confirm | Purple | Plan rendered, awaiting user confirmation |
| Executing | Blue (#3B9EFF) | Running steps with progress bar |
| Done | Green (#2ECC71) | Task completed successfully |
| Failed | Red (#E74C3C) | Task failed after maximum retries |

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- Node.js 18 or later (for Electron overlay)
- Gemini API key from [Google AI Studio](https://aistudio.google.com)

### Installation

```powershell
git clone https://github.com/your-username/alara.git
cd alara
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install Electron dependencies:

```powershell
cd electron
npm install
cd ..
```

### Configuration

```powershell
copy .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_key_here
MAX_RETRIES=3
STEP_TIMEOUT_S=30
DEBUG=false
LOG_FILE=alara.log
DB_PATH=alara.db
```

---

## Usage

### Terminal Mode

```powershell
# Interactive prompt loop
python -m alara.main

# Single goal, auto-confirm
python -m alara.main --goal "create a Python project called myapp"

# Debug mode — prints GoalContext, raw Gemini response,
# execution log, memory context, and memory health
python -m alara.main --debug
```

### Overlay + WebSocket Mode

Start the backend server and the Electron overlay in separate terminals:

```powershell
# Terminal 1
start_server.bat

# Terminal 2
start_ui.bat
```

Or manually:

```powershell
# Terminal 1
.venv\Scripts\activate
python -m alara.server.websocket_server

# Terminal 2
cd electron
npx electron .
```

Press `Ctrl+Space` to toggle the overlay.

### Example Goals

| Goal | What ALARA Does |
|---|---|
| `create a Python project called myapp with a venv` | Creates directory, initializes virtual environment, verifies both |
| `set up a FastAPI project with postgres called myapp_db` | Scaffolds full project structure, creates venv, installs FastAPI, asyncpg, SQLAlchemy, generates `main.py` and `.env`, installs python-dotenv |
| `create a folder called output in the documents folder inside downloads` | Resolves nested path to `C:/Users/.../Downloads/Documents/output`, creates both directories with dependency ordering |
| `install requests in the testapp virtual environment` | Locates venv Python executable, runs pip install, verifies exit code |
| `find all .tmp files in Downloads and delete them` | Searches for matches, deletes each, verifies deletion |
| `create a folder called testwork in downloads and write a file called notes.txt with the text hello from alara` | Creates directory, writes file with content, verifies `check_file_contains` |

---

## Configuration Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | — | Yes | Gemini 2.5 Flash API key for planning and reflection |
| `MAX_RETRIES` | `3` | No | Maximum retry attempts per step before Reflector invocation |
| `STEP_TIMEOUT_S` | `30` | No | Per-step CLI execution timeout in seconds |
| `DEBUG` | `false` | No | Enables verbose output including raw Gemini responses |
| `LOG_FILE` | `alara.log` | No | Path for persistent log output |
| `DB_PATH` | `alara.db` | No | SQLite memory database path |

---

## Project Structure

```text
alara/
├── .env.example
├── .gitignore
├── __init__.py
├── alara-banner.jpg
├── main.py                          # CLI entrypoint and prompt loop
├── README.md
├── requirements.txt
├── start_server.bat                 # Starts WebSocket backend
├── start_ui.bat                     # Starts Electron overlay
│
├── capabilities/
│   ├── base.py                      # BaseCapability, CapabilityResult
│   ├── cli.py                       # subprocess-based CLI execution
│   ├── filesystem.py                # pathlib-based filesystem operations
│   ├── system.py                    # env vars, process checking
│   └── windows/
│       ├── app_adapters.py
│       ├── os_control.py
│       └── ui_automation.py
│
├── core/
│   ├── execution_router.py          # Step-to-capability routing
│   ├── goal_understander.py         # Raw input → GoalContext
│   ├── orchestrator.py              # Full orchestration loop
│   ├── planner.py                   # GoalContext → TaskGraph
│   ├── reflector.py                 # Failure reflection and recovery
│   └── verifier.py                  # Post-step verification
│
├── electron/
│   ├── main.js                      # Main process, hotkey, WebSocket client
│   ├── preload.js                   # Context bridge
│   ├── index.html                   # Overlay UI
│   └── package.json
│
├── memory/
│   ├── __init__.py                  # MemoryManager singleton
│   ├── database.py                  # DatabaseManager, migrations, WAL
│   ├── models.py                    # SessionEntry, PreferenceEntry, SkillEntry
│   ├── preferences.py               # Preferences, path aliases, inference
│   ├── session.py                   # Session tracking and history
│   └── skills.py                    # Skill storage and similarity search
│
├── schemas/
│   ├── goal.py                      # GoalContext
│   └── task_graph.py                # TaskGraph, Step, StepResult, enums
│
├── server/
│   ├── __init__.py
│   └── websocket_server.py          # Async WebSocket server
│
├── tests/
│   ├── test_planner.py              # 8-goal planner integration suite
│   └── test_week56_integrations.py
│
└── utils/
    └── platform.py
```

---

## Testing

### Planner Integration Tests

Validates the planning stack against 8 benchmark goals with 14 assertions per goal covering step schema integrity, dependency graph correctness, enum validity, verification method whitelist compliance, path normalization, and timestamp format.

```powershell
python -m tests.test_planner
```

Exits with code `1` if any assertion fails. Requires `GEMINI_API_KEY` in `.env`.

### Memory Layer Health Check

```powershell
python -c "
from dotenv import load_dotenv
load_dotenv()
from alara.memory import MemoryManager
import json
print(json.dumps(MemoryManager.get_instance().health_check(), indent=2))
"
```

### End-to-End Execution

```powershell
python -m alara.main --debug --goal "create a folder called test on my desktop"
```

---

## Logging Standards

| Component | INFO | WARNING | ERROR |
|---|---|---|---|
| Orchestrator | Step start, step success, retry notice, skip notice | Step failure, verification failure, fallback used | Step escalated, unrecoverable failure |
| Router | — | Capability not implemented, fallback used | Routing exception |
| Verifier | — | Unknown verification method | — |
| Reflector | Reflection started, action decided | Parse failure, fallback to escalate | API failure |
| Capabilities | Operation, resolved path, command | Path not found, non-zero exit | Exception during execution |
| Memory | Initialization, successful stores | Inference failures, memory update failures | Database errors |
| WebSocket Server | Client connected/disconnected | Send failure | Pipeline exception |

---

## Roadmap

### Near Term
- **Voice input** — Deepgram streaming integrated into the overlay via microphone button
- **VS Code adapter** — Open projects, create files, run terminals within VS Code
- **Browser adapter** — Tab creation, navigation, and form interaction via CDP

### Medium Term
- **Multi-agent framework** — Base `Agent` class with independent planning loops and model assignment per agent type (Coding Agent, Writing Agent, Research Agent, Browser Agent)
- **Master Orchestrator upgrade** — Goal decomposition into agent assignments rather than single-agent step sequences
- **Parallel agent execution** — Independent agents running concurrently via async coordination

### Long Term
- **Skill marketplace** — Shareable, versioned skill templates
- **Multi-device memory sync** — PostgreSQL-backed preference and skill synchronization across machines
- **macOS support** — Platform-specific capability implementations for macOS

---

## License

MIT