<p align="center">
  <img src="./alara-banner.jpg" alt="ALARA Banner" width="100%" />
</p>

# ALARA
**Ambient Language & Reasoning Assistant**

ALARA is an agentic desktop AI platform for Windows that takes a natural language goal and decomposes it into an executable TaskGraph.

Version: 0.2.0  |  Platform: Windows 10/11  |  Python: 3.11+

## Overview

ALARA is an autonomous planning engine for desktop tasks. It is not a voice assistant, macro recorder, or conversational chatbot. A user provides a goal in natural language, and ALARA decomposes it into a structured execution plan with typed steps, dependencies, verification methods, and fallback strategies.

Current build focus: planning stack foundation (`GoalUnderstander` + `Planner` + `TaskGraph` rendering). Execution routing/orchestration is scaffolded and will be wired in subsequent builds.

## Capabilities

| Category | What it does |
|---|---|
| Filesystem Operations | Create, read, write, move, copy, and delete files and directories. Search by name or pattern. Operations are verified after execution. |
| CLI Execution | Run shell commands, scripts, and package managers in any working directory. Capture exit codes and use them as verification signals. |
| Project Scaffolding | Set up development project structures, including virtual environments, dependency installation, git initialization, and config file generation. |
| File Organization | Batch rename, move, sort, and clean files across directories using rules derived from the goal description. |
| App Control | Launch, focus, and close Windows applications by name, including argument and working-directory passing for supported apps. |
| System Queries | Read environment variables, list running processes, check disk usage, and retrieve host system information. |

## How It Works

```text
User Input
    │
    ▼
Goal Understander ──► extracts scope, constraints, working directory
    │
    ▼
Planner ──────────► decomposes goal into typed, ordered TaskGraph
    │
    ▼
┌─────────────────────────────────────────┐
│           Orchestration Loop            │
│                                         │
│   Execution Router                      │
│   └─► OS API → App Adapter → CLI →     │
│        UI Automation                    │
│              │                          │
│   Verifier ◄─┘                          │
│   └─► confirmed / failed                │
│              │                          │
│   Reflector (on failure)                │
│   └─► replan → retry                    │
└─────────────────────────────────────────┘
    │
    ▼
Result
```

**Goal Understander:** Parses raw input into a structured `GoalContext` that captures normalized intent, operational scope, explicit constraints, inferred working directory, and estimated complexity.

**Planner:** Sends `GoalContext` to Gemini 2.5 Flash using a constrained planning prompt and receives a typed `TaskGraph` of ordered atomic steps. Each step includes operation, parameters, expected outcome, verification method, dependencies, and fallback strategy.

**Execution Router:** Selects the best execution layer for each step in strict priority order: native OS API, application adapter, CLI execution, then UI automation as a last resort.

**Verifier:** Validates real-world state after each step against expected outcomes using programmatic checks such as file existence, process state, exit code status, port availability, and output inspection.

**Reflector:** On failed verification, sends full execution context to Gemini, including original goal, full plan, prior step results, and failure details. Gemini returns corrected actions or alternative paths, and ALARA retries within configured limits.

## Getting Started

### Prerequisites

ALARA requires Python 3.11 or later. An NVIDIA GPU is optional and not required for this release. Gemini API access is required before setup.

**Gemini API Key**

ALARA uses Gemini 2.5 Flash for planning and reflection. Create a free API key at `https://aistudio.google.com`.

### Installation

```powershell
git clone https://github.com/your-username/alara.git
cd alara
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```powershell
copy .env.example .env
```

```env
GEMINI_API_KEY=your_key_here
MAX_RETRIES=3
STEP_TIMEOUT_S=30
DEBUG=false
LOG_FILE=alara.log
DB_PATH=alara.db
```

## Usage

### Interactive Mode (Planner Preview)

```powershell
# If your current directory is the parent repo directory:
python -m alara.main

# If your current directory is the package directory:
python main.py
```

ALARA starts a prompt loop. Enter a natural language goal and press Enter. ALARA runs Goal Understanding + Planning and prints the TaskGraph in a Rich table.

### Single Goal Mode

```powershell
python -m alara.main --goal "create a Python project called myapp"
```

This mode plans one goal non-interactively and exits.

### Debug Mode

```powershell
python -m alara.main --debug
```

Debug mode prints parsed `GoalContext`, raw Gemini planner response, and additional TaskGraph metadata.

### Example Goals

| Goal | What ALARA Does |
|---|---|
| "Create a FastAPI project called myapp with a venv" | Scaffolds the project directory, creates a virtual environment, installs FastAPI, and generates `main.py` with a hello-world route. |
| "Find all .log files in Downloads and delete them" | Searches for matching files, deletes each match, and verifies deletion. |
| "Set up a git repository in my current project folder" | Runs `git init`, creates a Python `.gitignore`, and makes an initial commit. |
| "Rename all images in Desktop/photos to include today's date" | Enumerates image files, generates date-prefixed names, renames files, and verifies results. |
| "Find the 10 largest files in Documents" | Scans recursively, sorts by file size, and prints a formatted top-10 report. |
| "Install Python dependencies from requirements.txt" | Detects active virtual environment, runs `pip install -r requirements.txt`, and validates installed packages. |

## Planner Validation Tests

Run planner integration checks over 8 benchmark goals:

```powershell
# From parent repo directory:
python -m tests.test_planner

# From package directory:
python -m tests.test_planner
```

Notes:
- `GEMINI_API_KEY` must be set in `.env`.
- The script exits with code `1` if any goal fails validation.
- It validates step schema integrity, dependencies, verification methods, path normalization, and timestamp format.

## Project Structure

```text
alara/
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── __init__.py                   # Package marker
├── alara-banner.jpg              # README banner image
├── main.py                       # CLI entrypoint and prompt loop
├── README.md                     # Product documentation
├── requirements.txt              # Python dependencies
├── test_results.json             # Test output artifact
├── capabilities/
│   ├── __init__.py               # Capabilities package marker
│   ├── base.py                   # Capability base contract
│   ├── cli.py                    # CLI execution capability
│   ├── filesystem.py             # Filesystem execution capability
│   └── windows/
│       ├── __init__.py           # Windows capabilities marker
│       ├── app_adapters.py       # App adapter capability
│       ├── os_control.py         # Native Windows control capability
│       └── ui_automation.py      # UI automation fallback capability
├── core/
│   ├── __init__.py               # Core package marker
│   ├── action_registry.py        # Action definitions
│   ├── assistant.py              # Legacy assistant compatibility facade
│   ├── audio_preprocessor.py     # Audio preprocessing utilities
│   ├── execution_router.py       # Step-to-capability routing
│   ├── executor.py               # Action executor implementation
│   ├── goal_understander.py      # Raw goal to GoalContext extraction
│   ├── intent_engine.py          # Intent parsing engine
│   ├── normalizer.py             # Text normalization utilities
│   ├── orchestrator.py           # Top-level orchestration coordinator
│   ├── pipeline.py               # Legacy pipeline compatibility shim
│   ├── planner.py                # GoalContext to TaskGraph planning
│   ├── prompt_builder.py         # LLM prompt composition logic
│   ├── reflector.py              # Failure reflection and replan hook
│   ├── registry_loader.py        # Registry loading utilities
│   ├── verifier.py               # Step outcome verification
│   ├── voice_profile.py          # Voice profile data model
│   └── ws_server.py              # WebSocket bridge placeholder
├── integrations/
│   ├── __init__.py               # Integrations package marker
│   ├── browser.py                # Browser integration handlers
│   ├── terminal.py               # Terminal integration handlers
│   ├── vscode.py                 # VS Code integration handlers
│   └── windows_os.py             # Windows OS integration handlers
├── memory/
│   ├── __init__.py               # Memory package marker
│   ├── preferences.py            # Preferences store abstraction
│   ├── session.py                # Session memory abstraction
│   └── skills.py                 # Skill-pattern store abstraction
├── schemas/
│   ├── __init__.py               # Schemas package marker
│   ├── goal.py                   # GoalContext schema
│   └── task_graph.py             # TaskGraph and step schemas
├── tests/
│   ├── __init__.py               # Test package marker
│   ├── test_intent.py            # Intent engine benchmark tests
│   ├── test_planner.py           # Planner integration validation script
│   └── test_week56_integrations.py # Integration behavior tests
├── ui/
│   ├── alara.svg                 # Overlay brand asset
│   ├── index.html                # Overlay renderer UI
│   ├── main.js                   # Electron main process
│   ├── package-lock.json         # NPM lockfile
│   ├── package.json              # UI package manifest
│   └── preload.js                # Renderer IPC bridge
└── utils/
    ├── __init__.py               # Utilities package marker
    └── platform.py               # Platform/path helpers
```

## Configuration Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| GEMINI_API_KEY | — | Yes | Gemini 2.5 Flash API key. |
| MAX_RETRIES | 3 | No | Maximum retry attempts per step. |
| STEP_TIMEOUT_S | 30 | No | Timeout in seconds per step execution. |
| DEBUG | false | No | Enables verbose logging output. |
| LOG_FILE | alara.log | No | Log file path. |
| DB_PATH | alara.db | No | SQLite memory database path. |

## Overlay UI

ALARA includes an Electron-based floating overlay triggered system-wide with `Ctrl+Space`. The overlay behaves as a command palette for submitting goals and observing execution state in real time, and communicates with the backend over local WebSocket `ws://localhost:8765`.

| State | Indicator Color | Meaning |
|---|---|---|
| Ready | White | Waiting for input. |
| Planning | Purple | Decomposing goal into executable steps. |
| Executing | Blue | Running the active step. |
| Verifying | Amber | Validating step outcome. |
| Reflecting | Red | Correcting a failed step via replan. |
| Done | Green | Task completed successfully. |
| Failed | Red | Task failed after maximum retries. |

```powershell
cd ui
npm install
npm start
```

## License

MIT
