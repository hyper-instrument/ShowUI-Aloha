# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShowUI-Aloha is a human-taught computer-use agent for Windows/macOS. It learns from screen recordings and replays workflows via OS-level automation. The codebase has two top-level modules:

- **Aloha_Learn** — ingests raw recorder outputs and generates semantic trace JSONs via LLM.
- **Aloha_Act** — server/client runtime that plans and executes GUI actions from a trace.

## Common Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Generate a trace from a recorded project:
```bash
python Aloha_Learn/parser.py {project_name}
```
Input: `Aloha_Learn/projects/{project_name}/`. Output: `Aloha_Learn/projects/{project_name}_trace.json` and a copy in `Aloha_Act/trace_data/`.

Run the end-to-end orchestrator (launches server + client, sends task, tears down):
```bash
python Aloha_Act/scripts/aloha_run.py --task "Your task" --trace_id "{trace_id}"
```

Run server and client manually (for development/debugging):
```bash
# Terminal 1 — action server (planner + actor)
python Aloha_Act/app_server.py        # port 7887

# Terminal 2 — executor client (screenshot + pyautogui)
python Aloha_Act/app_client.py        # port 7888

# Terminal 3 — start a task via HTTP
curl -X POST http://127.0.0.1:7888/run_task \
  -H 'Content-Type: application/json' \
  -d '{"task":"open settings","trace_id":"example_trace","max_steps":10,"server_url":"http://127.0.0.1:7887/generate_action"}'
```

Run tests:
```bash
pytest Aloha_Act/scripts/test_aloha_run.py
python Aloha_Act/scripts/test_executor_from_action.py
python Aloha_Act/scripts/test_autogui.py
```

Lint:
```bash
ruff check .
```

## Architecture

### Data Flow

1. **Record** → `Aloha_Learn/projects/{project}/inputs/` (log + video)
2. **Parse** → `Aloha_Learn/parser.py` runs `LogProcessor → VideoScreenshotExtractor → TraceGenerator`
3. **Trace** → `Aloha_Act/trace_data/{trace_id}.json` (list of `step_idx` + `caption` objects)
4. **Execute** → `Aloha_Act/scripts/aloha_run.py` (or manual server/client) runs the planner/actor loop

### Server (app_server.py)

- Flask app on port 7887.
- Single endpoint: `POST /generate_action`
- Required payload fields: `screenshot` (base64), `query` (task string)
- Optional: `trace_name`, `task_id`, `action_history`
- Internally calls `ui_aloha_loop()` which runs:
  - `TrajectoryManager.get_trajectory_in_context()` — loads trace JSON and formats step captions into a string for in-context learning.
  - `AlohaPlanner` — Jinja2-templated LLM call that outputs `Observation`, `Reasoning`, `Action`, `Current Step in Guidance Trajectory`.
  - `AlohaActor` — routes to one of three backends and emits the final low-level action.

### Client (app_client.py)

- Flask app on port 7888.
- Endpoints: `POST /run_task`, `POST /stop`
- `simple_sampling_loop()` drives the execution:
  1. Capture screenshot via `gui_capture.capture_screenshot()`
  2. Send to server `/generate_action`
  3. Parse plan + action
  4. If action is `STOP`, finish; else feed action to `AlohaExecutor`
  5. Append executed action to `action_history` and repeat

### Actor Backends

`AlohaActor` selects the agent based on `mode` (default from `config.yaml`):
- `oai-operator` → `OAIOperatorAgent`
- `claude-computer-use` → `ClaudeComputerUseAgent` (model selected via `claude_model` in `config.yaml`; supports both Anthropic-native names like `claude-sonnet-4-5-20250929` and third-party vendor names like `Vendor2/Claude-4.6-Sonnet`. The agent auto-selects the right beta header — `computer-use-2025-11-24` for Sonnet 4.6+ / Opus 4.5+, `computer-use-2025-01-24` otherwise.)
- `ui-tars` → `UITarsAgent`

### Executor

`AlohaExecutor` receives the actor's JSON action, parses it into a list of tool calls, and runs them through `ComputerTool` (pyautogui-based). It handles coordinate conversion for multi-monitor setups using `screeninfo` (Windows) or `Quartz` (macOS).

### LLM Layer

`ui_aloha.act.gui_agent.llm.run_llm` is the centralized LLM caller. It uses the OpenAI client and handles both Chat Completions and the Responses API (for `gpt-5`). It prepares messages, detects image paths, base64-encodes them, and returns `(text, token_usage_dict)`.

## Configuration

- `Aloha_Act/config/config.yaml` — sets `planner_model`, `actor_model`, `claude_model`, `os_name`, `log_dir`, `trace_dir`, `kimi_base_url`.
- `Aloha_Act/config/api_keys.json` (git-ignored) — holds `OPENAI_API_KEY`, `CLAUDE_API_KEY`, `OPERATOR_OPENAI_API_KEY`, `GOOGLE_API_KEY`, plus optional `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` for third-party Anthropic-compatible vendors (e.g. gpugeek). Env vars with the same names are also respected; when both `ANTHROPIC_AUTH_TOKEN` and `CLAUDE_API_KEY` are present, the vendor bearer token wins.
- `Aloha_Act/prompt.json` — optional convenience file; `aloha_run.py` will read `task` and `trace` from it if present.

## Key File Paths

- Traces: `Aloha_Act/trace_data/{trace_id}_trace.json` (parser output). The `TrajectoryManager` also accepts `{trace_id}.json`, `{trace_id}/trace.json`, and `{trace_id}` as fallbacks.
- Logs: `Aloha_Act/logs/` (per-request subdirectories)
- Prompt templates: `Aloha_Act/ui_aloha/act/gui_agent/prompt_templates/` (referenced by `AlohaPlanner` and `AlohaActor` via Jinja2; may need to be created if missing)
- Default prompt for trace generation: `Aloha_Learn/default_prompt.json`

## Notes

- The parser pipeline (`Aloha_Learn/parser.py`) is the only supported way to turn raw recordings into traces. Do not hand-write trace JSONs unless for minimal testing.
- When adding a new action type, update both `AlohaExecutor.supported_actions` and the parser dispatch table `_parsers` in `Aloha_Act/ui_aloha/execute/executor/aloha_executor.py`.
- The server is stateless; all context (screenshot, action history, trace name) travels in each request.
- Multi-monitor support is handled in the executor via `_get_selected_screen_offset()`, not in the actor.
