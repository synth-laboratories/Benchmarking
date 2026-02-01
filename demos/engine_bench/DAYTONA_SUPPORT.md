# Daytona Support for EngineBench Eval

This document describes how to use Daytona sandboxing as an alternative to local execution for the EngineBench eval job.

## Overview

The `run_eval.py` script now supports two execution modes:

1. **Local Mode** (default): Runs the task app locally on your machine
2. **Daytona Mode**: Runs the task app in a cloud Daytona sandbox

Both modes are fully supported and validated. Choose based on your needs:
- **Local**: Faster iteration, requires local dependencies (Rust toolchain, etc.)
- **Daytona**: Isolated environment, no local dependencies needed, cloud-based

## Usage

### Local Mode (Default)

```bash
# Basic usage
uv run python demos/engine_bench/run_eval.py --local

# With options
uv run python demos/engine_bench/run_eval.py --local \
  --seeds 5 \
  --model gpt-4.1-mini \
  --agent opencode \
  --timeout 300
```

### Daytona Mode

```bash
# Set Daytona API key
export DAYTONA_API_KEY=your_daytona_api_key

# Optional: Set API URL and target
export DAYTONA_API_URL=https://app.daytona.io/api  # Optional
export DAYTONA_TARGET=us  # Optional

# Run with --daytona flag
uv run python demos/engine_bench/run_eval.py --daytona

# With options
uv run python demos/engine_bench/run_eval.py --daytona \
  --seeds 5 \
  --model gpt-4.1-mini \
  --agent opencode \
  --timeout 300
```

## Requirements

### Local Mode
- Python 3.11+
- Rust toolchain (for cargo tests)
- Engine-bench repo cloned locally
- All dependencies installed

### Daytona Mode
- Python 3.11+
- `daytona` SDK installed: `pip install daytona`
- `DAYTONA_API_KEY` environment variable set
- Daytona account with API access

## How It Works

### Local Mode
1. Starts FastAPI task app locally on `localhost:8017` (or next available port)
2. Task app creates temp directories for each rollout
3. Agents (opencode/codex) run locally in those temp directories
4. Results are collected and reported

### Daytona Mode
1. Provisions a new Daytona sandbox from base image
2. Uploads task app code (`localapi_engine_bench.py`) to `/app/task_app.py`
3. Installs dependencies (fastapi, uvicorn, synth-ai)
4. Sets environment variables (API keys, backend URL, etc.)
5. Starts task app in sandbox bound to `0.0.0.0:8000`
6. Gets Daytona preview URL (e.g., `https://8000-sandbox123.proxy.daytona.work`)
7. Uses preview URL for eval job instead of localhost
8. Cleans up sandbox after job completes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    run_eval.py                              │
│                                                             │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  Local Mode  │              │ Daytona Mode │            │
│  │              │              │              │            │
│  │ 1. Start     │              │ 1. Provision │            │
│  │    local     │              │    sandbox   │            │
│  │    server    │              │              │            │
│  │              │              │ 2. Upload    │            │
│  │ 2. Use       │              │    task app  │            │
│  │    localhost │              │              │            │
│  │    URL       │              │ 3. Get       │            │
│  │              │              │    preview   │            │
│  │              │              │    URL        │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                            │                     │
│         └────────────┬───────────────┘                     │
│                      │                                     │
│                      ▼                                     │
│              ┌──────────────┐                             │
│              │  EvalJob     │                             │
│              │  (SDK)       │                             │
│              └──────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Files Added
- `daytona_helper.py`: Helper module for Daytona operations
  - `DaytonaTaskAppRunner`: Manages sandbox lifecycle
  - `run_task_app_in_daytona()`: Convenience function

### Files Modified
- `run_eval.py`: Added `--daytona` flag and mode switching logic

### Key Changes
1. Added `--daytona` argument parser flag
2. Validation to ensure only one mode is selected
3. Daytona mode:
   - Provisions sandbox using `DaytonaTaskAppRunner`
   - Uploads task app file
   - Sets up environment
   - Starts task app in sandbox
   - Uses preview URL for eval job
   - Cleans up sandbox on completion

## Troubleshooting

### Daytona Mode Issues

**Error: "DAYTONA_API_KEY environment variable required"**
- Solution: Set `export DAYTONA_API_KEY=your_key`

**Error: "Daytona SDK not available"**
- Solution: Install with `pip install daytona`

**Error: "Preview URL not available"**
- Solution: Check Daytona account permissions and API access

**Task app fails to start in sandbox**
- Check sandbox logs: `sandbox.process.exec("cat /app/task_app.log")`
- Verify dependencies installed correctly
- Check environment variables are set

### Local Mode Issues

**Port already in use**
- Solution: Use `--port` flag to specify different port
- Or: Script automatically finds next available port

**Agent binary not found**
- Solution: Ensure `opencode` or `codex` is in PATH
- Or: Set `OPENCODE_BIN` or `CODEX_BIN` environment variables

## Validation

Both modes are validated to ensure:
- ✅ Task app starts successfully
- ✅ Health check passes
- ✅ Eval job can connect to task app
- ✅ Rollouts execute correctly
- ✅ Results are collected properly
- ✅ Cleanup happens on completion

## Future Enhancements

Potential improvements:
- Support for custom Daytona images
- Snapshot caching for faster sandbox provisioning
- Parallel sandboxes for concurrent eval jobs
- Better error handling and retry logic
- Support for hello_world_bench demo
