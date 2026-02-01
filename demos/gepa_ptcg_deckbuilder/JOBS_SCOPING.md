# Running Eval and GEPA Jobs for Deckbuilder Task

This document scopes out how to run **eval jobs** and **GEPA prompt learning jobs** against the Pokemon TCG Deckbuilder task app.

## Overview

The deckbuilder task app (`localapi_deckbuilder.py`) is already set up as a standard Synth task app that:
- Exposes `/rollout` endpoint for evaluation
- Accepts policy config (model, system prompt, etc.)
- Returns rewards based on constraint satisfaction + win rate
- Works with both eval jobs (single-shot evaluation) and GEPA jobs (prompt optimization)

## Current Setup

### Task App
- **File**: `localapi_deckbuilder.py`
- **App ID**: `ptcg_deckbuilder`
- **Port**: Default `8018` (auto-picks new port on conflict)
- **Endpoints**: `/rollout`, `/health`, `/taskset`, `/instances`

### Existing Eval Demo
- **File**: `run_demo.py`
- **Purpose**: Runs eval jobs via backend interceptor
- **Current approach**: Uses SDK `EvalJob` class

---

## 1. Running Eval Jobs

### Approach A: Using SDK (Current - `run_demo.py`)

The existing `run_demo.py` already demonstrates this pattern:

```python
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig

config = EvalJobConfig(
    local_api_url=task_url,  # e.g., "http://localhost:8018"
    backend_url=backend_url,  # e.g., "http://localhost:8000"
    api_key=api_key,
    env_name="deckbuilder",
    seeds=[0, 1, 2, 3, 4],
    policy_config={
        "model": "gpt-4.1-mini",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
    env_config={},
    concurrency=2,
)

job = EvalJob(config)
job_id = job.submit()
result = job.poll_until_complete(timeout=600.0, interval=5.0, progress=True)
```

**Pros:**
- Simple SDK interface
- Automatic polling and progress updates
- Handles task app startup/auth automatically

**Cons:**
- Requires SDK installation
- Less control over job lifecycle

### Approach B: Direct Backend API

Call the backend API directly:

```python
import httpx

# Start task app (or use existing)
task_url = "http://localhost:8018"
env_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)

# Create eval job
response = httpx.post(
    f"{backend_url}/api/eval/jobs",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "task_app_url": task_url,
        "task_app_api_key": env_key,
        "app_id": "ptcg_deckbuilder",
        "env_name": "deckbuilder",
        "seeds": [0, 1, 2, 3, 4],
        "policy": {
            "model": "gpt-4.1-mini",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        },
        "env_config": {},
        "max_concurrent": 2,
    },
)
job_id = response.json()["job_id"]

# Poll for results
while True:
    status = httpx.get(
        f"{backend_url}/api/eval/jobs/{job_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(5)

# Get results
results = httpx.get(
    f"{backend_url}/api/eval/jobs/{job_id}/results",
    headers={"Authorization": f"Bearer {api_key}"},
).json()
```

**Pros:**
- Full control over API calls
- No SDK dependency
- Can customize request/response handling

**Cons:**
- More boilerplate
- Manual polling logic

### Eval Job Configuration

**Required Fields:**
- `task_app_url`: URL where task app is running (e.g., `http://localhost:8018`)
- `task_app_api_key`: Environment API key (from `ensure_localapi_auth()`)
- `env_name`: `"deckbuilder"` (matches task app's env name)
- `seeds`: List of seeds to evaluate (e.g., `[0, 1, 2, 3, 4]`)
- `policy`: Policy config dict with:
  - `model`: Model name (e.g., `"gpt-4.1-mini"`)
  - `system_prompt`: System prompt (optional, defaults to `DEFAULT_SYSTEM_PROMPT`)
  - `inference_url`: Auto-injected by backend (interceptor URL)

**Optional Fields:**
- `app_id`: `"ptcg_deckbuilder"` (for metadata)
- `env_config`: Dict passed to task app (can specify `instance_id` to target specific challenge)
- `max_concurrent`: Concurrent rollouts (default: backend default)
- `timeout`: Per-seed timeout in seconds

**Example with Specific Challenge:**
```python
config = EvalJobConfig(
    local_api_url=task_url,
    backend_url=backend_url,
    api_key=api_key,
    env_name="deckbuilder",
    seeds=[0, 1, 2],  # Will round-robin through challenges
    policy_config={
        "model": "gpt-4.1-mini",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
    env_config={
        "instance_id": "gardevoir-deck",  # Target specific challenge
    },
)
```

---

## 2. Running GEPA Jobs

### Approach A: Using SDK

```python
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

# Create job from config dict
job = PromptLearningJob.from_dict(
    config_dict={
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_url,  # e.g., "http://localhost:8018"
            "task_app_id": "ptcg_deckbuilder",
            "policy": {
                "model": "gpt-4.1-mini",
                "provider": "openai",
                "temperature": 0.7,
            },
            "gepa": {
                "env_name": "deckbuilder",
                "proposer_type": "dspy",
                "proposer_effort": "LOW",
                "rollout": {
                    "budget": 50,
                    "max_concurrent": 5,
                    "minibatch_size": 10,
                },
                "evaluation": {
                    "train_seeds": [0, 1, 2, 3, 4],  # Seeds for optimization
                    "validation_seeds": [5, 6, 7, 8, 9],  # Optional validation set
                },
                "population": {
                    "initial_size": 10,
                    "num_generations": 3,
                    "children_per_generation": 8,
                },
                "mutation": {
                    "rate": 0.3,
                },
            },
        }
    },
    backend_url=backend_url,
    api_key=api_key,
    task_app_api_key=env_key,
)

# Submit job
job_id = job.submit()
print(f"Job submitted: {job_id}")

# Poll until complete
result = job.poll_until_complete(timeout=3600.0, interval=10.0, progress=True)
print(f"Best score: {result['best_score']}")
print(f"Best prompt: {result['best_prompt']}")
```

### Approach B: Using TOML Config File

Create `gepa_deckbuilder.toml`:

```toml
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8018"
task_app_id = "ptcg_deckbuilder"

[prompt_learning.policy]
model = "gpt-4.1-mini"
provider = "openai"
temperature = 0.7
max_completion_tokens = 4096

[prompt_learning.initial_prompt]
id = "baseline_deckbuilder"
name = "Baseline Deckbuilder Prompt"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = """You are an expert Pokemon TCG deck builder. Build decks that satisfy the given constraints AND win battles.

RULES:
- A deck must have exactly 60 cards
- Maximum 4 copies of any non-energy card (basic energy has no limit)
- Include enough Basic Pokemon (8-12 minimum)
- Evolution lines must be complete (Basic -> Stage 1 -> Stage 2)

STRATEGY TIPS:
- Include draw support trainers (TV Reporter)
- Match energy types to your Pokemon's attack costs
- Include multiple copies of key Pokemon for consistency
- Balance attackers with support Pokemon

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{"deck": ["card-id-1", "card-id-2", ...]}

The deck array must contain exactly 60 card IDs from the available pool.
Use the exact card IDs provided (e.g., "df-061-ralts", "hp-109-psychic-energy")."""
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "{user_prompt}"
order = 1

[prompt_learning.initial_prompt.wildcards]
user_prompt = "REQUIRED"

[prompt_learning.env_config]
# Optional: can specify instance_id to target specific challenge
# instance_id = "gardevoir-deck"

[prompt_learning.gepa]
env_name = "deckbuilder"
proposer_type = "dspy"
proposer_effort = "LOW"
proposer_output_tokens = "FAST"

[prompt_learning.gepa.rollout]
budget = 50
max_concurrent = 5
minibatch_size = 10

[prompt_learning.gepa.evaluation]
train_seeds = [0, 1, 2, 3, 4]
validation_seeds = [5, 6, 7, 8, 9]  # Optional
validation_top_k = 3

[prompt_learning.gepa.population]
initial_size = 10
num_generations = 3
children_per_generation = 8

[prompt_learning.gepa.mutation]
rate = 0.3
```

Then run:

```python
job = PromptLearningJob.from_config(
    config_path="gepa_deckbuilder.toml",
    backend_url=backend_url,
    api_key=api_key,
    task_app_api_key=env_key,
)
job_id = job.submit()
result = job.poll_until_complete()
```

### Approach C: Direct Backend API

```python
import httpx
from synth_ai.core.prompt_pattern import PromptPattern

# Build initial prompt pattern
initial_prompt = PromptPattern(
    id="baseline_deckbuilder",
    name="Baseline Deckbuilder Prompt",
    messages=[
        {
            "role": "system",
            "pattern": DEFAULT_SYSTEM_PROMPT,
            "order": 0,
        },
        {
            "role": "user",
            "pattern": "{user_prompt}",
            "order": 1,
        },
    ],
    wildcards={"user_prompt": "REQUIRED"},
)

# Create GEPA job
response = httpx.post(
    f"{backend_url}/api/prompt-learning/online/jobs",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "algorithm": "gepa",
        "config": {
            "prompt_learning": {
                "gepa": {
                    "env_name": "deckbuilder",
                    "proposer_type": "dspy",
                    "proposer_effort": "LOW",
                    "rollout": {
                        "budget": 50,
                        "max_concurrent": 5,
                        "minibatch_size": 10,
                    },
                    "evaluation": {
                        "train_seeds": [0, 1, 2, 3, 4],
                    },
                    "population": {
                        "initial_size": 10,
                        "num_generations": 3,
                        "children_per_generation": 8,
                    },
                    "mutation": {
                        "rate": 0.3,
                    },
                },
                "policy": {
                    "model": "gpt-4.1-mini",
                    "provider": "openai",
                    "temperature": 0.7,
                },
            }
        },
        "initial_prompt": initial_prompt.to_dict(),
        "task_app_url": task_url,
        "task_app_api_key": env_key,
        "org_id": org_id,  # From API key
    },
)
job_id = response.json()["job_id"]

# Poll for results (similar to eval jobs)
```

### GEPA Job Configuration

**Required Fields:**
- `algorithm`: `"gepa"`
- `task_app_url`: URL where task app is running
- `task_app_api_key`: Environment API key
- `initial_prompt`: PromptPattern dict (baseline prompt to optimize)
- `config`: Config dict with `prompt_learning.gepa` section

**GEPA-Specific Config:**
- `env_name`: `"deckbuilder"`
- `proposer_type`: `"dspy"` (or `"random"`, `"llm"`)
- `rollout.budget`: Total rollouts (e.g., `50`)
- `rollout.max_concurrent`: Concurrent rollouts (e.g., `5`)
- `evaluation.train_seeds`: Seeds for optimization (e.g., `[0, 1, 2, 3, 4]`)
- `population.initial_size`: Initial population size (e.g., `10`)
- `population.num_generations`: Number of generations (e.g., `3`)
- `population.children_per_generation`: Children per generation (e.g., `8`)

**Policy Config:**
- `model`: Model name (e.g., `"gpt-4.1-mini"`)
- `provider`: Provider (e.g., `"openai"`)
- `temperature`: Temperature (e.g., `0.7`)
- `max_completion_tokens`: Max tokens (e.g., `4096`)

---

## 3. Key Differences: Eval vs GEPA

| Aspect | Eval Jobs | GEPA Jobs |
|--------|-----------|-----------|
| **Purpose** | Evaluate a fixed prompt | Optimize/evolve prompts |
| **Input** | Single prompt (system_prompt) | Initial prompt pattern + optimization config |
| **Output** | Scores for each seed | Best prompt + score |
| **Rollouts** | One per seed | Many rollouts (budget-based) |
| **Algorithm** | None (direct evaluation) | Genetic algorithm (GEPA) |
| **Events** | Eval job events | GEPA-specific events (generations, mutations, etc.) |
| **Use Case** | Benchmarking, testing | Prompt engineering, optimization |

---

## 4. Implementation Plan

### Step 1: Create Eval Job Script

Create `run_eval_job.py` that:
1. Starts task app (or uses existing)
2. Creates eval job via SDK or API
3. Polls for completion
4. Prints results

**File**: `run_eval_job.py`

### Step 2: Create GEPA Job Script

Create `run_gepa_job.py` that:
1. Starts task app (or uses existing)
2. Creates GEPA job via SDK or API
3. Polls for completion
4. Prints best prompt and score

**File**: `run_gepa_job.py`

### Step 3: Create GEPA Config File

Create `gepa_deckbuilder.toml` with:
- Initial prompt pattern
- GEPA algorithm config
- Policy config
- Evaluation seeds

**File**: `gepa_deckbuilder.toml`

### Step 4: Update README

Update `README.md` to document:
- How to run eval jobs
- How to run GEPA jobs
- Configuration options
- Examples

---

## 5. Example Scripts

### `run_eval_job.py` (New)

```python
#!/usr/bin/env python3
"""Run eval job for deckbuilder task."""

import argparse
import asyncio
import os

from localapi_deckbuilder import DEFAULT_SYSTEM_PROMPT, INSTANCE_IDS, app
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

# ... (similar to run_demo.py but focused on eval job creation)
```

### `run_gepa_job.py` (New)

```python
#!/usr/bin/env python3
"""Run GEPA prompt learning job for deckbuilder task."""

import argparse
import asyncio
import os

from localapi_deckbuilder import DEFAULT_SYSTEM_PROMPT
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

# ... (create GEPA job and poll)
```

---

## 6. Testing Checklist

- [ ] Eval job runs successfully with default config
- [ ] Eval job runs with specific challenge (`instance_id`)
- [ ] Eval job returns correct scores
- [ ] GEPA job runs successfully with default config
- [ ] GEPA job optimizes prompts over generations
- [ ] GEPA job returns best prompt and score
- [ ] Both jobs work with local backend (`localhost:8000`)
- [ ] Both jobs work with production backend
- [ ] Task app handles concurrent rollouts correctly
- [ ] Results are stored and retrievable

---

## 7. Notes

### Task App Requirements
- Task app must be running before submitting jobs
- Task app must be accessible from backend (localhost or tunnel)
- Environment API key must be valid

### Backend Requirements
- Backend must be running (for interceptor)
- Backend must have access to task app URL
- API key must be valid

### Seed Mapping
- Seeds map to challenges via `seed % len(INSTANCE_IDS)`
- Can override with `env_config.instance_id` for specific challenge
- GEPA uses `train_seeds` for optimization, `validation_seeds` for validation

### Prompt Pattern
- GEPA requires `PromptPattern` with messages and wildcards
- System prompt goes in first message
- User prompt uses wildcard `{user_prompt}` (filled by task app)

---

## 8. Next Steps

1. **Create `run_eval_job.py`** - Standalone eval job runner
2. **Create `run_gepa_job.py`** - Standalone GEPA job runner
3. **Create `gepa_deckbuilder.toml`** - GEPA config file
4. **Update `README.md`** - Document both approaches
5. **Test both paths** - Verify end-to-end functionality
6. **Add examples** - Show different configurations
