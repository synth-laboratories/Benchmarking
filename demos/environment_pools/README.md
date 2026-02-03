# Environment Pools Demo

Demonstrates the Synth environment pools system: pool CRUD + harbor rollout execution in Daytona sandboxes.

## Setup

```bash
# Clone sibling repos
git clone https://github.com/synth-laboratories/synth-ai.git ../synth-ai
git clone https://github.com/synth-laboratories/terminal-bench-3.git ../terminal-bench-3

# Install SDK
cd ../synth-ai && pip install -e .
```

## Harbor E2E

Runs the crafter-achievement oracle solution through the full pipeline: creates a pool, submits a rollout, executes `cargo check` + `cargo test` in a Daytona sandbox, and verifies the reward score.

```bash
export SYNTH_API_KEY=sk_live_...
python demos/environment_pools/harbor_e2e.py
```

Expected output:
```
RESULT: PASSED
  reward:       1.0
  compile_ok:   1
  tests_passed: 6/6
```

### Options

```
--backend-url URL    Backend URL (default: https://api-dev.usesynth.ai)
--snapshot-id ID     Daytona snapshot (default: tb3-crafter-base-v1)
--oracle-path PATH   Path to oracle .rs file (auto-detected from terminal-bench-3)
--timeout SEC        Rollout timeout (default: 600)
```

## Smoke Test

Tests pool/queue APIs and optionally submits rollouts for each backend type (harbor, openenv, archipelago, browser).

```bash
export SYNTH_API_KEY=sk_live_...
python demos/environment_pools/test_env_pools.py
python demos/environment_pools/test_env_pools.py --only harbor --harbor-deployment-id <id>
```
