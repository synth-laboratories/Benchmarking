#!/usr/bin/env python3
"""
Harbor E2E: pool CRUD + rollout with real Daytona sandbox execution.

Demonstrates the full flow:
  1. Create a pool (sandbox type, harbor backend)
  2. Submit a rollout with an oracle solution + eval script
  3. Daytona sandbox runs: cargo check + cargo test
  4. Poll until succeeded, print reward + metrics
  5. Clean up pool

Usage:
    export SYNTH_API_KEY=sk_live_...
    python demos/environment_pools/harbor_e2e.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure synth-ai is importable when run from benchmarking repo
_SYNTH_AI = Path(__file__).resolve().parents[3] / "synth-ai"
if _SYNTH_AI.exists():
    sys.path.insert(0, str(_SYNTH_AI))

from synth_ai.sdk.environment_pools import (
    create_pool,
    create_rollout,
    delete_pool,
    get_rollout,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Harbor E2E: pool + rollout with Daytona sandbox")
parser.add_argument(
    "--backend-url",
    default="https://api-dev.usesynth.ai",
    help="Backend URL (default: https://api-dev.usesynth.ai)",
)
parser.add_argument("--api-key", default=None, help="API key (or set SYNTH_API_KEY)")
parser.add_argument("--snapshot-id", default="tb3-crafter-base-v1", help="Daytona snapshot")
parser.add_argument("--timeout", type=int, default=600, help="Rollout timeout in seconds")
parser.add_argument(
    "--oracle-path",
    default=None,
    help="Path to oracle solution .rs file (auto-detected if not set)",
)
args = parser.parse_args()

BACKEND = args.backend_url.rstrip("/")
API_KEY = args.api_key or os.environ.get("SYNTH_API_KEY", "")
SNAPSHOT = args.snapshot_id

if not API_KEY:
    print("ERROR: Set SYNTH_API_KEY or pass --api-key")
    sys.exit(1)

def _default_task_ref() -> dict[str, str]:
    dataset = os.environ.get("ENV_POOL_TASK_DATASET", "env-pools-demo")
    task_id = os.environ.get("ENV_POOL_TASK_ID", "harbor-e2e")
    version = os.environ.get("ENV_POOL_TASK_VERSION")
    ref: dict[str, str] = {"dataset": dataset, "task_id": task_id}
    if version:
        ref["version"] = version
    return ref


def _default_agent() -> dict[str, str]:
    harness = os.environ.get("ENV_POOL_AGENT_HARNESS", "opencode")
    harness_version = os.environ.get("ENV_POOL_AGENT_VERSION")
    model_id = os.environ.get("ENV_POOL_AGENT_MODEL_ID", "gpt-4o-mini")
    agent: dict[str, str] = {"harness": harness}
    if harness_version:
        agent["harness_version"] = harness_version
    if model_id:
        agent["model_id"] = model_id
    return agent


def _task_app_url() -> str | None:
    value = os.environ.get("ENV_POOL_TASK_APP_URL", "").strip()
    return value or None

# ---------------------------------------------------------------------------
# Oracle solution
# ---------------------------------------------------------------------------
ORACLE_CANDIDATES = [
    args.oracle_path,
    str(Path(__file__).resolve().parents[3] / "terminal-bench-3/tasks/crafter-achievement/solution/achievement.rs"),
    str(Path(__file__).resolve().parents[2] / "terminal-bench-3/tasks/crafter-achievement/solution/achievement.rs"),
]
oracle_solution = None
for candidate in ORACLE_CANDIDATES:
    if candidate and Path(candidate).exists():
        oracle_solution = Path(candidate).read_text()
        print(f"Oracle: {len(oracle_solution)} bytes from {candidate}")
        break
if not oracle_solution:
    print("ERROR: Oracle solution not found.")
    print("  Either pass --oracle-path or ensure terminal-bench-3 is checked out as a sibling.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Eval runner script (runs inside Daytona sandbox)
# ---------------------------------------------------------------------------
EVAL_RUNNER = r'''
import json, subprocess, re, sys, os
data = json.load(open("/tmp/rollout.json"))
solution = data.get("agent_solution", "")
if not solution:
    json.dump({"metrics":{"reward_mean":0},"success":False,"error":"no solution"}, open("/tmp/result.json","w"))
    sys.exit(0)
os.makedirs("/app/modules/achievement", exist_ok=True)
open("/app/modules/achievement/mod.rs","w").write(solution)
print(f"Wrote {len(solution)}b")
os.chdir("/app")
check = subprocess.run(["cargo","check"], capture_output=True, text=True, timeout=120)
print(check.stderr[-1000:])
compile_ok = 1 if check.returncode == 0 else 0
tp, tt = 0, 0
if compile_ok:
    t = subprocess.run(["cargo","test","--lib","achievement::modular_tests","--","--test-threads=1"], capture_output=True, text=True, timeout=120)
    print(t.stderr[-1000:])
    out = t.stdout + t.stderr
    m = re.search(r"(\d+) passed", out)
    if m: tp = int(m.group(1))
    m = re.search(r"(\d+) failed", out)
    f = int(m.group(1)) if m else 0
    tt = tp + f
    if t.returncode != 0 and tt == 0: tt = 6
score = (0.3 + 0.7 * tp/tt) if compile_ok and tt > 0 else (0.3 if compile_ok else 0.0)
result = {"metrics":{"reward_mean":score,"outcome_reward":score,"compile_ok":compile_ok,"tests_passed":tp,"tests_total":tt},"success":compile_ok==1}
json.dump(result, open("/tmp/result.json","w"))
print(json.dumps(result))
'''

ENTRYPOINT = (
    "python3 -c 'import json; data=json.load(open(\"/tmp/rollout.json\")); "
    "open(\"/tmp/eval_runner.py\",\"w\").write(data[\"eval_script\"]); "
    "print(\"bootstrap done\")' && python3 /tmp/eval_runner.py"
)


def common():
    return {"backend_base": BACKEND, "api_key": API_KEY}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("HARBOR E2E: Pool + Rollout with Daytona Sandbox")
    print("=" * 60)
    print(f"Backend:   {BACKEND}")
    print(f"Snapshot:  {SNAPSHOT}")
    print()

    # 1. Create pool
    print("1. Creating pool...")
    pool = create_pool(
        **common(),
        request={
            "pool_type": "sandbox",
            "capacity": 5,
            "concurrency": 2,
            "tasks": [
                {
                    "task_id": "crafter-achievement",
                    "backend": "harbor",
                    "docker_image": SNAPSHOT,
                    "harbor": {
                        "entrypoint": ENTRYPOINT,
                        "entrypoint_mode": "file",
                        "env_vars": {},
                        "limits": {"timeout_s": args.timeout, "memory_mb": 8192},
                        "snapshot_id": SNAPSHOT,
                    },
                }
            ],
        },
    )
    pool_id = pool["pool"]["pool_id"]
    print(f"   Pool: {pool_id}")

    status = "unknown"
    try:
        # 2. Submit rollout
        print("\n2. Submitting rollout...")
        rollout_request = {
            "task_ref": _default_task_ref(),
            "agent": _default_agent(),
            "environment": {"backend": "harbor", "docker_image": SNAPSHOT},
            "pool_id": pool_id,
            "harbor": {
                "input": {
                    "seed": 42,
                    "task_id": "crafter-achievement",
                    "trace_correlation_id": "harbor_e2e_demo",
                    "agent_solution": oracle_solution,
                    "eval_script": EVAL_RUNNER,
                    "snapshot_id": SNAPSHOT,
                },
                "deployment": {
                    "entrypoint": ENTRYPOINT,
                    "entrypoint_mode": "file",
                    "env_vars": {},
                    "limits": {"timeout_s": args.timeout, "memory_mb": 8192},
                    "snapshot_id": SNAPSHOT,
                },
            },
        }
        task_app_url = _task_app_url()
        if task_app_url:
            rollout_request["task_app_url"] = task_app_url
        r = create_rollout(
            **common(),
            request=rollout_request,
            timeout=30.0,
        )
        rollout_id = r.get("rollout_id") or r.get("trial_id")
        if not rollout_id:
            raise RuntimeError("rollout response missing rollout_id")
        print(f"   Rollout: {rollout_id} ({r['status']})")

        # 3. Poll until done
        print("\n3. Polling...")
        s = {}
        for i in range(120):
            time.sleep(5)
            s = get_rollout(rollout_id, **common())
            st = s.get("status", "?")
            elapsed = (i + 1) * 5
            print(f"   [{elapsed:4d}s] {st}")
            if st in ("succeeded", "failed"):
                break
        else:
            print("   TIMEOUT")
            s = get_rollout(rollout_id, **common())

        # 4. Print results
        print("\n" + "=" * 60)
        status = s.get("status", "?")
        reward = s.get("reward_primary")
        metrics = s.get("reward_metrics", {})
        error = s.get("error")

        if status == "succeeded":
            print("RESULT: PASSED")
            print(f"  reward:       {reward}")
            print(f"  compile_ok:   {metrics.get('compile_ok')}")
            print(f"  tests_passed: {int(metrics.get('tests_passed', 0))}/{int(metrics.get('tests_total', 0))}")
        else:
            print("RESULT: FAILED")
            print(f"  error: {error}")
            print(json.dumps(s, indent=2, default=str))

        print("=" * 60)

    finally:
        # 5. Cleanup
        print("\n5. Cleaning up pool...")
        try:
            delete_pool(pool_id, **common())
            print("   Done.")
        except Exception as e:
            print(f"   Cleanup failed: {e}")

    sys.exit(0 if status == "succeeded" else 1)


if __name__ == "__main__":
    main()
