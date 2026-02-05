#!/usr/bin/env python3
"""
Test the environment-pools proxy chain by creating one rollout of each type:
  1. Harbor          (EngineBench sandbox)
  2. OpenEnv         (Banking77 classification)
  3. Archipelago     (simple_task from apex agents)
  4. Browser         (LinkedIn bench)

Usage:
    export SYNTH_API_KEY=sk_live_...

    # Against dev
    uv run python demos/environment_pools/test_env_pools.py

    # Against local stack
    uv run python demos/environment_pools/test_env_pools.py --local

    # Run only one task type
    uv run python demos/environment_pools/test_env_pools.py --only harbor
"""

import argparse
import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Smoke-test environment-pools proxy chain")
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
parser.add_argument("--backend-url", type=str, default=None, help="Override backend URL")
parser.add_argument(
    "--only",
    type=str,
    choices=["harbor", "openenv", "archipelago", "browser"],
    help="Run only one task type",
)
parser.add_argument("--timeout", type=int, default=300, help="Per-rollout timeout (default: 300)")
parser.add_argument("--pool-id", type=str, default=None, help="Explicit pool_id for routing")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Use dry_run on create_rollout (validates + routes only)",
)
parser.add_argument(
    "--download-artifacts",
    action="store_true",
    help="Download artifacts.zip for completed rollouts (prints byte size)",
)
parser.add_argument(
    "--max-artifacts",
    type=int,
    default=5,
    help="Max artifacts to list/download per rollout",
)
parser.add_argument(
    "--harbor-deployment-id",
    type=str,
    default=None,
    help="Harbor deployment ID (required for harbor test)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
if args.local:
    BACKEND_URL = "http://localhost:8000"
elif args.backend_url:
    BACKEND_URL = args.backend_url.rstrip("/")
else:
    BACKEND_URL = "https://api-dev.usesynth.ai"

API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("ERROR: SYNTH_API_KEY not set")
    sys.exit(1)

print("=" * 60)
print("ENVIRONMENT POOLS SMOKE TEST")
print("=" * 60)
print(f"Backend:  {BACKEND_URL}")
print(f"API Key:  {API_KEY[:20]}...")
print(f"Timeout:  {args.timeout}s")
if args.pool_id:
    print(f"Pool ID:  {args.pool_id}")
print()

# ---------------------------------------------------------------------------
# SDK import
# ---------------------------------------------------------------------------
from synth_ai.sdk.environment_pools import (
    create_rollout,
    validate_rollout,
    list_rollouts,
    get_rollout,
    get_rollout_summary,
    get_rollout_usage,
    get_rollout_support_bundle,
    stream_rollout_events,
    list_rollout_artifacts,
    download_artifacts_zip,
    list_pools,
    get_pool_metrics,
    get_queue_status,
    get_capabilities,
    get_openapi_schema,
)


def _common_kwargs() -> dict[str, Any]:
    return {"backend_base": BACKEND_URL, "api_key": API_KEY}


def _default_task_ref(task_id: str) -> dict[str, Any]:
    dataset = os.environ.get("ENV_POOL_TASK_DATASET", "env-pools-demo")
    version = os.environ.get("ENV_POOL_TASK_VERSION")
    ref: dict[str, Any] = {"dataset": dataset, "task_id": task_id}
    if version:
        ref["version"] = version
    return ref


def _default_agent() -> dict[str, Any]:
    harness = os.environ.get("ENV_POOL_AGENT_HARNESS", "opencode")
    harness_version = os.environ.get("ENV_POOL_AGENT_VERSION")
    model_id = os.environ.get("ENV_POOL_AGENT_MODEL_ID", "gpt-4o-mini")
    agent: dict[str, Any] = {"harness": harness}
    if harness_version:
        agent["harness_version"] = harness_version
    if model_id:
        agent["model_id"] = model_id
    return agent


def _task_app_url() -> str | None:
    value = os.environ.get("ENV_POOL_TASK_APP_URL", "").strip()
    return value or None


def _poll_rollout(rollout_id: str, timeout: int) -> dict[str, Any]:
    """Poll rollout until terminal state or timeout."""
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        r = get_rollout(rollout_id, **_common_kwargs(), timeout=30)
        status = r.get("status", "unknown")
        if status != last_status:
            print(f"  [{rollout_id[:12]}] status: {status}")
            last_status = status
        if status in ("succeeded", "failed", "cancelled", "error", "completed"):
            return r
        time.sleep(3)
    print(f"  [{rollout_id[:12]}] TIMED OUT after {timeout}s")
    return get_rollout(rollout_id, **_common_kwargs())


def _validate_request(request: dict[str, Any], label: str) -> bool:
    """Run validation endpoint and report routing debug."""
    try:
        result = validate_rollout(request=request, **_common_kwargs(), timeout=15)
    except Exception as exc:
        print(f"  Validation failed ({label}): {exc}")
        return False
    valid = bool(result.get("valid"))
    selected = result.get("selected_pool_id")
    print(f"  Validation ({label}): valid={valid} selected_pool_id={selected}")
    routing_debug = result.get("routing_debug")
    if routing_debug:
        print(f"  Routing debug: {json.dumps(routing_debug, indent=2)[:400]}")
    if not valid:
        err = result.get("error")
        if err:
            print(f"  Validation error: {json.dumps(err, indent=2)[:400]}")
    return valid


def _post_rollout_checks(rollout_id: str) -> None:
    """Hit o11y endpoints for a rollout."""
    try:
        summary = get_rollout_summary(rollout_id, **_common_kwargs(), timeout=30)
        print(f"  Summary: {json.dumps(summary, indent=2)[:400]}")
    except Exception as exc:
        print(f"  Summary failed: {exc}")

    try:
        usage = get_rollout_usage(rollout_id, **_common_kwargs(), timeout=30)
        print(f"  Usage: {json.dumps(usage, indent=2)[:400]}")
    except Exception as exc:
        print(f"  Usage failed: {exc}")

    try:
        bundle = get_rollout_support_bundle(rollout_id, **_common_kwargs(), timeout=30)
        print(f"  Support bundle: {json.dumps(bundle, indent=2)[:400]}")
    except Exception as exc:
        print(f"  Support bundle failed: {exc}")

    try:
        artifacts = list_rollout_artifacts(
            rollout_id,
            **_common_kwargs(),
            limit=args.max_artifacts,
            timeout=30,
        )
        print(f"  Artifacts: {json.dumps(artifacts, indent=2)[:400]}")
        next_cursor = artifacts.get("next_cursor")
        if next_cursor:
            more = list_rollout_artifacts(
                rollout_id,
                **_common_kwargs(),
                cursor=next_cursor,
                limit=args.max_artifacts,
                timeout=30,
            )
            print(f"  Artifacts (page 2): {json.dumps(more, indent=2)[:400]}")
    except Exception as exc:
        print(f"  Artifacts listing failed: {exc}")

    if args.download_artifacts:
        try:
            zipped = download_artifacts_zip(
                rollout_id,
                **_common_kwargs(),
                limit=args.max_artifacts,
                timeout=120,
            )
            print(f"  Artifacts.zip bytes: {len(zipped)}")
        except Exception as exc:
            print(f"  Artifacts.zip failed: {exc}")


# ---------------------------------------------------------------------------
# 1. Harbor rollout
# ---------------------------------------------------------------------------
def test_harbor():
    deployment_id = args.harbor_deployment_id
    if not deployment_id:
        print("SKIP harbor: --harbor-deployment-id not provided")
        return None

    print("-" * 60)
    print("1. HARBOR ROLLOUT")
    print("-" * 60)

    request: dict[str, Any] = {
        "task_ref": _default_task_ref("harbor-smoke"),
        "agent": _default_agent(),
        "environment": {"backend": "harbor"},
        "harbor": {
            "input": {
                "seed": 0,
                "policy_config": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                },
            },
            "deployment": {
                "entrypoint": "python main.py",
            },
        },
        "timeouts": {"agent_sec": args.timeout},
    }
    task_app_url = _task_app_url()
    if task_app_url:
        request["task_app_url"] = task_app_url
    if args.pool_id:
        request["pool_id"] = args.pool_id

    if not _validate_request(request, "harbor"):
        return None

    if args.dry_run:
        result = create_rollout(request=request, **_common_kwargs(), timeout=60, dry_run=True)
        print(f"  Dry run response: {json.dumps(result, indent=2)[:400]}")
        return {"status": "dry_run", "validation": result}

    print(f"  Creating rollout (harbor, deployment={deployment_id[:12]})...")
    result = create_rollout(request=request, **_common_kwargs(), timeout=60)
    rollout_id = result.get("rollout_id") or result.get("trial_id", "")
    print(f"  Rollout ID: {rollout_id}")
    print(f"  Initial status: {result.get('status')}")

    final = _poll_rollout(rollout_id, args.timeout)
    print(f"  Final: status={final.get('status')}, reward={final.get('reward_primary')}")
    _post_rollout_checks(rollout_id)
    return final


# ---------------------------------------------------------------------------
# 2. OpenEnv rollout (Banking77-style)
# ---------------------------------------------------------------------------
def test_openenv():
    print("-" * 60)
    print("2. OPENENV ROLLOUT (banking77)")
    print("-" * 60)

    request: dict[str, Any] = {
        "task_ref": _default_task_ref("openenv-banking77"),
        "agent": _default_agent(),
        "environment": {"backend": "openenv"},
        "openenv": {
            "reset": {
                "seed": 0,
                "options": {"split": "test"},
            },
            "steps": 1,
        },
        "pool_tags": ["openenv", "banking77"],
        "timeouts": {"agent_sec": args.timeout},
    }
    task_app_url = _task_app_url()
    if task_app_url:
        request["task_app_url"] = task_app_url
    if args.pool_id:
        request["pool_id"] = args.pool_id

    if not _validate_request(request, "openenv"):
        return None

    if args.dry_run:
        result = create_rollout(request=request, **_common_kwargs(), timeout=60, dry_run=True)
        print(f"  Dry run response: {json.dumps(result, indent=2)[:400]}")
        return {"status": "dry_run", "validation": result}

    print("  Creating rollout (openenv/banking77)...")
    result = create_rollout(request=request, **_common_kwargs(), timeout=60)
    rollout_id = result.get("rollout_id") or result.get("trial_id", "")
    print(f"  Rollout ID: {rollout_id}")
    print(f"  Initial status: {result.get('status')}")

    final = _poll_rollout(rollout_id, args.timeout)
    print(f"  Final: status={final.get('status')}, reward={final.get('reward_primary')}")
    _post_rollout_checks(rollout_id)
    return final


# ---------------------------------------------------------------------------
# 3. Archipelago rollout (simple_task / apex)
# ---------------------------------------------------------------------------
def test_archipelago():
    print("-" * 60)
    print("3. ARCHIPELAGO ROLLOUT (apex simple_task)")
    print("-" * 60)

    request: dict[str, Any] = {
        "task_ref": _default_task_ref("archipelago-simple-task"),
        "agent": _default_agent(),
        "environment": {"backend": "archipelago"},
        "archipelago": {
            "env_image": os.environ.get("RHODES_APEX_ENV_IMAGE", ""),
            "agent_image": os.environ.get("RHODES_APEX_AGENT_IMAGE", ""),
            "grading_image": os.environ.get("RHODES_APEX_GRADING_IMAGE", ""),
            "agent_config_path": "/configs/agent_config.json",
            "orchestrator_config_path": "/configs/orchestrator_config.json",
            "verifiers_path": "/configs/verifiers.json",
            "eval_configs_path": "/configs/eval_configs.json",
        },
        "pool_tags": ["archipelago"],
        "timeouts": {"agent_sec": args.timeout, "verifier_sec": 120},
    }
    task_app_url = _task_app_url()
    if task_app_url:
        request["task_app_url"] = task_app_url
    if args.pool_id:
        request["pool_id"] = args.pool_id

    # Check that images are set
    if not request["archipelago"]["env_image"]:
        print("  SKIP archipelago: RHODES_APEX_ENV_IMAGE not set")
        print("  Set RHODES_APEX_ENV_IMAGE, RHODES_APEX_AGENT_IMAGE, RHODES_APEX_GRADING_IMAGE")
        return None

    if not _validate_request(request, "archipelago"):
        return None

    if args.dry_run:
        result = create_rollout(request=request, **_common_kwargs(), timeout=60, dry_run=True)
        print(f"  Dry run response: {json.dumps(result, indent=2)[:400]}")
        return {"status": "dry_run", "validation": result}

    print("  Creating rollout (archipelago)...")
    result = create_rollout(request=request, **_common_kwargs(), timeout=60)
    rollout_id = result.get("rollout_id") or result.get("trial_id", "")
    print(f"  Rollout ID: {rollout_id}")
    print(f"  Initial status: {result.get('status')}")

    final = _poll_rollout(rollout_id, args.timeout)
    print(f"  Final: status={final.get('status')}, reward={final.get('reward_primary')}")
    _post_rollout_checks(rollout_id)
    return final


# ---------------------------------------------------------------------------
# 4. Browser rollout (LinkedIn bench)
# ---------------------------------------------------------------------------
def test_browser():
    print("-" * 60)
    print("4. BROWSER ROLLOUT (linkedin bench)")
    print("-" * 60)

    kernel_api_key = os.environ.get("KERNEL_API_KEY", "")
    if not kernel_api_key:
        print("  SKIP browser: KERNEL_API_KEY not set")
        return None

    request: dict[str, Any] = {
        "task_ref": _default_task_ref("browser-linkedin"),
        "agent": _default_agent(),
        "environment": {"backend": "browser"},
        "browser": {
            "task_prompt": "Go to linkedin.com and tell me the headline of the first post in the feed.",
            "skill": "You are a browser automation agent. Navigate LinkedIn and extract information.",
            "skill_domain": "linkedin.com",
            "profile": "linkedin",
            "timeout_sec": args.timeout,
            "headless": True,
            "capture_screenshot": True,
            "verifier_model": "claude-sonnet-4-20250514",
            "expected": "The headline text of the first LinkedIn feed post",
        },
        "pool_tags": ["browser", "kernel"],
        "timeouts": {"agent_sec": args.timeout},
    }
    task_app_url = _task_app_url()
    if task_app_url:
        request["task_app_url"] = task_app_url
    if args.pool_id:
        request["pool_id"] = args.pool_id

    if not _validate_request(request, "browser"):
        return None

    if args.dry_run:
        result = create_rollout(request=request, **_common_kwargs(), timeout=60, dry_run=True)
        print(f"  Dry run response: {json.dumps(result, indent=2)[:400]}")
        return {"status": "dry_run", "validation": result}

    print("  Creating rollout (browser/linkedin)...")
    result = create_rollout(request=request, **_common_kwargs(), timeout=60)
    rollout_id = result.get("rollout_id") or result.get("trial_id", "")
    print(f"  Rollout ID: {rollout_id}")
    print(f"  Initial status: {result.get('status')}")

    final = _poll_rollout(rollout_id, args.timeout)
    print(f"  Final: status={final.get('status')}, reward={final.get('reward_primary')}")
    _post_rollout_checks(rollout_id)
    return final


# ---------------------------------------------------------------------------
# Bonus: test list/metrics/queue endpoints
# ---------------------------------------------------------------------------
def test_pool_apis():
    print("-" * 60)
    print("POOL / QUEUE API SMOKE TEST")
    print("-" * 60)

    try:
        caps = get_capabilities(**_common_kwargs())
        print(f"  Capabilities: {json.dumps(caps, indent=2)[:400]}")
    except Exception as exc:
        print(f"  Capabilities failed: {exc}")

    try:
        schema = get_openapi_schema(**_common_kwargs())
        paths = schema.get("paths") if isinstance(schema, dict) else None
        print(f"  OpenAPI paths: {len(paths) if isinstance(paths, dict) else 'unknown'}")
    except Exception as exc:
        print(f"  OpenAPI schema failed: {exc}")

    print("  Listing pools...")
    pools = list_pools(**_common_kwargs())
    print(f"  Found {len(pools)} pools")
    for p in pools[:5]:
        pid = p.get("pool_id", p.get("pool", {}).get("pool_id", "?"))
        print(f"    - {pid}")

    if pools:
        first_pool_id = pools[0].get("pool_id", pools[0].get("pool", {}).get("pool_id"))
        if first_pool_id:
            print(f"\n  Getting metrics for pool {first_pool_id}...")
            metrics = get_pool_metrics(first_pool_id, **_common_kwargs())
            print(f"    queue_depth: {metrics.get('queue_depth')}")
            print(f"    running: {metrics.get('running')}")
            print(f"    total_count: {metrics.get('total_count')}")
            print(f"    success_rate: {metrics.get('success_rate')}")

    print("\n  Getting queue status...")
    qs = get_queue_status(**_common_kwargs())
    print(f"    Queue status: {json.dumps(qs, indent=2)[:500]}")

    try:
        rollouts = list_rollouts(**_common_kwargs(), limit=3)
        if isinstance(rollouts, dict):
            sample = rollouts.get("rollouts", rollouts)
        else:
            sample = rollouts
        if isinstance(sample, list):
            print(f"  Recent rollouts: {len(sample)}")
        else:
            print(f"  Recent rollouts: {json.dumps(sample, indent=2)[:300]}")
    except Exception as exc:
        print(f"  List rollouts failed: {exc}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = {}

    # Always test pool/queue APIs first (lightweight)
    try:
        test_pool_apis()
        print()
    except Exception as e:
        print(f"  Pool API test failed: {e}")
        print()

    # Run the requested tests
    tests = {
        "harbor": test_harbor,
        "openenv": test_openenv,
        "archipelago": test_archipelago,
        "browser": test_browser,
    }

    if args.only:
        tests = {args.only: tests[args.only]}

    for name, fn in tests.items():
        try:
            result = fn()
            results[name] = {
                "status": result.get("status") if result else "skipped",
                "rollout_id": (result.get("rollout_id") or result.get("trial_id")) if result else None,
                "reward": result.get("reward_primary") if result else None,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"status": "error", "error": str(e)}
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        status = r.get("status", "?")
        reward = r.get("reward")
        rid = r.get("rollout_id", "")
        err = r.get("error", "")
        line = f"  {name:15s} status={status}"
        if reward is not None:
            line += f"  reward={reward}"
        if rid:
            line += f"  id={rid[:16]}"
        if err:
            line += f"  err={err[:60]}"
        print(line)


if __name__ == "__main__":
    main()
