#!/usr/bin/env python3
"""
Minimal CLI runner for testing Hosted Task Apps infrastructure.

This task app does NO real work - it just validates the Harbor→CLI→response
pipeline end-to-end. It:
1. Reads /tmp/rollout.json (Harbor file mode input)
2. Parses the HarborRolloutRequest format into an SDK RolloutRequest
3. Returns a success response to /tmp/result.json

Use this to verify the pool→Harbor→Daytona→entrypoint→result pipeline
without needing external dependencies (agents, data repos, etc.).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_hosted_taskapp")


def main():
    input_path = Path("/tmp/rollout.json")
    output_path = Path("/tmp/result.json")
    start = time.time()

    logger.info("=== Test Hosted Task App CLI Runner ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working dir: {os.getcwd()}")

    # 1. Read input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        result = {"error": f"Input file not found: {input_path}", "success": False}
        output_path.write_text(json.dumps(result, indent=2))
        return

    try:
        input_data = json.loads(input_path.read_text())
        logger.info(f"Loaded rollout request from {input_path}")
        logger.info(f"Top-level keys: {sorted(input_data.keys())}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_path}: {e}")
        result = {"error": f"Invalid JSON: {e}", "success": False}
        output_path.write_text(json.dumps(result, indent=2))
        return

    # 2. Parse the request - support both Harbor and direct formats
    trace_id = "unknown"
    seed = 0
    policy_config = {}
    env_config = {}
    format_detected = "unknown"

    try:
        if "run_id" in input_data and "seed" in input_data:
            # Harbor pool rollout format (HarborRolloutRequest)
            format_detected = "harbor"
            trace_id = input_data.get("trace_correlation_id", input_data.get("run_id", "unknown"))
            seed = input_data.get("seed", 0)
            env_config = input_data.get("env_config", {})
            policy_config = input_data.get("policy_config", {})

            task_app_req = input_data.get("task_app_request", {})
            logger.info(f"Harbor format: trace_id={trace_id}, seed={seed}")
            logger.info(f"  env_config keys: {sorted(env_config.keys()) if env_config else '(empty)'}")
            logger.info(f"  policy_config keys: {sorted(policy_config.keys()) if policy_config else '(empty)'}")
            logger.info(f"  task_app_request keys: {sorted(task_app_req.keys()) if task_app_req else '(empty)'}")

        elif "env" in input_data and "policy" in input_data:
            # Direct SDK format
            format_detected = "sdk_direct"
            trace_id = input_data.get("trace_correlation_id", "unknown")
            env_data = input_data.get("env", {})
            policy_data = input_data.get("policy", {})
            seed = env_data.get("seed", 0)
            env_config = env_data.get("config", {})
            policy_config = policy_data.get("config", {})
            logger.info(f"SDK direct format: trace_id={trace_id}, seed={seed}")

        elif "trace_correlation_id" in input_data:
            format_detected = "sdk_full"
            trace_id = input_data["trace_correlation_id"]
            logger.info(f"SDK full format: trace_id={trace_id}")

        else:
            format_detected = "unknown"
            logger.warning(f"Unknown format. Keys: {sorted(input_data.keys())}")

    except Exception as e:
        logger.error(f"Parse error: {e}")
        result = {"error": f"Parse error: {e}", "success": False}
        output_path.write_text(json.dumps(result, indent=2))
        return

    elapsed = time.time() - start

    # 3. Return success response
    result = {
        "success": "SUCCESS",
        "outcome_reward": 1.0,
        "metrics": {
            "outcome_reward": 1.0,
            "details": {
                "format_detected": format_detected,
                "trace_correlation_id": trace_id,
                "seed": seed,
                "policy_config_keys": sorted(policy_config.keys()) if policy_config else [],
                "env_config_keys": sorted(env_config.keys()) if env_config else [],
                "elapsed_s": round(elapsed, 3),
                "message": "Test task app executed successfully. Hosted Task Apps pipeline is working!",
            },
        },
        "artifacts": [],
        "trace_id": trace_id,
    }

    output_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote success result to {output_path} (elapsed: {elapsed:.3f}s)")
    logger.info("=== Test Hosted Task App Complete ===")


if __name__ == "__main__":
    main()
