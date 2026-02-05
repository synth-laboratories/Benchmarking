#!/usr/bin/env python3
"""
CLI runner for engine_bench task app.

This script enables the task app to work in Harbor's "file" mode:
1. Reads rollout request from /tmp/rollout.json
2. Calls the rollout handler directly
3. Writes response to /tmp/result.json

This bridges the gap between the FastAPI-based LocalAPI pattern
and Harbor's file-based execution model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Setup logging early
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add /app to path for Daytona sandbox environment
# In sandbox: /app/engine_bench/cli_runner.py needs to import engine_bench.localapi_engine_bench
# Locally: need to import from same directory
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))  # /app in sandbox, or parent dir locally

from synth_ai.sdk.task.contracts import (
    RolloutRequest,
    RolloutEnvSpec,
    RolloutPolicySpec,
)

# Import the run_rollout function - try both import patterns
try:
    # Sandbox path (after Dockerfile sed rewrites imports)
    from engine_bench.localapi_engine_bench import run_rollout
except ImportError:
    # Local dev path
    from localapi_engine_bench import run_rollout


async def main():
    input_path = Path("/tmp/rollout.json")
    output_path = Path("/tmp/result.json")
    
    # Read input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        # Write error result
        result = {
            "error": f"Input file not found: {input_path}",
            "success": False,
        }
        output_path.write_text(json.dumps(result, indent=2))
        return
    
    try:
        input_data = json.loads(input_path.read_text())
        logger.info(f"Loaded rollout request from {input_path}")
        logger.info(f"Request keys: {list(input_data.keys())}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_path}: {e}")
        result = {"error": f"Invalid JSON: {e}", "success": False}
        output_path.write_text(json.dumps(result, indent=2))
        return
    
    # Parse the rollout request
    # Harbor pool rollout sends input in HarborRolloutRequest format:
    # {
    #   "run_id": "...",
    #   "trace_correlation_id": "...",
    #   "seed": 0,
    #   "env_name": "...",
    #   "env_config": {...},
    #   "policy_config": {...},
    #   "task_app_request": { ... }  // optional nested request
    # }
    #
    # The SDK's RolloutRequest requires:
    # - trace_correlation_id: str
    # - env: RolloutEnvSpec (with seed, config)
    # - policy: RolloutPolicySpec (with config)
    try:
        # Check for Harbor pool rollout format (has run_id and seed at top level)
        if "run_id" in input_data and "seed" in input_data:
            logger.info("Detected Harbor pool rollout format")
            
            # Extract fields from Harbor format
            trace_id = input_data.get("trace_correlation_id", input_data.get("run_id", "unknown"))
            seed = input_data.get("seed", 0)
            env_config = input_data.get("env_config", {})
            policy_config = input_data.get("policy_config", {})
            
            # Try to get context_overrides from task_app_request if present
            task_app_req = input_data.get("task_app_request", {})
            context_overrides = task_app_req.get("context_overrides")
            override_bundle_id = task_app_req.get("override_bundle_id")
            
            # Construct the SDK RolloutRequest properly
            request = RolloutRequest(
                trace_correlation_id=trace_id,
                env=RolloutEnvSpec(
                    seed=seed,
                    config=env_config,
                ),
                policy=RolloutPolicySpec(
                    config=policy_config,
                ),
                context_overrides=context_overrides,
                override_bundle_id=override_bundle_id,
            )
            logger.info(f"Constructed RolloutRequest: trace_id={trace_id}, seed={seed}")
            
        # Check for direct API format (has env and policy at top level as dicts/objects)
        elif "env" in input_data and "policy" in input_data:
            logger.info("Detected direct API format with env/policy")
            
            trace_id = input_data.get("trace_correlation_id", "unknown")
            env_data = input_data.get("env", {})
            policy_data = input_data.get("policy", {})
            
            # env_data might already be a RolloutEnvSpec dict
            if isinstance(env_data, dict):
                env_spec = RolloutEnvSpec(
                    seed=env_data.get("seed", 0),
                    config=env_data.get("config", {}),
                    env_id=env_data.get("env_id"),
                    env_name=env_data.get("env_name"),
                )
            else:
                env_spec = env_data
                
            if isinstance(policy_data, dict):
                policy_spec = RolloutPolicySpec(
                    config=policy_data.get("config", {}),
                    policy_id=policy_data.get("policy_id"),
                    policy_name=policy_data.get("policy_name"),
                )
            else:
                policy_spec = policy_data
            
            request = RolloutRequest(
                trace_correlation_id=trace_id,
                env=env_spec,
                policy=policy_spec,
                context_overrides=input_data.get("context_overrides"),
                override_bundle_id=input_data.get("override_bundle_id"),
            )
            logger.info(f"Constructed RolloutRequest from direct format: trace_id={trace_id}")
            
        # Check for trace_correlation_id (might be the SDK format already)
        elif "trace_correlation_id" in input_data:
            logger.info("Attempting direct RolloutRequest construction (SDK format)")
            request = RolloutRequest(**input_data)
            logger.info(f"Parsed RolloutRequest: trace_id={request.trace_correlation_id}")
        else:
            # Unknown format - log and fail
            logger.error(f"Unknown request format. Keys: {list(input_data.keys())}")
            result = {"error": f"Unknown request format. Expected Harbor or SDK format. Got keys: {list(input_data.keys())}", "success": False}
            output_path.write_text(json.dumps(result, indent=2))
            return
            
    except Exception as e:
        logger.error(f"Failed to parse RolloutRequest: {e}")
        logger.error(f"Input data keys: {list(input_data.keys())}")
        logger.error(f"Input data sample: {str(input_data)[:500]}")
        result = {"error": f"Invalid request format: {e}", "success": False}
        output_path.write_text(json.dumps(result, indent=2))
        return
    
    # Create a mock FastAPI request (the run_rollout function expects it for headers)
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.state = MagicMock()
    
    # Run the rollout
    try:
        logger.info("Starting rollout...")
        response = await run_rollout(request, mock_request)
        logger.info(f"Rollout completed: success={response.success_status}")
        
        # Convert response to dict
        result = {
            "success": response.success_status.value if hasattr(response.success_status, 'value') else str(response.success_status),
            "outcome_reward": response.outcome_reward,
            "metrics": response.metrics.model_dump() if response.metrics else None,
            "artifacts": [a.model_dump() for a in response.artifacts] if response.artifacts else [],
            "trace_id": response.trace_id,
        }
        
        logger.info(f"Result: outcome_reward={response.outcome_reward}")
        
    except Exception as e:
        logger.exception(f"Rollout failed: {e}")
        result = {
            "error": str(e),
            "success": "FAILURE",
            "outcome_reward": 0.0,
        }
    
    # Write output
    output_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote result to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
