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

from synth_ai.sdk.task.contracts import RolloutRequest

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
    try:
        # The input may be a direct RolloutRequest or wrapped in env/policy from Harbor
        if "env" in input_data and "policy" in input_data:
            # Harbor format: extract and build RolloutRequest
            env = input_data.get("env", {})
            policy = input_data.get("policy", {})
            
            request = RolloutRequest(
                instance_id=str(env.get("seed", 0)),
                policy_config=policy,
                context_override=input_data.get("context_override"),
            )
        else:
            # Direct RolloutRequest format
            request = RolloutRequest(**input_data)
        
        logger.info(f"Parsed RolloutRequest: instance_id={request.instance_id}")
    except Exception as e:
        logger.error(f"Failed to parse RolloutRequest: {e}")
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
