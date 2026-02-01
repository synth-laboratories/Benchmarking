#!/usr/bin/env python3
"""Check status of a GEPA boosting job.

Usage:
  uv run python demos/boosting/check_status.py <job_id>
  uv run python demos/boosting/check_status.py <job_id> --backend-url https://api-dev.usesynth.ai
"""

import argparse
import os
import sys

from synth_ai.core.utils.env import mint_demo_api_key
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob

parser = argparse.ArgumentParser(description="Check GEPA job status")
parser.add_argument("job_id", help="Job ID (e.g., pl_9c58b711c2644083)")
parser.add_argument(
    "--backend-url",
    type=str,
    default=os.environ.get("SYNTH_BACKEND_URL", "https://api-dev.usesynth.ai"),
    help="Backend URL",
)
args = parser.parse_args()

backend_url = args.backend_url
api_key = os.environ.get("SYNTH_API_KEY", "")
if not api_key:
    print("No SYNTH_API_KEY found, minting demo key...")
    api_key = mint_demo_api_key(backend_url=backend_url)
else:
    print(f"Using SYNTH_API_KEY: {api_key[:20]}...")

print(f"Checking status for job: {args.job_id}")
print(f"Backend: {backend_url}")

try:
    job = PromptLearningJob.from_job_id(
        job_id=args.job_id,
        backend_url=backend_url,
        api_key=api_key,
    )
    
    status = job.get_status()
    
    print("\n=== Job Status ===")
    print(f"Status: {status.get('status', 'unknown')}")
    
    best_score = status.get("best_score") or status.get("best_reward")
    if best_score is not None:
        print(f"Best Score: {best_score:.4f}")
    
    iteration = status.get("iteration") or status.get("current_iteration")
    if iteration is not None:
        print(f"Iteration: {iteration}")
    
    generation = status.get("generation")
    if generation is not None:
        print(f"Generation: {generation}")
    
    # Check for errors
    error = status.get("error") or status.get("error_message")
    if error:
        print(f"\nError: {error}")
    
    # Show progress info
    progress = status.get("progress")
    if progress:
        print(f"\nProgress: {progress}")
    
    # Show all other fields
    print("\n=== Full Status ===")
    for key, value in sorted(status.items()):
        if key not in ["status", "best_score", "best_reward", "iteration", "current_iteration", "generation", "error", "error_message", "progress"]:
            print(f"{key}: {value}")
            
except Exception as e:
    print(f"Error checking status: {e}", file=sys.stderr)
    sys.exit(1)
