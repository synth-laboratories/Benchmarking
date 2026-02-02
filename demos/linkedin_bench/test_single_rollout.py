"""
Test script for a single rollout through the LinkedIn GEPA task app.

Uses SynthTunnel for auth and calls the rollout function in-process,
verifying that the response includes trace_correlation_id, structured
artifacts, and proper reward_info.

Usage:
    # Without interceptor (direct Anthropic API):
    uv run python test_single_rollout.py

    # With Synth interceptor:
    uv run python test_single_rollout.py \
        --inference-url "https://interceptor.synth.ai/v1/messages?cid=test-123"
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()


async def run_test(inference_url: str | None = None):
    """Call run_rollout directly in-process and validate the response."""
    from synth_ai.sdk.localapi._impl.contracts import (
        RolloutEnvSpec,
        RolloutPolicySpec,
        RolloutRequest,
    )

    from linkedin_bench.task_app import run_rollout

    # Build a RolloutRequest
    trace_correlation_id = str(uuid.uuid4())
    policy_config = {"timeout": 300}
    if inference_url:
        policy_config["inference_url"] = inference_url

    request = RolloutRequest(
        trace_correlation_id=trace_correlation_id,
        env=RolloutEnvSpec(seed=0),
        policy=RolloutPolicySpec(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            config=policy_config,
        ),
    )

    print(f"{'=' * 60}")
    print(f"SINGLE ROLLOUT TEST")
    print(f"{'=' * 60}")
    print(f"  trace_correlation_id: {trace_correlation_id}")
    print(f"  inference_url: {inference_url or '(direct API)'}")
    print(f"  seed: 0 (task: commenters_stripe_ai_post)")
    print()

    # Call run_rollout directly (no HTTP, no auth needed)
    response = await run_rollout(request, fastapi_request=None)

    # Serialize for inspection
    data = response.model_dump() if hasattr(response, "model_dump") else response.dict()

    print()
    print(f"{'=' * 60}")
    print(f"RESPONSE VALIDATION")
    print(f"{'=' * 60}")

    errors = []

    # Check trace_correlation_id
    resp_cid = data.get("trace_correlation_id")
    if resp_cid == trace_correlation_id:
        print(f"[PASS] trace_correlation_id echoed: {resp_cid}")
    elif resp_cid:
        print(f"[WARN] trace_correlation_id mismatch: sent={trace_correlation_id}, got={resp_cid}")
    else:
        print("[FAIL] trace_correlation_id missing from response")
        errors.append("missing trace_correlation_id")

    # Check reward_info
    reward_info = data.get("reward_info") or data.get("metrics")
    if reward_info:
        outcome = reward_info.get("outcome_reward")
        print(f"[PASS] outcome_reward: {outcome}")
        details = reward_info.get("details", {})
        if details:
            print(f"       correct: {details.get('correct')}")
            print(f"       extracted_answer: {details.get('extracted_answer')}")
            print(f"       elapsed_seconds: {details.get('elapsed_seconds')}")
    else:
        print("[FAIL] reward_info missing")
        errors.append("missing reward_info")

    # Check artifact
    artifacts = data.get("artifact")
    if artifacts and len(artifacts) > 0:
        art = artifacts[0]
        print(f"[PASS] artifact present: content_type={art.get('content_type')}")
        print(f"       content preview: {str(art.get('content', ''))[:100]}")
        meta = art.get("metadata", {})
        print(f"       metadata: {meta}")
    else:
        print("[FAIL] artifact missing or empty")
        errors.append("missing artifact")

    # Check inference_url echo
    resp_url = data.get("inference_url")
    if inference_url:
        if resp_url:
            print(f"[PASS] inference_url echoed: {resp_url}")
        else:
            print("[WARN] inference_url not echoed in response")
    else:
        print(f"[INFO] inference_url in response: {resp_url or '(none)'}")

    # Check success status
    status = data.get("success_status")
    detail = data.get("status_detail")
    print(f"       success_status: {status}")
    print(f"       status_detail: {detail}")

    print()
    if errors:
        print(f"RESULT: FAIL ({len(errors)} errors: {', '.join(errors)})")
        return False
    else:
        print("RESULT: PASS - all required fields present")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test single rollout")
    parser.add_argument(
        "--inference-url", default=None, help="Synth interceptor inference URL"
    )
    args = parser.parse_args()

    success = asyncio.run(run_test(inference_url=args.inference_url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
