"""
Fully local test: interceptor trace → RLM verifier scoring.

No external services needed except:
- Local Rust backend on port 8090 (interceptor + graph service)
- Redis on port 6379
- MinIO on port 9000
- Anthropic API key (for the actual LLM call + RLM verifier model)

Flow:
1. Register trial on local interceptor
2. Run Claude Code on a Kernel browser through local interceptor
3. Fetch the captured trace from local interceptor
4. Call local graph service with trace + rubric → RLM score
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = "http://localhost:8090"
INTERCEPTOR_BASE = f"{BACKEND_URL}/api/interceptor/v1"


def register_trial(trial_id: str) -> dict:
    """Register a trial on the local interceptor."""
    resp = httpx.post(
        f"{INTERCEPTOR_BASE}/debug/register_trial/{trial_id}",
        json={
            "job_id": f"local-rlm-test",
            "seed": 0,
            "stage_key": {"pipeline_id": "eval", "stage_id": "passthrough"},
            "baseline_messages": [],
            "deltas": {},
            "ttl_seconds": 3600,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_trace(correlation_id: str) -> dict:
    """Fetch a trace from the local interceptor."""
    resp = httpx.get(
        f"{INTERCEPTOR_BASE}/trace/by-correlation/{correlation_id}",
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def build_rubric() -> dict:
    """Build the LinkedIn task rubric in the format expected by RLM verifier."""
    return {
        "version": "1.0",
        "goal_text": (
            "Evaluate whether the browser automation agent successfully completed "
            "a LinkedIn data-retrieval task. The agent was asked to navigate to a "
            "LinkedIn profile and extract a follower count."
        ),
        "criteria": [
            {
                "id": "task_completion",
                "description": (
                    "Did the agent return the requested numeric value (follower count)? "
                    "A plausible number (~2000 for this profile) counts as success. "
                    "The agent should output a clear answer."
                ),
                "weight": 0.5,
            },
            {
                "id": "navigation_efficiency",
                "description": (
                    "Did the agent navigate to the correct LinkedIn page efficiently, "
                    "without excessive retries, unnecessary page loads, or blind sleep "
                    "commands? Fewer steps and no hardcoded sleeps is better."
                ),
                "weight": 0.3,
            },
            {
                "id": "tool_usage",
                "description": (
                    "Did the agent use browser automation tools appropriately "
                    "(navigate, extract text, screenshot) rather than attempting "
                    "raw HTTP requests, scraping, or other brittle approaches?"
                ),
                "weight": 0.2,
            },
        ],
        "aggregation": "weighted_sum",
    }


async def run_rlm_verifier(trace: dict, rubric: dict, model: str = "gpt-4o-mini") -> dict:
    """Run the RLM verifier graph on the local backend.

    Uses POST /v1/runs to create a graph execution, then polls for result.
    """
    # Build the session trace in the format expected by the verifier
    # The trace from the interceptor has request/response bodies.
    # We need to convert it to a session trace format.
    session_trace = {
        "session_id": trace.get("correlation_id", "unknown"),
        "event_history": [],
        "metadata": {
            "trial_id": trace.get("trial_id", ""),
            "status": trace.get("status", 0),
        },
    }

    # Extract the request messages and response from the interceptor trace
    req_body = trace.get("request", {}).get("body") or {}
    resp_body = trace.get("response", {}).get("body") or {}

    # Add request as user message event
    if req_body.get("messages"):
        for msg in req_body["messages"]:
            session_trace["event_history"].append({
                "type": "message",
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Add response as assistant message event
    if resp_body.get("content"):
        content_parts = resp_body["content"]
        text = ""
        for part in content_parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")
            elif isinstance(part, str):
                text += part
        session_trace["event_history"].append({
            "type": "message",
            "role": "assistant",
            "content": text,
        })
    elif not resp_body:
        # Streaming response — no parsed body, but we have raw_base64
        session_trace["event_history"].append({
            "type": "message",
            "role": "assistant",
            "content": "[Streaming response — raw trace available]",
        })

    # If the trace has raw_base64 (streaming), note it
    raw_b64 = trace.get("response", {}).get("raw_base64")
    if raw_b64:
        session_trace["metadata"]["streaming"] = True
        session_trace["metadata"]["raw_response_length"] = len(raw_b64)

    # Anthropic API key needed for the verifier model
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Create graph run
        run_resp = await client.post(
            f"{BACKEND_URL}/v1/runs",
            json={
                "graph_id": "zero_shot_verifier_rubric_single",  # simpler verifier for local test
                "inputs": {
                    "trace": session_trace,
                    "rubric": rubric,
                    "query": "Evaluate the agent's performance on this LinkedIn data retrieval task.",
                    "options": {
                        "model": model,
                    },
                },
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        run_resp.raise_for_status()
        run_data = run_resp.json()
        run_id = run_data.get("run_id") or run_data.get("id")
        print(f"   Graph run created: {run_id}")

        # Poll for result
        for i in range(60):  # up to 5 minutes
            await asyncio.sleep(5)
            result_resp = await client.get(
                f"{BACKEND_URL}/v1/runs/{run_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            result_data = result_resp.json()
            status = result_data.get("status", "unknown")
            print(f"   [{i*5}s] Run status: {status}")

            if status in ("succeeded", "completed", "failed", "cancelled"):
                # Get full result
                full_resp = await client.get(
                    f"{BACKEND_URL}/v1/runs/{run_id}/result",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if full_resp.status_code == 200:
                    return full_resp.json()
                return result_data

        return {"error": "timeout", "last_status": status}


async def main():
    # Check env
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Verify backend is running
    try:
        health = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        print(f"Backend: {health.json()['status']}")
    except Exception as e:
        print(f"ERROR: Backend not reachable at {BACKEND_URL}: {e}")
        sys.exit(1)

    print("=" * 60)
    print("LOCAL RLM VERIFIER TEST")
    print("=" * 60)

    # Check if we already have a LinkedIn eval trace we can reuse
    existing_traces = []
    try:
        import redis
        r = redis.Redis()
        keys = r.smembers("interceptor:trace-keys")
        eval_keys = [k.decode() for k in keys if b"eval_eval" in k]
        existing_traces = eval_keys
    except Exception:
        pass

    if existing_traces:
        print(f"\n1. Found {len(existing_traces)} existing eval trace(s)")
        correlation_id = existing_traces[-1]  # use most recent
        print(f"   Using: {correlation_id}")

        print("\n2. Fetching trace from local interceptor...")
        trace = fetch_trace(correlation_id)
        print(f"   Trial: {trace['trial_id']}")
        print(f"   Status: {trace['status']}")

        req_body = trace.get("request", {}).get("body") or {}
        resp_body = trace.get("response", {}).get("body") or {}
        raw_b64 = trace.get("response", {}).get("raw_base64")
        print(f"   Request model: {req_body.get('model', 'N/A')}")
        print(f"   Response streaming: {raw_b64 is not None}")
        if resp_body.get("content"):
            first_text = ""
            for part in resp_body["content"]:
                if isinstance(part, dict):
                    first_text = part.get("text", "")[:100]
                    break
            print(f"   Response preview: {first_text}...")
    else:
        # No existing trace — send a fresh request through interceptor
        print("\n1. No existing eval traces. Sending fresh request...")
        trial_id = f"rlm-test-{int(time.time())}"
        correlation_id = f"rlm-corr-{int(time.time())}"

        register_trial(trial_id)
        print(f"   Registered trial: {trial_id}")

        # Send a request that simulates an agent doing a task
        resp = httpx.post(
            f"{INTERCEPTOR_BASE}/{trial_id}/{correlation_id}/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "stream": False,
                "messages": [{
                    "role": "user",
                    "content": (
                        "You are a browser automation agent. You were asked to navigate to "
                        "a LinkedIn profile and get the follower count. After navigating to "
                        "the profile page, you extracted the page text and found: "
                        "'2,002 followers'. Report your finding."
                    ),
                }],
            },
            headers={
                "Content-Type": "application/json",
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        print(f"   Request completed (status {resp.status_code})")

        print("\n2. Fetching trace...")
        trace = fetch_trace(correlation_id)
        print(f"   Trace found: {trace['correlation_id']}")

    # 3. Build rubric
    print("\n3. Building rubric...")
    rubric = build_rubric()
    print(f"   Criteria: {[c['id'] for c in rubric['criteria']]}")

    # 4. Run RLM verifier
    print("\n4. Running RLM verifier on local backend...")
    verifier_model = "gpt-4o-mini"
    print(f"   Graph: zero_shot_verifier_rubric_single")
    print(f"   Model: {verifier_model}")

    result = await run_rlm_verifier(
        trace=trace,
        rubric=rubric,
        model=verifier_model,
    )

    print(f"\n{'=' * 60}")
    print("RLM VERIFIER RESULTS")
    print(f"{'=' * 60}")
    print(json.dumps(result, indent=2, default=str)[:3000])

    # Extract score
    output = result.get("output") or result
    outcome = (output.get("outcome_review") or {}) if isinstance(output, dict) else {}
    if outcome:
        total = outcome.get("total")
        print(f"\n   FINAL SCORE: {total}")
        criteria = outcome.get("criteria", {})
        for cid, cdata in criteria.items():
            if isinstance(cdata, dict):
                print(f"   {cid}: {cdata.get('reward', 'N/A')} - {cdata.get('reason', '')[:80]}")
    else:
        print("\n   Could not extract outcome_review from result")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
