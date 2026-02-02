"""
End-to-end eval: LinkedIn browser task scored by Synth RLM verifier.

Instead of the task app's built-in LLM judge, this uses the Synth backend's
RLM (Reasoning Language Model) verifier to score the agent trace against
a rubric defined in the task app.

Usage:
    uv run python test_rlm_eval.py
"""

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()


def wait_for_health(port: int, api_key: str = "", timeout: float = 30.0) -> None:
    import httpx

    url = f"http://localhost:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, headers=headers, timeout=5.0)
            if r.status_code in (200, 400):
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed after {timeout}s")


async def main():
    # Check env
    required = ["SYNTH_API_KEY", "KERNEL_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    synth_api_key = os.environ["SYNTH_API_KEY"]
    backend_url = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
    port = 8030

    print("=" * 60)
    print("LINKEDIN EVAL WITH RLM VERIFIER")
    print("=" * 60)

    # 1. Get environment API key
    print("\n1. Authenticating with Synth backend...")
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth

    environment_api_key = ensure_localapi_auth(
        backend_base=backend_url, synth_api_key=synth_api_key
    )
    os.environ["ENVIRONMENT_API_KEY"] = environment_api_key
    print(f"   Environment key: {environment_api_key[:15]}...")

    # 2. Start task app
    print(f"\n2. Starting task app on port {port}...")
    import uvicorn
    from linkedin_bench.task_app import app

    def run_server():
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    wait_for_health(port, environment_api_key)
    print("   Task app ready.")

    # Verify rubrics are advertised
    import httpx

    info_resp = httpx.get(
        f"http://localhost:{port}/info",
        headers={"X-API-Key": environment_api_key},
        timeout=10.0,
    )
    info = info_resp.json()
    has_rubrics = info.get("rubrics") is not None
    print(f"   Rubrics advertised: {has_rubrics}")
    if has_rubrics:
        rubric_data = info["rubrics"]
        if "outcome" in rubric_data and rubric_data["outcome"]:
            criteria = rubric_data["outcome"].get("criteria", [])
            print(f"   Outcome criteria: {[c['id'] for c in criteria]}")

    # 3. Create SynthTunnel
    print("\n3. Creating SynthTunnel...")
    from synth_ai.core.tunnels import TunneledLocalAPI

    tunnel = await TunneledLocalAPI.create(local_port=port, api_key=synth_api_key)
    task_app_url = tunnel.url
    worker_token = tunnel.worker_token
    print(f"   Tunnel URL: {task_app_url}")

    try:
        # 4. Submit eval job with RLM verifier
        seeds = [0, 1, 2]  # All 3 LinkedIn tasks
        concurrency = 3
        print(f"\n4. Submitting eval job with RLM verifier (seeds={seeds}, concurrency={concurrency})...")
        from synth_ai.sdk.eval import EvalJob, EvalJobConfig

        config = EvalJobConfig(
            task_app_url=task_app_url,
            task_app_worker_token=worker_token,
            api_key=synth_api_key,
            backend_url=backend_url,
            env_name="linkedin_bench",
            seeds=seeds,
            policy_config={
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "timeout": 300,
            },
            verifier_config={
                "enabled": True,
                "verifier_graph_id": "zero_shot_verifier_rubric_rlm",
                "reward_source": "verifier",  # Only use RLM score, ignore task app reward
                "backend_base": backend_url,
                "backend_model": "gpt-4o-mini",
                "backend_outcome_enabled": True,
                "backend_event_enabled": False,
                "weight_env": 0.0,       # Ignore task app reward
                "weight_event": 0.0,
                "weight_outcome": 1.0,   # 100% from RLM verifier
                "concurrency": concurrency,
                "timeout": 300.0,
            },
            timeout=1200,
            concurrency=concurrency,
        )

        job = EvalJob(config)
        job_id = job.submit()
        print(f"   Job submitted: {job_id}")

        # 5. Poll for results
        print("\n5. Polling for results (this may take several minutes)...")
        result = job.poll_until_complete(timeout=900.0, interval=10.0, progress=True)

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"   Status: {result.status}")

        if hasattr(result, "failed") and result.failed:
            print(f"   FAILED: {result.error}")
        else:
            if hasattr(result, "best_score") and result.best_score is not None:
                print(f"   Best score (RLM): {result.best_score:.4f}")
            if hasattr(result, "raw") and result.raw:
                print(f"   Raw: {json.dumps(result.raw, indent=2, default=str)[:3000]}")

        # 6. Check traces
        print(f"\n6. Checking traces for job {job_id}...")
        try:
            resp = httpx.get(
                f"{backend_url}/api/eval/jobs/{job_id}/traces",
                headers={"Authorization": f"Bearer {synth_api_key}"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                traces = resp.json()
                print(f"   Traces found: {len(traces) if isinstance(traces, list) else 'response received'}")
            else:
                print(f"   Traces endpoint: {resp.status_code}")
        except Exception as e:
            print(f"   Could not fetch traces: {e}")

        # 7. Check verifier results
        print(f"\n7. Checking verifier results...")
        try:
            resp = httpx.get(
                f"{backend_url}/api/eval/jobs/{job_id}",
                headers={"Authorization": f"Bearer {synth_api_key}"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                job_data = resp.json()
                print(f"   Job data keys: {list(job_data.keys())[:15]}")
                # Look for verifier results in the response
                rollouts = job_data.get("rollouts", [])
                for i, rollout in enumerate(rollouts):
                    print(f"\n   Rollout {i}:")
                    print(f"     Env reward: {rollout.get('outcome_reward', 'N/A')}")
                    verifier = rollout.get("verifier_result") or rollout.get("verifier_outcome")
                    if verifier:
                        print(f"     Verifier result: {json.dumps(verifier, indent=6, default=str)[:1000]}")
                    fused = rollout.get("fused_reward") or rollout.get("final_reward")
                    if fused is not None:
                        print(f"     Fused/final reward: {fused}")
            else:
                print(f"   Job endpoint: {resp.status_code}")
        except Exception as e:
            print(f"   Could not fetch job details: {e}")

        print("\nDone!")

    finally:
        print("\nClosing tunnel...")
        tunnel.close()


if __name__ == "__main__":
    asyncio.run(main())
