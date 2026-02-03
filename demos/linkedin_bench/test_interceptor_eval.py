"""
End-to-end test: submit a single-seed eval job through the Synth backend.

This tests the full interceptor flow:
1. Start task app locally
2. Create SynthTunnel to expose it
3. Submit eval job (backend assigns trial_id, routes through interceptor)
4. Poll for results
5. Verify traces were captured

Usage:
    uv run python test_interceptor_eval.py
"""

import asyncio
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
    backend_url = os.environ.get("SYNTH_BACKEND_URL", "http://127.0.0.1:8080")
    port = 8030

    print("=" * 60)
    print("INTERCEPTOR EVAL TEST")
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

    # 3. Create SynthTunnel
    print("\n3. Creating SynthTunnel...")
    from synth_ai.core.tunnels import TunneledLocalAPI

    tunnel = await TunneledLocalAPI.create(local_port=port, api_key=synth_api_key)
    task_app_url = tunnel.url
    worker_token = tunnel.worker_token
    print(f"   Tunnel URL: {task_app_url}")
    print(f"   Worker token: {worker_token[:20]}...")

    try:
        # 4. Submit eval job (force save_trace so we can verify trace hydration)
        print("\n4. Submitting eval job (1 seed)...")
        import httpx
        from synth_ai.sdk.eval import EvalJob

        api_base = backend_url.rstrip("/")
        if not api_base.endswith("/api"):
            api_base = f"{api_base}/api"

        payload = {
            "task_app_url": task_app_url,
            "task_app_worker_token": worker_token,
            "env_name": "linkedin_bench",
            "seeds": [0],
            "policy": {
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "timeout": 300,
            },
            "timeout": 600,
            "max_concurrent": 1,
            "save_trace": True,
        }

        headers = {
            "Authorization": f"Bearer {synth_api_key}",
            "X-SynthTunnel-Worker-Token": worker_token,
        }
        resp = httpx.post(
            f"{api_base}/eval/jobs",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print(f"   Job submitted: {job_id}")

        # 5. Poll for results
        print("\n5. Polling for results...")
        job = EvalJob.from_job_id(job_id, backend_url=backend_url, api_key=synth_api_key)
        result = job.poll_until_complete(timeout=600.0, interval=10.0, progress=True)

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"   Status: {result.status}")

        if hasattr(result, "failed") and result.failed:
            print(f"   FAILED: {result.error}")
        else:
            if hasattr(result, "best_score") and result.best_score is not None:
                print(f"   Best score: {result.best_score:.4f}")
            if hasattr(result, "raw") and result.raw:
                import json

                print(f"   Raw: {json.dumps(result.raw, indent=2, default=str)[:2000]}")

        # 6. Check traces
        print(f"\n6. Checking traces for job {job_id}...")
        try:
            import httpx

            resp = httpx.get(
                f"{api_base}/eval/jobs/{job_id}/traces/list",
                headers={"Authorization": f"Bearer {synth_api_key}"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                traces = resp.json()
                print(f"   Trace rows: {len(traces) if isinstance(traces, list) else 'response received'}")
                if isinstance(traces, list) and traces:
                    print(f"   First trace row: {traces[0]}")
            else:
                print(f"   Traces list endpoint: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"   Could not fetch traces: {e}")

        print("\nDone!")

    finally:
        print("\nClosing tunnel...")
        tunnel.close()


if __name__ == "__main__":
    asyncio.run(main())
