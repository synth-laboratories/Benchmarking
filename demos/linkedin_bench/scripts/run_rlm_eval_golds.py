"""
Run Synth eval job using RLM verifier on tasks with gold answers.

Usage:
  uv run python scripts/run_rlm_eval_golds.py
  uv run python scripts/run_rlm_eval_golds.py --model claude-haiku-4-5-20251001 --concurrency 5
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    parser = argparse.ArgumentParser(description="Run RLM eval on gold tasks")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model for policy")
    parser.add_argument("--concurrency", type=int, default=5, help="Eval concurrency")
    parser.add_argument("--timeout", type=int, default=7200, help="Overall job timeout (seconds)")
    parser.add_argument("--task-timeout", type=int, default=600, help="Per-task timeout (seconds)")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Limit eval to the first N gold seeds (useful for smoke tests).",
    )
    parser.add_argument(
        "--tunnel-backend",
        default=None,
        help=(
            "Tunnel backend: synthtunnel, cloudflare_managed_lease, cloudflare_quick, localhost. "
            "If unset, uses synthtunnel unless SYNTH_BACKEND_URL is localhost."
        ),
    )

    args = parser.parse_args()

    required = ["SYNTH_API_KEY", "KERNEL_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    synth_api_key = os.environ["SYNTH_API_KEY"]
    backend_url = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
    # When the backend itself is exposed via SynthTunnel, it requires a worker token
    # for auth. Allow overriding the backend API key separately so we can still use
    # the real SYNTH_API_KEY for tunnel creation + localapi auth.
    backend_api_key = (
        os.environ.get("SYNTH_BACKEND_API_KEY")
        or os.environ.get("SYNTH_BACKEND_WORKER_TOKEN")
        or synth_api_key
    )
    # Tunnel provisioning should hit the real Synth backend, not a tunneled backend.
    tunnel_backend_url = os.environ.get("SYNTH_TUNNEL_BACKEND_URL")
    port = 8030

    # Load gold task IDs and map to seeds
    gold_path = Path(__file__).parent.parent / "data" / "gold_answers.json"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold answers not found: {gold_path}")

    gold_answers = json.loads(gold_path.read_text())
    gold_task_ids = set(gold_answers.keys())

    from linkedin_bench.tasks import TASKS

    seeds = [i for i, task in enumerate(TASKS) if task.id in gold_task_ids]
    if args.seed_count is not None:
        seeds = seeds[: max(0, args.seed_count)]

    print("=" * 60)
    print("LINKEDIN EVAL WITH RLM VERIFIER (GOLD TASKS)")
    print("=" * 60)
    print(f"Gold tasks: {len(seeds)}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")

    # 1. Get environment API key
    print("\n1. Authenticating with Synth backend...")
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth

    # Use backend_api_key for local backend auth (dev stack), otherwise use Synth API key.
    backend_url_lower = (backend_url or "").lower()
    auth_key = (
        backend_api_key
        if backend_url_lower.startswith("http://localhost")
        or backend_url_lower.startswith("http://127.0.0.1")
        else synth_api_key
    )
    environment_api_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=auth_key)
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

    # 3. Create tunnel (or use localhost)
    from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

    backend_url_lower = (backend_url or "").lower()
    if args.tunnel_backend:
        tunnel_backend = TunnelBackend(args.tunnel_backend)
    elif backend_url_lower.startswith("http://localhost") or backend_url_lower.startswith("http://127.0.0.1"):
        tunnel_backend = TunnelBackend.Localhost
    else:
        tunnel_backend = TunnelBackend.SynthTunnel

    if tunnel_backend == TunnelBackend.Localhost:
        print("\n3. Using localhost task app (no tunnel)...")
        tunnel = None
        task_app_url = f"http://localhost:{port}"
        worker_token = None
    else:
        print(f"\n3. Creating tunnel ({tunnel_backend.value})...")
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=tunnel_backend,
            api_key=synth_api_key,
            backend_url=tunnel_backend_url,
        )
        task_app_url = tunnel.url
        worker_token = tunnel.worker_token
        print(f"   Tunnel URL: {task_app_url}")

    try:
        # 4. Submit eval job with RLM verifier
        print(f"\n4. Submitting eval job with RLM verifier (seeds={len(seeds)}, concurrency={args.concurrency})...")
        from synth_ai.sdk.eval import EvalJob, EvalJobConfig

        task_app_api_key = os.environ.get("ENVIRONMENT_API_KEY")
        config = EvalJobConfig(
            task_app_url=task_app_url,
            task_app_worker_token=worker_token,
            task_app_api_key=task_app_api_key,
            api_key=backend_api_key,
            backend_url=backend_url,
            env_name="linkedin_bench",
            seeds=seeds,
            policy_config={
                "model": args.model,
                "provider": "anthropic",
                "timeout": args.task_timeout,
            },
            verifier_config={
                "enabled": True,
                "verifier_graph_id": "zero_shot_verifier_rubric_rlm",
                "reward_source": "verifier",  # Only use RLM score
                "backend_base": backend_url,
                "backend_model": "gpt-4o-mini",
                "backend_outcome_enabled": True,
                "backend_event_enabled": False,
                "weight_env": 0.0,
                "weight_event": 0.0,
                "weight_outcome": 1.0,
                "concurrency": args.concurrency,
                "timeout": float(args.task_timeout),
            },
            timeout=args.timeout,
            concurrency=args.concurrency,
        )

        job = EvalJob(config)
        job_id = job.submit()
        print(f"   Job submitted: {job_id}")

        # 5. Poll for results
        print("\n5. Polling for results (this may take a while)...")
        result = job.poll_until_complete(timeout=float(args.timeout), interval=10.0, progress=True)

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
        print("\n7. Checking verifier results...")
        try:
            resp = httpx.get(
                f"{backend_url}/api/eval/jobs/{job_id}",
                headers={"Authorization": f"Bearer {synth_api_key}"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                job_data = resp.json()
                print(f"   Job data keys: {list(job_data.keys())[:15]}")
                rollouts = job_data.get("rollouts", [])
                print(f"   Rollouts: {len(rollouts)}")
            else:
                print(f"   Job endpoint: {resp.status_code}")
        except Exception as e:
            print(f"   Could not fetch job details: {e}")

        print("\nDone!")

    finally:
        if tunnel is not None:
            print("\nClosing tunnel...")
            tunnel.close()


if __name__ == "__main__":
    asyncio.run(main())
