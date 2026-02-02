#!/usr/bin/env python3
"""
Run GEPA optimization on LinkedIn browser automation skill.

This script:
1. Starts the task app locally
2. Creates a SynthTunnel to expose it to Synth's infrastructure
3. Submits a GEPA job to the Synth backend
4. Polls for results
5. Saves the optimized skill

Usage:
    # Run with SynthTunnel (default - currently broken, see problems/synthtunnel.md)
    uv run python run_gepa.py

    # Workaround: Use ngrok instead
    ngrok http 8030 --url your-subdomain.ngrok-free.app
    TASK_APP_URL=https://your-subdomain.ngrok-free.app uv run python run_gepa.py

    # Override generations and budget
    uv run python run_gepa.py --generations 5 --budget 50

    # Use local Synth backend (no tunnel needed)
    uv run python run_gepa.py --local
"""

import argparse
import asyncio
import json
import os
import sys
import time
import tomllib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from linkedin_bench.skill_template import get_skill_content

load_dotenv()


def wait_for_health(host: str, port: int, api_key: str = "", timeout: float = 30.0) -> None:
    """Wait for task app to be healthy."""
    import httpx

    url = f"http://{host}:{port}/health"
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
    raise RuntimeError(f"Health check failed: {url}")


async def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization for LinkedIn skill")
    parser.add_argument("--local", action="store_true", help="Use local Synth backend")
    parser.add_argument("--local-port", type=int, default=8080, help="Local backend port (default: 8080)")
    parser.add_argument("--port", type=int, default=8030, help="Task app port (default: 8030)")
    parser.add_argument("--config", type=str, default="linkedin_gepa.toml", help="GEPA config file")
    parser.add_argument("--generations", type=int, help="Override number of generations")
    parser.add_argument("--budget", type=int, help="Override rollout budget")
    parser.add_argument("--timeout", type=int, default=120, help="Agent timeout per rollout")
    parser.add_argument("--backend-url", type=str, help="Override backend URL (default: prod)")
    args = parser.parse_args()

    # Check environment
    required_vars = ["SYNTH_API_KEY", "KERNEL_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    synth_api_key = os.environ["SYNTH_API_KEY"]
    if args.backend_url:
        backend_url = args.backend_url.rstrip("/")
    elif args.local:
        backend_url = f"http://localhost:{args.local_port}"
    else:
        backend_url = "https://api.usesynth.ai"

    print("=" * 60)
    print("GEPA OPTIMIZATION - LinkedIn Corporate Monitoring Skill")
    print("=" * 60)
    print(f"Backend: {backend_url}")

    # Resolve environment API key for task app auth
    env_override = os.environ.get("ENVIRONMENT_API_KEY") or os.environ.get("DEV_ENVIRONMENT_API_KEY")
    if env_override:
        environment_api_key = env_override
        print(f"Using ENVIRONMENT_API_KEY from env: {environment_api_key[:15]}...")
    else:
        print("Authenticating with Synth backend...")
        try:
            from synth_ai.sdk.localapi.auth import ensure_localapi_auth
            environment_api_key = ensure_localapi_auth(
                backend_base=backend_url,
                synth_api_key=synth_api_key,
            )
            print(f"Environment key ready: {environment_api_key[:15]}...")
            # Ensure the task app accepts the same environment key that the backend will use.
            os.environ["ENVIRONMENT_API_KEY"] = environment_api_key
            os.environ.setdefault("DEV_ENVIRONMENT_API_KEY", environment_api_key)
        except Exception as e:
            print(f"Warning: Could not get environment API key: {e}")
            environment_api_key = synth_api_key
    print(f"Config: {args.config}")

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Apply overrides
    if args.generations:
        config["prompt_learning"]["gepa"]["population"]["num_generations"] = args.generations
    if args.budget:
        config["prompt_learning"]["gepa"]["rollout"]["budget"] = args.budget


    # Inject the initial skill content into the wildcard
    # GEPA needs to know the initial value to mutate from
    initial_skill = get_skill_content()
    config["prompt_learning"]["initial_prompt"]["wildcards"]["skill_content"] = initial_skill
    print(f"Initial skill content: {len(initial_skill)} chars")

    # Also inject initial skill content into the baseline_context_override so
    # the Rust backend can populate context_overrides with the real initial value
    # (rather than the placeholder "REQUIRED").
    gepa_cfg = config.get("prompt_learning", {}).get("gepa", {})
    baseline_co = gepa_cfg.get("baseline_context_override", {})
    file_artifacts = baseline_co.get("file_artifacts", {})
    for path in list(file_artifacts.keys()):
        if "skill" in path.lower():
            file_artifacts[path] = initial_skill

    print(f"Generations: {config['prompt_learning']['gepa']['population']['num_generations']}")
    print(f"Budget: {config['prompt_learning']['gepa']['rollout']['budget']}")
    print(f"Timeout: {args.timeout}s")

    # Start task app in background
    print("\nStarting task app...")
    import threading
    import uvicorn

    # Import the app from our module
    from linkedin_bench.task_app import app

    port = args.port

    def run_server():
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for task app to be ready
    wait_for_health("localhost", port, environment_api_key)
    print(f"Task app ready on port {port}")

    # Set up tunnel or use override URL
    tunnel = None
    worker_token = None
    task_app_url_override = os.environ.get("TASK_APP_URL")

    if task_app_url_override:
        # Manual override (e.g., ngrok)
        task_app_url = task_app_url_override
        print(f"Task app URL (from env): {task_app_url}")
    else:
        # Use SynthTunnel to expose local task app to the backend.
        # For --local, the tunnel connects to the local relay; otherwise production.
        print("\nCreating SynthTunnel...")
        from synth_ai.core.tunnels import TunneledLocalAPI

        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            api_key=synth_api_key,
            backend_url=backend_url,
            env_api_key=environment_api_key,
        )
        task_app_url = tunnel.url
        worker_token = tunnel.worker_token
        print(f"Tunnel URL: {task_app_url}")
        print(f"Worker token: {worker_token[:20]}...")

    config["prompt_learning"]["task_app_url"] = task_app_url

    try:
        # Submit GEPA job
        print("\nSubmitting GEPA job...")

        # Import Synth SDK
        try:
            from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
        except ImportError:
            print("ERROR: synth-ai SDK not installed. Install with: pip install synth-ai")
            sys.exit(1)

        # Always pass task_app_api_key so the backend has it for X-API-Key auth.
        # For SynthTunnel, also pass worker_token for relay auth.
        # The backend's inject_task_app_api_key handles secure credential
        # injection at job start time for both tunnel modes.
        job_kwargs = {
            "config_dict": config,
            "backend_url": backend_url,
            "api_key": synth_api_key,
            "skip_health_check": True,
            "task_app_api_key": environment_api_key,
        }
        if worker_token:
            job_kwargs["task_app_worker_token"] = worker_token

        job = PromptLearningJob.from_dict(**job_kwargs)

        job_id = job.submit()
        print(f"Job submitted: {job_id}")

        # Poll for results with progress updates
        print("\nPolling for results (this may take a while)...")
        result = job.poll_until_complete(
            timeout=7200.0,  # 2 hours max
            interval=15.0,
            progress=True,
        )

        print("\n" + "=" * 60)
        print("GEPA RESULTS")
        print("=" * 60)
        print(f"Status: {result.status}")

        if result.failed:
            print(f"Job failed: {result.error}")
            if result.raw:
                print(f"Raw response: {json.dumps(result.raw, indent=2)[:2000]}")
        else:
            # Try common attribute names for the best score
            best_score = getattr(result, 'best_score', None) or getattr(result, 'best_reward', None)
            if best_score is not None:
                print(f"Best score: {best_score:.4f}")

            # Save the optimized skill
            if result.best_prompt:
                output_dir = Path(__file__).parent / "output"
                output_dir.mkdir(exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"optimized_skill_{timestamp}.md"

                if isinstance(result.best_prompt, dict):
                    # Extract skill content from messages
                    messages = result.best_prompt.get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "system":
                            skill_content = msg.get("pattern", "")
                            output_file.write_text(skill_content)
                            print(f"\nOptimized skill saved to: {output_file}")
                            print("\nFirst 500 chars of optimized skill:")
                            print("-" * 40)
                            print(skill_content[:500])
                            break
                elif isinstance(result.best_prompt, str):
                    output_file.write_text(result.best_prompt)
                    print(f"\nOptimized skill saved to: {output_file}")

            # Print training stats
            if result.raw:
                best_train = result.raw.get("best_train_score")
                best_val = result.raw.get("best_validation_score")
                if best_train is not None:
                    print(f"\nBest train score: {best_train:.4f}")
                if best_val is not None:
                    print(f"Best validation score: {best_val:.4f}")

    finally:
        # Cleanup tunnel if created
        if tunnel:
            print("\nClosing SynthTunnel...")
            tunnel.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
