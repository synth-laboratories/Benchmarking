#!/usr/bin/env python3
"""
Run Banking77 MIPRO Comparison using Synth SDK

This script runs MIPRO optimization on Banking77 using the synth-ai SDK directly.
Based on demos/mipro_banking77/run_demo.py

Usage:
    python run_synth_mipro_banking77.py
    python run_synth_mipro_banking77.py --local  # Use localhost backend
    python run_synth_mipro_banking77.py --rollouts 100 --model gpt-4.1-nano
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Add synth-ai to path if running from monorepo
synth_ai_path = Path(__file__).resolve().parents[4] / "synth-ai"
if synth_ai_path.exists() and str(synth_ai_path) not in sys.path:
    sys.path.insert(0, str(synth_ai_path))

try:
    from datasets import load_dataset
    from fastapi import Request
    from openai import AsyncOpenAI
    from synth_ai.core.utils.env import mint_demo_api_key
    from synth_ai.data.enums import SuccessStatus
    from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth
    from synth_ai.sdk.localapi._impl.server import run_server_background
    from synth_ai.sdk.localapi._impl.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
    from synth_ai.sdk.localapi._impl.trace_correlation_helpers import extract_trace_correlation_id
    from synth_ai.core.tunnels import (
        PortConflictBehavior,
        TunnelBackend,
        TunneledLocalAPI,
        acquire_port,
        cleanup_all,
    )
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install with: pip install synth-ai datasets")
    sys.exit(1)


# Parse arguments
parser = argparse.ArgumentParser(description="Run Banking77 MIPRO using Synth SDK")
parser.add_argument("--local", action="store_true", help="Use localhost backend (no tunnels)")
parser.add_argument("--local-host", type=str, default="localhost", help="Local API hostname")
parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model to use")
parser.add_argument("--rollouts", type=int, default=50, help="Rollout budget for bootstrap")
parser.add_argument("--train-size", type=int, default=100, help="Training seeds count")
parser.add_argument("--val-size", type=int, default=50, help="Validation seeds count")
parser.add_argument("--mode", type=str, default="offline", choices=["offline", "online"], help="MIPRO mode")
parser.add_argument("--output", type=str, default="banking77_synth_mipro_results.json", help="Output file")
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host

# Backend configuration
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    TUNNEL_BACKEND = TunnelBackend.Localhost
    LOCAL_API_PORT = 8015
    print("="*60)
    print("RUNNING IN LOCAL MODE")
    print("="*60)
else:
    SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api-dev.usesynth.ai")
    TUNNEL_BACKEND = TunnelBackend.CloudflareManagedTunnel
    LOCAL_API_PORT = 8015

print(f"Backend: {SYNTH_API_BASE}")
print(f"Tunnel backend: {TUNNEL_BACKEND.value}")
print(f"Model: {args.model}")
print(f"Rollouts: {args.rollouts}")
print(f"Mode: {args.mode}")

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code != 200:
    print(f"ERROR: Backend not healthy (status {r.status_code})")
    sys.exit(1)
print(f"Backend health: {r.json()}")

# Get API Key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY found, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

# Ensure Environment Key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


# Banking77 configuration
APP_ID = "banking77"
APP_NAME = "Banking77 Intent Classification"

BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

BASELINE_SYSTEM_PROMPT = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."

TOOL_NAME = "banking77_classify"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the predicted banking77 intent label.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}


def format_available_intents(label_names: list) -> str:
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


async def classify_banking77_query(
    query: str,
    system_prompt: str,
    available_intents: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    inference_url: str | None = None,
) -> str:
    user_msg = (
        f"Customer Query: {query}\n\n"
        f"Available Intents:\n{available_intents}\n\n"
        f"Classify this query into one of the above banking intents using the tool call."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    if inference_url:
        default_headers = {"X-API-Key": api_key} if api_key else {}
        client = AsyncOpenAI(
            base_url=inference_url,
            api_key="synth-interceptor",
            default_headers=default_headers,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments
    else:
        client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments

    if not args_raw:
        raise RuntimeError("No tool call arguments returned from model")

    args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    return args_dict["intent"]


class Banking77Dataset:
    def __init__(self):
        self._cache = {}
        self._label_names = None

    def _load_split(self, split: str):
        if split not in self._cache:
            ds = load_dataset("banking77", split=split, trust_remote_code=False)
            self._cache[split] = ds
            if self._label_names is None and hasattr(ds.features.get("label"), "names"):
                self._label_names = ds.features["label"].names
        return self._cache[split]

    def ensure_ready(self, splits):
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, *, split: str, index: int) -> dict:
        ds = self._load_split(split)
        idx = index % len(ds)
        row = ds[idx]
        label_idx = int(row.get("label", 0))
        label_text = (
            self._label_names[label_idx]
            if self._label_names and label_idx < len(self._label_names)
            else f"label_{label_idx}"
        )
        return {"index": idx, "split": split, "text": str(row.get("text", "")), "label": label_text}

    @property
    def label_names(self) -> list:
        if self._label_names is None:
            self._load_split("train")
        return self._label_names or []


def create_banking77_local_api(system_prompt: str):
    dataset = Banking77Dataset()
    dataset.ensure_ready(["train", "test"])

    async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        split = request.env.config.get("split", "train")
        seed = request.env.seed
        sample = dataset.sample(split=split, index=seed)

        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url")
        api_key = policy_config.get("api_key")

        start = time.perf_counter()
        predicted_intent = await classify_banking77_query(
            query=sample["text"],
            system_prompt=system_prompt,
            available_intents=format_available_intents(dataset.label_names),
            model=policy_config.get("model", "gpt-4o-mini"),
            api_key=api_key,
            inference_url=inference_url,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        expected_intent = sample["label"]
        is_correct = (
            predicted_intent.lower().replace("_", " ").strip()
            == expected_intent.lower().replace("_", " ").strip()
        )
        reward = 1.0 if is_correct else 0.0

        policy_cfg_for_trace = {
            key: value
            for key, value in policy_config.items()
            if key not in {"trace_correlation_id", "trace"}
        }
        trace_correlation_id = extract_trace_correlation_id(
            policy_config=policy_cfg_for_trace,
            inference_url=str(inference_url or ""),
        )

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                outcome_objectives={"reward": reward, "latency_ms": latency_ms},
                instance_objectives=[{"reward": reward, "latency_ms": latency_ms}],
                details={"latency_ms": latency_ms},
            ),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
            success_status=SuccessStatus.SUCCESS,
        )

    def provide_taskset_description():
        return {
            "splits": ["train", "test"],
            "sizes": {"train": dataset.size("train"), "test": dataset.size("test")},
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(split="train", index=seed)
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": sample["split"], "index": sample["index"]},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={"query": sample["text"], "expected_intent": sample["label"]},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description=f"{APP_NAME} local API for classifying customer queries into banking intents.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, headers=headers, timeout=5.0)
            if response.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)

    raise RuntimeError(f"Health check failed: {health_url} not ready after {timeout}s")


def build_initial_prompt() -> dict[str, Any]:
    user_prompt = (
        "Customer Query: {query}\n\n"
        "Available Intents:\n{available_intents}\n\n"
        "Classify this query into one of the above banking intents using the tool call."
    )
    return {
        "id": "banking77_pattern",
        "name": "Banking77 Classification",
        "messages": [
            {"role": "system", "order": 0, "pattern": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "order": 1, "pattern": user_prompt},
        ],
        "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
    }


def build_mipro_config(
    *,
    task_app_url: str,
    mode: str,
    train_seeds: list[int],
    val_seeds: list[int],
) -> dict[str, Any]:
    policy_model = os.environ.get("BANKING77_POLICY_MODEL", args.model)
    policy_provider = os.environ.get("BANKING77_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("BANKING77_PROPOSER_MODEL", policy_model)
    proposer_provider = os.environ.get("BANKING77_PROPOSER_PROVIDER", policy_provider)
    proposer_url = os.environ.get(
        "BANKING77_PROPOSER_URL", "https://api.openai.com/v1/responses"
    )
    # Note: task_app_api_key is not included - backend resolves from uploaded ENVIRONMENT_API_KEY
    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": "banking77",
            "task_app_url": task_app_url,
            "initial_prompt": build_initial_prompt(),
            "policy": {
                "model": policy_model,
                "provider": policy_provider,
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "mipro": {
                "mode": mode,
                "bootstrap_train_seeds": train_seeds,
                "val_seeds": val_seeds,
                "online_pool": train_seeds,
                "online_proposer_mode": "inline",
                "online_proposer_min_rollouts": 20,
                "proposer": {
                    "mode": "instruction_only",
                    "model": proposer_model,
                    "provider": proposer_provider,
                    "inference_url": proposer_url,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "generate_at_iterations": [0],
                    "instructions_per_batch": 1,
                },
            },
        },
    }


def create_job(backend_url: str, api_key: str, config_body: dict[str, Any]) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/jobs",
        json={"algorithm": "mipro", "config_body": config_body},
        headers=headers,
        timeout=60.0,
    )
    if response.status_code != 200:
        print(f"Error response: {response.status_code}")
        print(f"Response body: {response.text}")
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Missing job_id in response: {payload}")
    return str(job_id)


def get_job_detail(
    backend_url: str, api_key: str, job_id: str, *, include_metadata: bool = True
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/jobs/{job_id}",
        params={
            "include_events": False,
            "include_snapshot": False,
            "include_metadata": include_metadata,
        },
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def poll_job(
    backend_url: str,
    api_key: str,
    job_id: str,
    *,
    timeout: float = 1800.0,
    interval: float = 5.0,
) -> dict[str, Any]:
    start = time.time()
    last_status = ""
    while time.time() - start < timeout:
        detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
        status = detail.get("status", "")
        if status != last_status:
            elapsed = time.time() - start
            print(f"  Status: {status} (elapsed: {elapsed:.0f}s)")
            last_status = status
        if status in {"succeeded", "failed"}:
            return detail
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


async def main():
    print("\n" + "="*60)
    print("Setting up Banking77 Local API for MIPRO")
    print("="*60)

    baseline_app = create_banking77_local_api(BASELINE_SYSTEM_PROMPT)
    baseline_port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if baseline_port != LOCAL_API_PORT:
        print(f"Port {LOCAL_API_PORT} in use, using port {baseline_port} instead")

    run_server_background(baseline_app, baseline_port)
    print(f"Waiting for local API on port {baseline_port}...")
    wait_for_health_check_sync("localhost", baseline_port, ENVIRONMENT_API_KEY, timeout=30.0)
    print("Local API ready!")

    if LOCAL_MODE:
        print(f"\nUsing {LOCAL_HOST} (no tunnel)...")
        task_app_url = f"http://{LOCAL_HOST}:{baseline_port}"
        tunnel = None
    else:
        print("\nProvisioning Cloudflare tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=baseline_port,
            backend=TUNNEL_BACKEND,
            backend_url=SYNTH_API_BASE,
            progress=True,
        )
        task_app_url = tunnel.url
        # Wait for tunnel propagation
        print("Waiting for tunnel propagation...")
        await asyncio.sleep(10.0)

    print(f"Task App URL: {task_app_url}")

    print("\n" + "="*60)
    print("Running MIPRO Optimization")
    print("="*60)

    train_seeds = list(range(args.train_size))
    val_seeds = list(range(args.train_size, args.train_size + args.val_size))
    
    config_body = build_mipro_config(
        task_app_url=task_app_url,
        mode=args.mode,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )

    print(f"Creating MIPRO job...")
    print(f"  Model: {args.model}")
    print(f"  Mode: {args.mode}")
    print(f"  Rollouts: {args.rollouts}")
    print(f"  Train seeds: {args.train_size}")
    print(f"  Val seeds: {args.val_size}")

    start_time = time.time()
    
    job_id = create_job(SYNTH_API_BASE, API_KEY, config_body)
    print(f"\nJob ID: {job_id}")
    print("Polling for completion...\n")

    mipro_result = poll_job(SYNTH_API_BASE, API_KEY, job_id, timeout=1800.0)

    elapsed = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Job completed in {elapsed:.1f}s")
    print(f"Status: {mipro_result.get('status')}")
    print("="*60)

    results = {
        "method": "synth_mipro",
        "job_id": job_id,
        "status": mipro_result.get("status"),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": args.model,
            "mode": args.mode,
            "rollouts": args.rollouts,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
    }

    if mipro_result.get("status") == "succeeded":
        best_score = mipro_result.get("best_score")
        print(f"\nBest score: {best_score}")
        results["results"] = {
            "best_score": best_score,
            "metadata": mipro_result.get("metadata", {}),
        }
    else:
        error = mipro_result.get("error", "Unknown error")
        print(f"\nError: {error}")
        results["error"] = error

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"banking77_synth_mipro_{timestamp}.json"
    output_path = results_dir / output_filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Cleanup
    if tunnel:
        print("\nCleaning up tunnels...")
        cleanup_all()

    print("\nDone!")
    return results


if __name__ == "__main__":
    asyncio.run(main())
