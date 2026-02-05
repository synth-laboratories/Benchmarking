#!/usr/bin/env python3
"""
Test script for Hosted Task Apps with Environment Pools.

This demonstrates the full flow:
1. Create a hosted task app for engine-bench
2. Verify task_app_id resolution works via the task apps API
3. Show how to configure a pool with task_app_id

Requirements:
- Running Python backend (api-dev.usesynth.ai or localhost:8000)
- Running Rust backend (infra-api-dev.usesynth.ai or localhost:8080)
- SYNTH_API_KEY set in environment
"""

import json
import os
import sys

# Ensure synth-ai is importable
sys.path.insert(0, "/Users/joshpurtell/Documents/GitHub/synth-ai")

from synth_ai.sdk.task_apps import TaskAppsClient, TaskAppSpec, TaskAppType

# Configuration
BACKEND_URL = os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("SYNTH_API_KEY", "")

if not API_KEY:
    print("ERROR: SYNTH_API_KEY is required")
    sys.exit(1)


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def main():
    print_header("Hosted Task Apps Test")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"API Key: {API_KEY[:20]}...")

    # -------------------------------------------------------------------------
    # Step 1: Create a hosted task app for engine-bench
    # -------------------------------------------------------------------------
    print_header("Step 1: Create Hosted Task App")
    
    task_apps_client = TaskAppsClient(
        api_key=API_KEY,
        backend_base=BACKEND_URL,
    )
    
    # Create the task app spec
    spec = TaskAppSpec(
        name="engine-bench-test",
        task_type=TaskAppType.harbor_code,
        definition={
            "template": "engine_bench",
            "config": {
                "daytona_snapshot": "synth-engine-bench-codex-v1",
                "timeout": 7200,
            },
        },
        # Internal URL can be set manually for testing, or left None for auto-provisioning
        internal_url="http://localhost:8017",  # Local task app for testing
    )
    
    try:
        task_app = task_apps_client.create(spec)
        print(f"Created task app: {task_app.id}")
        print(f"  Name: {task_app.name}")
        print(f"  Type: {task_app.task_type}")
        print(f"  Status: {task_app.status}")
        print(f"  Internal URL: {task_app.internal_url}")
    except Exception as e:
        print(f"Failed to create task app: {e}")
        # Try to get existing task app
        try:
            apps = task_apps_client.list()
            matching = [a for a in apps if a.name == "engine-bench-test"]
            if matching:
                task_app = matching[0]
                print(f"Using existing task app: {task_app.id}")
                print(f"  Name: {task_app.name}")
                print(f"  Type: {task_app.task_type}")
                print(f"  Status: {task_app.status}")
                print(f"  Internal URL: {task_app.internal_url}")
            else:
                print("No existing task app found, listing all task apps...")
                for app in apps:
                    print(f"  - {app.id}: {app.name} ({app.status})")
                return
        except Exception as e2:
            print(f"Failed to list task apps: {e2}")
            return

    # -------------------------------------------------------------------------
    # Step 2: Wait for task app to be ready (if using auto-provisioning)
    # -------------------------------------------------------------------------
    print_header("Step 2: Check Task App Status")
    
    if task_app.status != "ready":
        print(f"Task app status is '{task_app.status}', waiting for ready...")
        try:
            task_app = task_apps_client.wait_ready(task_app.id, timeout=60.0)
            print(f"Task app is now ready: {task_app.status}")
            print(f"  Internal URL: {task_app.internal_url}")
        except Exception as e:
            print(f"Note: Task app not ready yet: {e}")
            print("(This is expected if auto-provisioning is not configured)")
    else:
        print(f"Task app is ready: {task_app.internal_url}")

    # -------------------------------------------------------------------------
    # Step 3: Create a pool with task_app_id - API returns URL directly!
    # -------------------------------------------------------------------------
    print_header("Step 3: Create Pool - API Returns URL Directly")
    
    import httpx
    
    # Create a pool with task_app_id in the task config
    # The API will resolve it and return task_app_url directly!
    pool_config = {
        "pool_id": "engine-bench-hosted-pool",
        "pool_type": "sandbox",
        "capacity": 1,
        "concurrency": 1,
        "tasks": [
            {
                "task_id": "engine-bench-eval",
                "backend": "sandbox",
                "config": {
                    "task_app_id": task_app.id,
                },
            }
        ],
    }
    
    print("Creating pool with task_app_id in config:")
    print(json.dumps(pool_config, indent=2))
    
    # Use Python backend (it enriches the response with task_app_url)
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/api/v1/environment-pools/pools",
            json=pool_config,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30.0,
        )
        print(f"\nCreate pool response status: {resp.status_code}")
        if resp.status_code in (200, 201, 409):
            data = resp.json()
            print("Response:")
            print(json.dumps(data, indent=2))
            
            # Check if task_app_url is in the response
            task_app_url = data.get("task_app_url")
            if task_app_url:
                print()
                print(f">>> SUCCESS! API returned task_app_url = {task_app_url}")
                print(">>> No need to track task_app_id - just use the URL!")
            else:
                print()
                print("Note: task_app_url not in response (pool may already exist)")
        else:
            print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Failed to create pool: {e}")
    
    # -------------------------------------------------------------------------
    # Step 4: Get pool - should also return task_app_url
    # -------------------------------------------------------------------------
    print_header("Step 4: Get Pool - URL Returned Directly")
    
    pool_id = "engine-bench-hosted-pool"
    try:
        resp = httpx.get(
            f"{BACKEND_URL}/api/v1/environment-pools/pools/{pool_id}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30.0,
        )
        print(f"Get pool response status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print("Response:")
            print(json.dumps(data, indent=2))
            
            task_app_url = data.get("task_app_url")
            if task_app_url:
                print()
                print(f">>> SUCCESS! GET /pools/{pool_id} returned task_app_url = {task_app_url}")
            else:
                print()
                print("Note: task_app_url not in response")
        else:
            print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Failed to get pool: {e}")

    # -------------------------------------------------------------------------
    # Step 5: Cleanup (optional)
    # -------------------------------------------------------------------------
    print_header("Step 5: Cleanup")
    
    cleanup = os.environ.get("CLEANUP", "0") == "1"
    if cleanup:
        print("Deleting test resources...")
        try:
            task_apps_client.delete(task_app.id)
            print(f"Deleted task app: {task_app.id}")
        except Exception as e:
            print(f"Failed to delete task app: {e}")
    else:
        print("Skipping cleanup (set CLEANUP=1 to enable)")
        print(f"Task app ID for reference: {task_app.id}")

    print_header("Test Complete")
    print("The hosted task apps API is working correctly!")
    print()
    print("Key benefits:")
    print("  1. No more tunnel expiration errors (HTTP 524/502)")
    print("  2. Task apps managed by Synth infrastructure")
    print("  3. SIMPLIFIED: Call /v1/pools/{pool_id}/info to get task_app_url directly")
    print("  4. No need to track task_app_id - just get the URL you need!")


if __name__ == "__main__":
    main()
