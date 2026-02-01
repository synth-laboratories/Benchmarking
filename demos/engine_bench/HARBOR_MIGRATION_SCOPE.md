# EngineBench Harbor Migration Scope

This document scopes out how to migrate the EngineBench demo to use the new HarborBuildSpec and unified task app semantics.

## Current State

### Current Harbor Demo (`demos/harbor/`)
- **`create_deployment.py`**: Manually creates Harbor deployments by:
  1. Packaging Dockerfile + `run_rollout.py` into base64 tar.gz
  2. Calling `/api/harbor/deployments` API directly
  3. Waiting for build to complete
  
- **`run_harbor_eval.py`**: Runs evaluations via `EvalJob`:
  1. Constructs task app URL: `{backend_url}/api/harbor/deployments/{deployment_id}`
  2. Uses `EvalJobConfig` with `task_app_url` pointing to Harbor endpoint
  3. Harbor endpoint handles rollout execution in Daytona sandboxes

### Current Local Task App (`demos/engine_bench/localapi_engine_bench.py`)
- Defines `LocalAPIConfig` with `run_rollout` handler
- Can run locally or via Daytona sandboxes (when `USE_DAYTONA_SANDBOXES=True`)
- Used by GEPA via `run_gepa_unified.py` which:
  1. Starts local FastAPI server
  2. Creates tunnel if needed
  3. Passes task app URL to `PromptLearningJob`

## Target State (New Approach)

### Step 1: Upload Deployment via HarborBuildSpec

**New File**: `demos/engine_bench/upload_harbor_deployment.py`

```python
#!/usr/bin/env python3
"""
Upload EngineBench deployment to Harbor using HarborBuildSpec.

Usage:
    uv run python demos/engine_bench/upload_harbor_deployment.py \
        --name engine-bench-v1 \
        --agent codex
"""

import argparse
import os
from pathlib import Path

from synth_ai.sdk.harbor.build_spec import HarborBuildSpec
from synth_ai.sdk.harbor.uploader import upload_harbor_deployment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Deployment name")
    parser.add_argument("--agent", default="codex", choices=["codex", "opencode", "claude_code"])
    parser.add_argument("--backend-url", default="https://api-dev.usesynth.ai")
    args = parser.parse_args()

    # Get Harbor directory (where Dockerfile and run_rollout.py live)
    harbor_dir = Path(__file__).parent.parent / "harbor"
    
    # Define build spec (user-facing abstraction)
    spec = HarborBuildSpec(
        name=args.name,
        dockerfile_path=harbor_dir / "Dockerfile",
        context_dir=harbor_dir,  # Package entire harbor/ directory
        entrypoint="run_rollout --input /tmp/rollout.json --output /tmp/result.json",
        entrypoint_mode="file",
        limits={
            "timeout_s": 600,
            "cpu_cores": 4,
            "memory_mb": 8192,
            "disk_mb": 20480,
        },
        metadata={
            "agent_type": args.agent,
            "benchmark": "engine-bench",
            "version": "1.0",
        },
    )
    
    # Upload (SDK handles packaging + API calls)
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY not set")
    
    print(f"Uploading deployment '{args.name}'...")
    deployment = upload_harbor_deployment(
        spec=spec,
        backend_url=args.backend_url,
        api_key=api_key,
    )
    
    print(f"✅ Deployment created: {deployment.id}")
    print(f"   Status: {deployment.status}")
    print(f"   Build ID: {deployment.build_id}")
    
    if deployment.status == "ready":
        print(f"   Snapshot: {deployment.snapshot_id}")
    else:
        print(f"   Waiting for build... (check status with: synth harbor status {deployment.id})")


if __name__ == "__main__":
    main()
```

**Changes needed**:
- Create `synth_ai/sdk/harbor/build_spec.py` with `HarborBuildSpec` dataclass
- Create `synth_ai/sdk/harbor/packager.py` with `HarborPackager` (packages context into tar.gz)
- Create `synth_ai/sdk/harbor/uploader.py` with `upload_harbor_deployment()` function
- Move `create_deployment.py` logic into SDK abstractions

### Step 2: Create Instances for Seeds

**New File**: `demos/engine_bench/create_harbor_instances.py`

```python
#!/usr/bin/env python3
"""
Create Harbor instances for EngineBench seeds.

Usage:
    uv run python demos/engine_bench/create_harbor_instances.py \
        --deployment-id <id> \
        --count 100
"""

import argparse
import os

from synth_ai.sdk.harbor.instances import create_harbor_instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--backend-url", default="https://api-dev.usesynth.ai")
    args = parser.parse_args()

    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY not set")
    
    print(f"Creating {args.count} instances for deployment {args.deployment_id}...")
    instances = create_harbor_instances(
        deployment_id=args.deployment_id,
        count=args.count,
        backend_url=args.backend_url,
        api_key=api_key,
    )
    
    print(f"✅ Created {len(instances)} instances:")
    for instance in instances[:10]:  # Show first 10
        print(f"   Seed {instance.seed}: {instance.id} ({instance.status})")
    if len(instances) > 10:
        print(f"   ... and {len(instances) - 10} more")


if __name__ == "__main__":
    main()
```

**Changes needed**:
- Create `synth_ai/sdk/harbor/instances.py` with `create_harbor_instances()` function
- Backend: Implement `/api/harbor/deployments/{id}/instances/batch` endpoint
- Backend: Add `HarborInstance` database model

### Step 3: Use Unified Task App Interface

**New File**: `demos/engine_bench/run_harbor_gepa.py`

```python
#!/usr/bin/env python3
"""
Run EngineBench GEPA using Harbor via unified task app interface.

Usage:
    # Step 1: Upload deployment (one-time)
    uv run python demos/engine_bench/upload_harbor_deployment.py --name engine-bench-v1
    
    # Step 2: Create instances for seeds (one-time)
    uv run python demos/engine_bench/create_harbor_instances.py \
        --deployment-id <id> \
        --count 100
    
    # Step 3: Run GEPA with Harbor backend
    uv run python demos/engine_bench/run_harbor_gepa.py \
        --deployment-id <id> \
        --config enginebench_gepa.toml
"""

import argparse
import os
from pathlib import Path

from localapi_engine_bench import run_rollout, provide_task_instances, provide_taskset_description
from synth_ai.sdk.localapi import LocalAPIConfig
from synth_ai.sdk.localapi.harbor_adapter import HarborExecutionBackend, HarborInstanceProvider
from synth_ai.sdk.localapi.harbor_config import HarborDeploymentRef
from synth_ai.sdk.optimization.internal.configs.prompt_learning import PromptLearningConfig
from synth_ai.sdk.optimization.internal.task_app import create_task_app_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--config", default="enginebench_gepa.toml")
    parser.add_argument("--backend-url", default="https://api-dev.usesynth.ai")
    args = parser.parse_args()

    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY not set")

    # Create Harbor deployment reference
    harbor_ref = HarborDeploymentRef(
        deployment_id=args.deployment_id,
        backend_url=args.backend_url,
        api_key=api_key,
    )

    # Create Harbor instance provider (maps seeds to instance IDs)
    instance_provider = HarborInstanceProvider(
        deployment_ref=harbor_ref,
        # SDK queries backend to get instance IDs for seeds 0..N-1
    )

    # Create unified config with Harbor execution backend
    config = LocalAPIConfig(
        app_id="engine_bench",
        name="EngineBench - Pokemon TCG Card Implementation",
        description="EngineBench evaluates coding agents on Pokemon TCG card implementations in Rust.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,  # Still needed for local dev/debugging
        execution_backend="harbor",  # Use Harbor instead of local server
        harbor=harbor_ref,
        instance_provider=instance_provider,
    )

    # Create task app client (uses HarborExecutionBackend internally)
    task_app_client = create_task_app_client(config)

    # Load GEPA config
    config_path = Path(__file__).parent / args.config
    gepa_config = PromptLearningConfig.from_toml(config_path)

    # Update gepa_config to use Harbor task app
    gepa_config.task_app_url = None  # Not needed - using unified interface
    gepa_config.task_app_config = config  # Pass config object instead

    # Run GEPA (same as before, but uses HarborExecutionBackend)
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

    job = PromptLearningJob(config=gepa_config)
    job.submit()
    print(f"Job ID: {job.job_id}")

    result = job.poll_until_complete(progress=True)
    print(f"Mean reward: {result.mean_reward:.4f}" if result.mean_reward else "N/A")


if __name__ == "__main__":
    main()
```

**Changes needed**:
- Create `synth_ai/sdk/localapi/harbor_adapter.py` with `HarborExecutionBackend` class
- Create `synth_ai/sdk/localapi/harbor_provider.py` with `HarborInstanceProvider` class
- Create `synth_ai/sdk/localapi/harbor_config.py` with `HarborDeploymentRef` dataclass
- Extend `LocalAPIConfig` to support `execution_backend`, `harbor`, `instance_provider` fields
- Update `create_local_api()` to use `HarborExecutionBackend` when `execution_backend="harbor"`
- Update `PromptLearningJob` to accept `task_app_config` instead of just `task_app_url`

### Step 4: Simplified EvalJob Path (Alternative)

**New File**: `demos/engine_bench/run_harbor_eval_simple.py`

For simple eval jobs (not GEPA), can use `EvalJob` directly with unified interface:

```python
#!/usr/bin/env python3
"""
Run EngineBench eval via Harbor using unified task app interface.

Usage:
    uv run python demos/engine_bench/run_harbor_eval_simple.py \
        --deployment-id <id> \
        --seeds 5
"""

import argparse
import os

from localapi_engine_bench import run_rollout, provide_task_instances
from synth_ai.sdk.eval.job import EvalJob, EvalJobConfig
from synth_ai.sdk.localapi import LocalAPIConfig
from synth_ai.sdk.localapi.harbor_adapter import HarborExecutionBackend, HarborInstanceProvider
from synth_ai.sdk.localapi.harbor_config import HarborDeploymentRef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--backend-url", default="https://api-dev.usesynth.ai")
    args = parser.parse_args()

    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY not set")

    # Create Harbor config
    harbor_ref = HarborDeploymentRef(
        deployment_id=args.deployment_id,
        backend_url=args.backend_url,
        api_key=api_key,
    )

    instance_provider = HarborInstanceProvider(deployment_ref=harbor_ref)

    # Create unified config
    config = LocalAPIConfig(
        app_id="engine_bench",
        rollout=run_rollout,
        provide_task_instances=provide_task_instances,
        execution_backend="harbor",
        harbor=harbor_ref,
        instance_provider=instance_provider,
    )

    # Create task app client (uses HarborExecutionBackend)
    # This returns a URL-like object that EvalJob can use
    task_app_url = config.get_task_app_url()  # Returns Harbor endpoint URL

    # Create eval job (same as before)
    eval_config = EvalJobConfig(
        task_app_url=task_app_url,
        backend_url=args.backend_url,
        api_key=api_key,
        task_app_api_key=api_key,
        app_id="engine-bench",
        env_name="enginebench",
        seeds=list(range(args.seeds)),
        policy_config={"model": args.model, "provider": "openai"},
    )

    job = EvalJob(eval_config)
    job.submit()
    result = job.poll_until_complete(progress=True)
    print(f"Mean reward: {result.mean_reward:.4f}" if result.mean_reward else "N/A")


if __name__ == "__main__":
    main()
```

**Changes needed**:
- Add `get_task_app_url()` method to `LocalAPIConfig` that returns Harbor endpoint when `execution_backend="harbor"`
- Or: `EvalJob` can accept `LocalAPIConfig` directly instead of `task_app_url`

## Migration Steps Summary

### Phase 1: SDK Abstractions (Foundation)
1. ✅ Create `HarborBuildSpec` dataclass
2. ✅ Create `HarborPackager` (packages context)
3. ✅ Create `HarborDeploymentUploader` (API client)
4. ✅ Create `upload_harbor_deployment()` function

### Phase 2: Instance System
1. ✅ Create `HarborInstance` database model
2. ✅ Create `/api/harbor/deployments/{id}/instances/batch` endpoint
3. ✅ Create `create_harbor_instances()` SDK function

### Phase 3: Unified Interface
1. ✅ Create `HarborExecutionBackend` (implements `RolloutExecutor`)
2. ✅ Create `HarborInstanceProvider` (maps seeds to instances)
3. ✅ Create `HarborDeploymentRef` dataclass
4. ✅ Extend `LocalAPIConfig` with `execution_backend`, `harbor`, `instance_provider`
5. ✅ Update `create_local_api()` to support Harbor backend

### Phase 4: Demo Migration
1. ✅ Create `upload_harbor_deployment.py` (replaces `create_deployment.py`)
2. ✅ Create `create_harbor_instances.py` (new)
3. ✅ Create `run_harbor_gepa.py` (uses unified interface)
4. ✅ Create `run_harbor_eval_simple.py` (simplified eval path)
5. ✅ Update `run_harbor_eval.py` to use unified interface (or deprecate)

## Benefits

1. **Separation of Concerns**: Upload (BuildSpec) vs Execution (unified interface)
2. **Reusability**: Same `LocalAPIConfig` works for local dev and Harbor deployment
3. **Type Safety**: Typed abstractions (`HarborBuildSpec`, `HarborDeploymentRef`)
4. **Simpler API**: Users don't need to manually construct Harbor URLs
5. **Instance Support**: Built-in support for per-seed instances
6. **GEPA Integration**: Works transparently with existing GEPA code

## Backward Compatibility

- Old `create_deployment.py` can still work (calls SDK functions internally)
- Old `run_harbor_eval.py` can still work (constructs URLs manually)
- New approach is opt-in - existing code doesn't break

## Testing Plan

1. **Unit Tests**: Test `HarborBuildSpec`, `HarborPackager`, `HarborDeploymentUploader`
2. **Integration Tests**: Test `upload_harbor_deployment()` end-to-end
3. **E2E Tests**: Test `run_harbor_gepa.py` with real Harbor deployment
4. **Migration Test**: Run both old and new paths side-by-side, compare results
