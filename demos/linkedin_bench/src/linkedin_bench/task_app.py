"""
FastAPI Task App for LinkedIn Corporate Monitoring Benchmark.

This task app is called by the Synth backend during GEPA optimization.
Uses the synth_ai SDK's LocalAPI pattern for proper integration.
"""

import os
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from fastapi import Request
from kernel import AsyncKernel

# Import Synth SDK types for proper protocol compatibility
from synth_ai.data.artifacts import Artifact
from synth_ai.data.coding_agent_context import ContextOverride
from synth_ai.data.enums import SuccessStatus
from synth_ai.data.rubrics import Criterion, Rubric
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.localapi._impl.server import RubricBundle
from synth_ai.sdk.localapi._impl.rollout_helpers import (
    build_rollout_response as _build_rollout_response_raw,
)
from synth_ai.sdk.localapi._impl.trace_correlation_helpers import (
    build_trace_payload,
    extract_trace_correlation_id,
)

from .kernel_runner import (
    DEFAULT_POOL_NAME,
    DEFAULT_POOL_SIZE,
    ensure_browser_ready,
    get_or_create_pool,
    run_claude_code,
    write_skill_file,
)
from .skill_template import get_skill_content
from .tasks import TASKS, Task, get_task_by_seed
def _sanitize_rollout_request(request: "RolloutRequest") -> None:
    """Ensure numeric fields are int where Rust expects i32 (avoids float→i32 serde error)."""
    if hasattr(request, "safety") and request.safety is not None:
        if hasattr(request.safety, "max_time_s") and isinstance(request.safety.max_time_s, float):
            request.safety.max_time_s = int(request.safety.max_time_s)
        if hasattr(request.safety, "max_memory_mb") and isinstance(
            request.safety.max_memory_mb, float
        ):
            request.safety.max_memory_mb = int(request.safety.max_memory_mb)


def build_rollout_response(request, *args, **kwargs):
    """Wrapper that sanitizes request before calling the Rust-backed builder."""
    _sanitize_rollout_request(request)
    # The Rust deserializer doesn't handle the Python Artifact class, and the
    # Pydantic RolloutResponse model doesn't accept plain dicts for artifact.
    # Workaround: strip artifacts before the Rust call, then inject them back.
    saved_artifacts = kwargs.pop("artifact", None)
    response = _build_rollout_response_raw(request, *args, **kwargs)
    if saved_artifacts:
        response.artifact = saved_artifacts
    return response


from .verifier import (
    VerificationResult,
    calculate_reward,
    count_agent_steps,
    count_sleep_commands,
    verify_with_llm,
)

# Load environment variables
load_dotenv()

# App configuration
APP_ID = "linkedin_bench"
APP_NAME = "LinkedIn Corporate Monitoring Benchmark"

# Global Kernel client
kernel_client: AsyncKernel | None = None


def normalize_interceptor_base(inference_url: str) -> tuple[str, str | None]:
    """Normalize a Synth interceptor URL into (base_url, correlation_id).

    Strips known API suffixes and extracts the ?cid= query parameter.
    """
    parsed = urlparse(inference_url)
    base_path = parsed.path or ""
    for suffix in [
        "/v1/chat/completions",
        "/chat/completions",
        "/responses",
        "/v1/responses",
        "/v1/messages",
        "/messages",
    ]:
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    base = f"{parsed.scheme}://{parsed.netloc}{base_path}"
    cid_values = parse_qs(parsed.query or "").get("cid", [])
    correlation_id = cid_values[0] if cid_values else None
    return base, correlation_id


def provide_taskset_description() -> str:
    """Provide description of the task set."""
    return """LinkedIn Corporate Monitoring Benchmark

This benchmark evaluates LinkedIn automation tasks for corporate monitoring:
- Commenter and reactor breakdowns on company posts
- Executive engagement detection
- Employee/title census tasks
- Hiring/attrition tracking
- Post engagement and hashtag analytics
- Shared connections and top follower counts

The agent uses Claude Code with agent-browser CLI to automate Chrome.
"""


def provide_task_instances(seeds=None) -> list[TaskInfo]:
    """Provide list of task instances."""
    return [
        TaskInfo(
            id=str(i),
            name=task.id,
            description=task.prompt,
        )
        for i, task in enumerate(TASKS)
    ]


async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """
    Handle a single evaluation rollout.

    This runs the full pipeline:
    1. Acquire browser from Kernel pool (with direct fallback)
    2. Write skill file (which GEPA optimizes)
    3. Run Claude Code with task prompt (optionally through Synth interceptor)
    4. Verify result with LLM judge
    5. Return outcome reward with trace_correlation_id and structured artifacts
    """
    global kernel_client

    seed = request.env.seed or 0
    policy_config = request.policy.config if request.policy else {}

    # Get the task for this seed
    task = get_task_by_seed(seed)

    # Use task-specific timeout
    timeout = task.timeout or policy_config.get("timeout", 120)

    # Extract interceptor URL from policy_config
    inference_url = policy_config.get("inference_url")

    # Allow overriding the interceptor host for local development.
    # When INTERCEPTOR_BASE_OVERRIDE is set, the host portion of inference_url is
    # replaced while preserving the path (trial_id, correlation_id).
    interceptor_override = os.environ.get("INTERCEPTOR_BASE_OVERRIDE")
    if inference_url and interceptor_override:
        parsed_orig = urlparse(inference_url)
        parsed_override = urlparse(interceptor_override)
        inference_url = f"{parsed_override.scheme}://{parsed_override.netloc}{parsed_orig.path}"
        if parsed_orig.query:
            inference_url += f"?{parsed_orig.query}"
        print(f"Interceptor override: {inference_url}")

        # Auto-register the trial on the local interceptor so it doesn't 404.
        # Extract trial_id from the path: /api/interceptor/v1/{trial_id}/...
        path_parts = [p for p in parsed_orig.path.split("/") if p]
        # Find trial_id: it's the segment after "v1" in the interceptor path
        trial_id_for_reg = None
        for i, part in enumerate(path_parts):
            if part == "v1" and i + 1 < len(path_parts):
                trial_id_for_reg = path_parts[i + 1]
                break
        if trial_id_for_reg:
            try:
                import httpx as _hx

                reg_url = (
                    f"{parsed_override.scheme}://{parsed_override.netloc}"
                    f"/api/interceptor/v1/debug/register_trial/{trial_id_for_reg}"
                )
                reg_resp = _hx.post(
                    reg_url,
                    json={
                        "job_id": request.trace_correlation_id,
                        "seed": seed,
                        "stage_key": {"pipeline_id": "eval", "stage_id": "passthrough"},
                        "baseline_messages": [],
                        "deltas": {},
                        "ttl_seconds": 7200,
                    },
                    timeout=10.0,
                )
                print(f"Auto-registered trial '{trial_id_for_reg}' on override: {reg_resp.status_code}")
            except Exception as e:
                print(f"Warning: Failed to auto-register trial on override: {e}")

    # Compute interceptor base URL and correlation ID if inference_url is present
    interceptor_base_url = None
    interceptor_auth_token = None
    if inference_url:
        base_url, correlation_id = normalize_interceptor_base(inference_url)
        if correlation_id:
            interceptor_base_url = f"{base_url}/{correlation_id}"
        else:
            interceptor_base_url = base_url
        # Use SYNTH_API_KEY for interceptor auth, fall back to ANTHROPIC_API_KEY
        interceptor_auth_token = os.environ.get("SYNTH_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        print(f"Interceptor URL: {interceptor_base_url}")

    # Get skill content from context_overrides (GEPA provides optimized versions)
    skill_content = get_skill_content()  # default
    using_override = False
    if request.context_overrides:
        print(f"  [CTX] context_overrides present: {len(request.context_overrides)} override(s)")
        for i, override in enumerate(request.context_overrides):
            print(f"  [CTX] override[{i}]: file_artifacts={bool(override.file_artifacts)}, "
                  f"env_vars={bool(override.env_vars)}, preflight={bool(override.preflight_script)}")
            if override.file_artifacts:
                for path, content in override.file_artifacts.items():
                    print(f"  [CTX]   file: {path} ({len(content)} chars)")
                    if "skill" in path.lower():
                        skill_content = content
                        using_override = True
    else:
        print(f"  [CTX] No context_overrides in request")

    # Validate and log skill content
    if using_override:
        print(f"  [SKILL] Using MUTATED skill ({len(skill_content)} chars)")
    else:
        print(f"  [SKILL] Using DEFAULT skill ({len(skill_content)} chars)")

    # Check for common issues with skill content
    if skill_content.lstrip().startswith("{"):
        import json as _json
        try:
            parsed = _json.loads(skill_content)
            if "instruction" in parsed:
                print(f"  [SKILL] WARNING: Skill content is JSON-wrapped! Unwrapping 'instruction' field")
                skill_content = parsed["instruction"]
                print(f"  [SKILL] Unwrapped skill content ({len(skill_content)} chars)")
        except _json.JSONDecodeError:
            print(f"  [SKILL] WARNING: Skill starts with '{{' but isn't valid JSON")

    has_frontmatter = skill_content.strip().startswith("---")
    has_name = "name:" in skill_content[:200] if has_frontmatter else False

    # Ensure YAML frontmatter is present — Claude Code requires it to discover skills.
    # The proposer sometimes strips it during mutation.
    REQUIRED_FRONTMATTER = (
        "---\n"
        "name: linkedin\n"
        "description: Automate LinkedIn tasks using agent-browser connected to the local Chrome browser via CDP.\n"
        "---\n\n"
    )
    if not has_frontmatter:
        print(f"  [SKILL] FIX: Adding missing YAML frontmatter for skill discovery")
        skill_content = REQUIRED_FRONTMATTER + skill_content
        has_frontmatter = True
        has_name = True
    elif has_frontmatter and not has_name:
        # Has frontmatter but missing name field — replace it
        end_marker = skill_content.find("---", 3)
        if end_marker > 0:
            print(f"  [SKILL] FIX: Frontmatter missing 'name:' field, replacing frontmatter")
            skill_content = REQUIRED_FRONTMATTER + skill_content[end_marker + 3:].lstrip("\n")
            has_name = True

    print(f"  [SKILL] Has frontmatter: {has_frontmatter}, Has name: {has_name}")
    print(f"  [SKILL] First 300 chars:\n{skill_content[:300]}")
    print(f"  [SKILL] ---")

    print(f"\n{'=' * 60}")
    print(f"Rollout: seed={seed}, task={task.id}")
    print(f"Timeout: {timeout}s")
    print(f"Trace correlation ID: {request.trace_correlation_id}")
    print(f"{'=' * 60}")

    # Initialize Kernel client if needed
    if kernel_client is None:
        api_key = os.environ.get("KERNEL_API_KEY")
        if not api_key:
            return build_rollout_response(
                request=request,
                outcome_reward=0.0,
                inference_url=inference_url,
                success_status=SuccessStatus.FAILURE,
                status_detail="KERNEL_API_KEY not set",
            )
        kernel_client = AsyncKernel(api_key=api_key)

        # Ensure browser pool exists
        try:
            await get_or_create_pool(kernel_client, DEFAULT_POOL_NAME, "linkedin", DEFAULT_POOL_SIZE)
        except Exception as e:
            print(f"Warning: Could not create browser pool: {e}")

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key and not interceptor_base_url:
        return build_rollout_response(
            request=request,
            outcome_reward=0.0,
            inference_url=inference_url,
            success_status=SuccessStatus.FAILURE,
            status_detail="ANTHROPIC_API_KEY not set and no interceptor configured",
        )

    session_id = None
    use_pool = True
    start_time = time.time()

    try:
        # Acquire browser from pool, fall back to direct creation
        print(f"Acquiring browser from pool '{DEFAULT_POOL_NAME}'...")
        try:
            acquire_result = await kernel_client.browser_pools.acquire(DEFAULT_POOL_NAME)
            session_id = acquire_result.session_id
            print(f"  Session: {session_id}")
            print(f"  Live view: {acquire_result.browser_live_view_url}")
        except Exception as e:
            print(f"  Pool unavailable ({e}), creating browser directly...")
            use_pool = False
            browser = await kernel_client.browsers.create(
                profile={"name": "linkedin"},
                stealth=True,
                timeout_seconds=max(timeout + 120, 600),
            )
            session_id = browser.session_id
            print(f"  Created session: {session_id}")
            print(f"  Live view: {browser.browser_live_view_url}")

        # Ensure browser is ready
        print("Ensuring browser is ready...")
        await ensure_browser_ready(kernel_client, session_id)

        # Write skill file
        print("Writing skill file...")
        await write_skill_file(kernel_client, session_id, skill_content)

        # Run Claude Code (through interceptor if configured)
        print("Running Claude Code...")
        run_result = await run_claude_code(
            kernel_client,
            session_id,
            task.prompt,
            anthropic_api_key or "",
            timeout=timeout,
            interceptor_base_url=interceptor_base_url,
            interceptor_auth_token=interceptor_auth_token,
        )
        elapsed = time.time() - start_time

        print(f"Claude finished in {elapsed:.1f}s (exit code: {run_result.exit_code})")
        print(f"Output length: {len(run_result.output)} chars")

        # Log agent output summary for debugging
        output_lines = run_result.output.strip().split("\n")
        # Show last 10 meaningful lines (skip empty)
        meaningful = [l for l in output_lines if l.strip()][-10:]
        print(f"  [OUTPUT] Last 10 lines:")
        for line in meaningful:
            print(f"    {line[:200]}")
        # Check for common failure patterns
        output_lower = run_result.output.lower()
        if "skill not available" in output_lower or "skill is not" in output_lower:
            print(f"  [OUTPUT] WARNING: Agent reports skill not available!")
        if "unknown skill" in output_lower:
            print(f"  [OUTPUT] WARNING: Agent reports unknown skill!")
        if "500+" in run_result.output:
            print(f"  [OUTPUT] WARNING: Agent returned '500+' (truncated LinkedIn count)")

        # Count agent steps and sleep commands
        num_steps = count_agent_steps(run_result.output)
        num_sleeps, total_sleep_ms = count_sleep_commands(run_result.output)
        print(f"Agent steps: {num_steps}")
        if num_sleeps > 0:
            print(f"Sleep commands: {num_sleeps} ({total_sleep_ms}ms total) - PENALIZED")

        # Verify result
        print("Verifying result with LLM judge...")
        verification = await verify_with_llm(task, run_result.output, anthropic_api_key or "")
        print(f"  Correct: {verification.correct}")
        print(f"  Extracted: {verification.extracted_answer}")
        print(f"  Reason: {verification.reason}")

        # Calculate reward (with sleep penalty)
        reward = calculate_reward(
            correctness_score=verification.raw_score,
            elapsed_seconds=elapsed,
            num_agent_steps=num_steps,
            num_sleep_commands=num_sleeps,
            total_sleep_ms=total_sleep_ms,
            max_time=float(timeout),
            max_steps=20,
        )
        print(f"Final reward: {reward:.3f}")

        return build_rollout_response(
            request=request,
            outcome_reward=reward,
            inference_url=inference_url,
            success_status=SuccessStatus.SUCCESS,
            status_detail=f"Correct: {verification.correct}, Reward: {reward:.3f}",
            artifact=[
                Artifact(
                    content=verification.extracted_answer or "",
                    content_type="agent_result",
                    metadata={
                        "task_id": task.id,
                        "correct": verification.correct,
                        "reason": verification.reason,
                    },
                )
            ],
            details={
                "task_id": task.id,
                "correct": verification.correct,
                "extracted_answer": verification.extracted_answer,
                "verification_reason": verification.reason,
                "elapsed_seconds": elapsed,
                "num_steps": num_steps,
                "num_sleeps": num_sleeps,
                "total_sleep_ms": total_sleep_ms,
                "exit_code": run_result.exit_code,
            },
            outcome_objectives={"reward": reward, "latency_ms": int(elapsed * 1000)},
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        elapsed = time.time() - start_time
        print(f"Rollout failed: {e}")

        return build_rollout_response(
            request=request,
            outcome_reward=0.0,
            inference_url=inference_url,
            success_status=SuccessStatus.FAILURE,
            status_detail=f"Error: {str(e)[:200]}",
            details={"error": str(e), "task_id": task.id, "elapsed_seconds": elapsed},
        )

    finally:
        # Release browser back to pool or clean up direct browser
        if session_id and kernel_client:
            if use_pool:
                print(f"Releasing browser {session_id}...")
                try:
                    await kernel_client.browser_pools.release(
                        DEFAULT_POOL_NAME, session_id=session_id, reuse=True
                    )
                    print("  Released with reuse=True")
                except Exception as e:
                    print(f"  Warning: Failed to release: {e}")
            else:
                print(f"Deleting browser {session_id}...")
                try:
                    import httpx

                    kernel_api_key = os.environ.get("KERNEL_API_KEY", "")
                    httpx.delete(
                        f"https://api.onkernel.com/browsers/{session_id}",
                        headers={"Authorization": f"Bearer {kernel_api_key}"},
                    )
                    print("  Deleted")
                except Exception as e:
                    print(f"  Warning: cleanup failed: {e}")


# Rubrics for RLM-based scoring (used when verifier is enabled on the eval job)
LINKEDIN_RUBRICS = RubricBundle(
    outcome=Rubric(
        version="1.0",
        goal_text=(
            "Evaluate whether the browser automation agent successfully completed "
            "a LinkedIn data-retrieval task using Claude Code and agent-browser CLI."
        ),
        criteria=[
            Criterion(
                id="task_completion",
                description=(
                    "Did the agent return the requested numeric value (follower count, "
                    "connection count, etc.)? A plausible number extracted from the "
                    "correct LinkedIn profile counts as success."
                ),
                weight=0.5,
                required=True,
            ),
            Criterion(
                id="navigation_efficiency",
                description=(
                    "Did the agent navigate to the correct LinkedIn page efficiently, "
                    "without excessive retries, unnecessary page loads, or blind sleep "
                    "commands? Fewer steps and no hardcoded sleeps is better."
                ),
                weight=0.3,
            ),
            Criterion(
                id="tool_usage",
                description=(
                    "Did the agent use the agent-browser CLI tools appropriately "
                    "(navigate, click, extract, screenshot) rather than attempting "
                    "raw HTTP requests, scraping, or other brittle approaches?"
                ),
                weight=0.2,
            ),
        ],
        aggregation="weighted_sum",
    )
)

# Create the app using Synth's LocalAPI helper
app = create_local_api(
    LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description="Evaluate browser automation skills on LinkedIn tasks using Claude Code and agent-browser.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        rubrics=LINKEDIN_RUBRICS,
        cors_origins=["*"],
    )
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8030"))
    uvicorn.run(app, host="0.0.0.0", port=port)
