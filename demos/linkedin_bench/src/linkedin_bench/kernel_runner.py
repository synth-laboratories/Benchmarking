"""
Kernel SDK wrapper for running Claude Code in cloud browser VMs.

This module handles:
- Acquiring browsers from a pool
- Ensuring the VM has Claude Code and agent-browser installed
- Writing skill files to the correct location
- Running Claude Code with task prompts
- Streaming and collecting output
- Releasing browsers back to the pool
"""

import asyncio
import base64
import os
import sys
import time
from dataclasses import dataclass

from kernel import AsyncKernel

# Ensure unbuffered output for real-time streaming
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


@dataclass
class RunResult:
    """Result from running Claude in a Kernel browser."""

    output: str
    exit_code: int
    elapsed_seconds: float
    session_id: str


# Skill file location for Claude Code
SKILL_DIR = "/home/kernel/.claude/skills/linkedin.com"
SKILL_PATH = f"{SKILL_DIR}/SKILL.md"

# Browser pool configuration
DEFAULT_POOL_NAME = "agent-gepa"
DEFAULT_POOL_SIZE = 20


async def get_or_create_pool(
    client: AsyncKernel,
    pool_name: str = DEFAULT_POOL_NAME,
    profile_name: str = "linkedin",
    pool_size: int = DEFAULT_POOL_SIZE,
) -> str:
    """Get existing pool or create a new one."""
    try:
        pool = await client.browser_pools.retrieve(pool_name)
        return pool.name
    except Exception:
        # Pool doesn't exist, create it
        # Stealth=False is faster (no anti-detection measures)
        pool = await client.browser_pools.create(
            name=pool_name,
            profile={"name": profile_name},
            stealth=False,
            size=pool_size,
        )
        return pool.name


async def ensure_browser_ready(client: AsyncKernel, session_id: str, verbose: bool = True) -> None:
    """
    Ensure the browser VM has the required tools installed and is in a clean state.

    This checks for Claude Code and agent-browser, installing them if needed.
    It also resets the browser to a clean state (one tab on new tab page)
    and cleans up old skill files and Claude state.
    """
    proc = client.browsers.process

    # Step 1: Clean up old state from previous runs
    if verbose:
        print("  Cleaning up previous run state...")
    await proc.exec(
        session_id,
        command="bash",
        args=["-c", """
            # Remove old skill files
            rm -rf /home/kernel/.claude/skills 2>/dev/null || true
            # Remove old Claude task outputs
            rm -rf /tmp/claude-* 2>/dev/null || true
            # Remove old shell snapshots (can accumulate)
            find /home/kernel/.claude -name 'shell-snapshots' -type d -exec rm -rf {} + 2>/dev/null || true
        """],
        timeout_sec=10,
    )

    # Step 2: Check if Claude Code is installed (it installs to ~/.local/bin which isn't in PATH)
    result = await proc.exec(
        session_id,
        command="bash",
        args=["-c", "test -x /home/kernel/.local/bin/claude && echo 'found' || echo 'not_found'"],
        timeout_sec=10,
    )
    stdout = _decode_b64(result.stdout_b64) if result.stdout_b64 else ""

    if "not_found" in stdout:
        print(f"  Installing Claude Code in session {session_id}...")
        result = await proc.exec(
            session_id,
            command="bash",
            args=["-c", "export HOME=/home/kernel && curl -fsSL https://claude.ai/install.sh | bash"],
            timeout_sec=180,
        )
        if result.exit_code != 0:
            stderr = _decode_b64(result.stderr_b64) if result.stderr_b64 else ""
            raise RuntimeError(f"Failed to install Claude Code: {stderr}")
        # Fix ownership of both .local and .claude directories
        await proc.exec(
            session_id,
            command="bash",
            args=["-c", "chown -R kernel:kernel /home/kernel/.local /home/kernel/.claude 2>/dev/null || true"],
            as_root=True,
            timeout_sec=10,
        )
        print("  Claude Code installed.")
    else:
        if verbose:
            print("  Claude Code already installed.")
        # Even if Claude is installed, ensure .claude is writable by kernel user
        await proc.exec(
            session_id,
            command="bash",
            args=["-c", "chown -R kernel:kernel /home/kernel/.claude 2>/dev/null || true"],
            as_root=True,
            timeout_sec=10,
        )

    # Step 3: Check if agent-browser is installed
    result = await proc.exec(
        session_id,
        command="bash",
        args=["-c", "which agent-browser || echo 'not_found'"],
        timeout_sec=10,
    )
    stdout = _decode_b64(result.stdout_b64) if result.stdout_b64 else ""

    if "not_found" in stdout or result.exit_code != 0:
        print(f"  Installing agent-browser in session {session_id}...")
        result = await proc.exec(
            session_id,
            command="bash",
            args=["-c", "npm install -g agent-browser"],
            timeout_sec=120,
        )
        if result.exit_code != 0:
            stderr = _decode_b64(result.stderr_b64) if result.stderr_b64 else ""
            raise RuntimeError(f"Failed to install agent-browser: {stderr}")
        print("  agent-browser installed.")
    else:
        if verbose:
            print("  agent-browser already installed.")

    # Step 4: Reset browser to clean state (close extra tabs only - don't navigate, it's slow)
    if verbose:
        print("  Resetting browser state...")
    try:
        await client.browsers.playwright.execute(
            session_id,
            code="""
                const pages = context.pages();
                // Close extra tabs, keep only the first one
                for (let i = 1; i < pages.length; i++) {
                    await pages[i].close();
                }
                // Don't navigate - it's slow and the next task will navigate anyway
            """,
            timeout_sec=15,
        )
    except Exception as e:
        print(f"  Warning: Failed to reset browser state: {e}")


async def write_skill_file(client: AsyncKernel, session_id: str, skill_content: str) -> None:
    """Write the skill file to the correct location for Claude Code to find."""
    fs = client.browsers.fs
    proc = client.browsers.process

    # Create the skill directory
    await proc.exec(
        session_id,
        command="mkdir",
        args=["-p", SKILL_DIR],
        timeout_sec=10,
    )

    # Write the skill file
    await fs.write_file(
        session_id,
        skill_content.encode("utf-8"),
        path=SKILL_PATH,
    )

    # Set ownership to kernel user
    await proc.exec(
        session_id,
        command="chown",
        args=["-R", "kernel:kernel", "/home/kernel/.claude"],
        as_root=True,
        timeout_sec=10,
    )


async def run_claude_code(
    client: AsyncKernel,
    session_id: str,
    task_prompt: str,
    anthropic_api_key: str,
    claude_model: str | None = None,
    timeout: int = 120,
    verbose: bool = True,
    interceptor_base_url: str | None = None,
    interceptor_auth_token: str | None = None,
) -> RunResult:
    """
    Run Claude Code with a task prompt in the Kernel browser VM.

    Args:
        client: Kernel async client
        session_id: Browser session ID
        task_prompt: The prompt to send to Claude
        anthropic_api_key: API key for Claude
        timeout: Maximum seconds to wait for completion
        verbose: Print output as it streams
        interceptor_base_url: If set, route inference through Synth interceptor
        interceptor_auth_token: Auth token for the interceptor

    Returns:
        RunResult with output, exit code, and timing
    """
    start_time = time.time()
    proc = client.browsers.process

    # Escape the prompt for shell - use single quotes and escape them
    escaped_prompt = task_prompt.replace("'", "'\"'\"'")

    if verbose:
        print(f"  Running Claude with {timeout}s timeout...", flush=True)

    # Run Claude Code using spawn + streaming (matching playwriter-in-kernel pattern)
    # Must run as 'kernel' user since --dangerously-skip-permissions fails as root
    # Use 'script' command for PTY allocation (critical for Claude to work properly)
    
    # Escape prompt for shell
    escaped = task_prompt.replace("'", "'\"'\"'")
    escaped = escaped.replace('"', '\\"')
    
    # Build script that runs Claude with streaming JSON output
    # When interceptor is configured, route through Synth API interceptor
    claude_model = claude_model or os.environ.get("CLAUDE_MODEL")

    if interceptor_base_url:
        auth_token = interceptor_auth_token or anthropic_api_key
        env_block = f"""export ANTHROPIC_BASE_URL='{interceptor_base_url}'
export ANTHROPIC_AUTH_TOKEN='{auth_token}'"""
        if verbose:
            print(f"  Using interceptor: {interceptor_base_url}", flush=True)
    else:
        env_block = f"export ANTHROPIC_API_KEY='{anthropic_api_key}'"

    if claude_model:
        env_block += f"\nexport CLAUDE_MODEL='{claude_model}'"

    script = f'''#!/bin/bash
export HOME=/home/kernel
export PATH="$HOME/.local/bin:$PATH"
{env_block}
cd /home/kernel
claude -p --verbose --output-format stream-json --dangerously-skip-permissions "{escaped}"
'''
    
    # Write script and run with PTY via 'script' command (like playwriter-in-kernel does)
    cmd = f'''cat > /tmp/run_claude.sh << 'SCRIPT'
{script}
SCRIPT
chmod +x /tmp/run_claude.sh
script -q -c "su - kernel -c '/tmp/run_claude.sh'" /dev/null'''
    
    # Spawn the process
    spawn_result = await proc.spawn(
        session_id,
        command="bash",
        args=["-c", cmd],
    )
    process_id = spawn_result.process_id
    
    if verbose:
        print(f"  Process spawned: {process_id}", flush=True)
        print("-" * 60, flush=True)
    
    # Stream output in real-time
    output_chunks = []
    exit_code = -1
    
    try:
        # Get streaming iterator (with long timeout for HTTP connection)
        stream = await proc.stdout_stream(process_id, id=session_id, timeout=float(timeout + 120))
        
        deadline = time.time() + timeout
        
        async for event in stream:
            # Check timeout
            if time.time() > deadline:
                if verbose:
                    print("\n[TIMEOUT - killing process]", flush=True)
                try:
                    await proc.kill(process_id, id=session_id)
                except Exception:
                    pass
                break
            
            # Process exit event
            if hasattr(event, "event") and event.event == "exit":
                exit_code = getattr(event, "exit_code", 0) or 0
                break
            
            # Process data event - parse JSON stream events
            if hasattr(event, "data_b64") and event.data_b64:
                data = _decode_b64(event.data_b64)
                output_chunks.append(data)
                
                if verbose:
                    # Try to parse as JSON stream events for pretty output
                    _print_stream_event(data)
    
    except Exception as e:
        if verbose:
            print(f"\n[Stream error: {e}]", flush=True)
        # Try to get final status
        try:
            status = await proc.status(process_id, id=session_id)
            if hasattr(status, "exit_code") and status.exit_code is not None:
                exit_code = status.exit_code
        except Exception:
            pass
    
    if verbose:
        print("-" * 60, flush=True)
    
    output = "".join(output_chunks)

    elapsed = time.time() - start_time

    if verbose:
        print(f"  Claude finished in {elapsed:.1f}s (exit code: {exit_code})", flush=True)

    return RunResult(
        output=output,
        exit_code=exit_code,
        elapsed_seconds=elapsed,
        session_id=session_id,
    )


def _print_stream_event(data: str) -> None:
    """Parse and pretty-print Claude stream-json output."""
    import json
    
    # Split by lines and try to parse each as JSON
    for line in data.split('\n'):
        line = line.strip()
        if not line or line.startswith('[?') or line.startswith('\x1b['):
            continue
        
        try:
            event = json.loads(line)
            event_type = event.get("type", "")
            
            if event_type == "assistant":
                # Print assistant message content including tool_use
                content = event.get("message", {}).get("content", [])
                for c in content:
                    c_type = c.get("type", "")
                    if c_type == "text":
                        text = c.get("text", "").strip()
                        if text:
                            # Truncate long text
                            if len(text) > 200:
                                text = text[:200] + "..."
                            print(f"[assistant] {text}", flush=True)
                    elif c_type == "tool_use":
                        tool_name = c.get("name", "unknown")
                        tool_input = c.get("input", {})
                        # Show command for Bash tool - always show the actual command
                        if tool_name == "Bash":
                            cmd = tool_input.get("command", "")
                            desc = tool_input.get("description", "")
                            if cmd:
                                # Show description if present
                                if desc:
                                    print(f"[tool:Bash] {desc}", flush=True)
                                # Always show the actual command
                                # Truncate long commands but show more
                                display_cmd = cmd.replace('\n', ' ').strip()
                                if len(display_cmd) > 300:
                                    display_cmd = display_cmd[:300] + "..."
                                print(f"  $ {display_cmd}", flush=True)
                        else:
                            # Show tool name and brief input
                            input_str = str(tool_input)[:150]
                            print(f"[tool:{tool_name}] {input_str}", flush=True)
            
            elif event_type == "user":
                # Tool results
                content = event.get("message", {}).get("content", [])
                for c in content:
                    if c.get("type") == "tool_result":
                        result_content = c.get("content", "")
                        if isinstance(result_content, str) and result_content:
                            # Show truncated result
                            if len(result_content) > 200:
                                result_content = result_content[:200] + "..."
                            print(f"  -> {result_content}", flush=True)
            
            elif event_type == "result":
                result = event.get("result", "")
                subtype = event.get("subtype", "")
                if result:
                    print(f"[RESULT:{subtype}] {result[:300]}...", flush=True)
        
        except json.JSONDecodeError:
            # Not JSON, print raw (but skip terminal control sequences)
            if line and not line.startswith('[?'):
                print(line, flush=True)


def _decode_b64(s: str) -> str:
    """Decode a base64 string."""
    if not s:
        return ""
    try:
        return base64.b64decode(s).decode("utf-8", errors="replace")
    except Exception:
        return ""


async def run_task_in_kernel(
    task_prompt: str,
    skill_content: str,
    pool_name: str = DEFAULT_POOL_NAME,
    timeout: int = 120,
    api_key: str | None = None,
    anthropic_api_key: str | None = None,
    claude_model: str | None = None,
    interceptor_base_url: str | None = None,
    interceptor_auth_token: str | None = None,
) -> RunResult:
    """
    High-level function to run a task in a Kernel browser.

    This handles the full lifecycle:
    1. Acquire browser from pool
    2. Ensure tools are installed
    3. Write skill file
    4. Run Claude Code
    5. Release browser back to pool

    Args:
        task_prompt: The task for Claude to complete
        skill_content: The skill file content
        pool_name: Name of the browser pool to use
        timeout: Maximum seconds for Claude execution
        api_key: Kernel API key (uses env if not provided)
        anthropic_api_key: Anthropic API key (uses env if not provided)
        interceptor_base_url: If set, route inference through Synth interceptor
        interceptor_auth_token: Auth token for the interceptor

    Returns:
        RunResult with output, timing, and exit code
    """
    api_key = api_key or os.environ.get("KERNEL_API_KEY")
    anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("KERNEL_API_KEY not set")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Set a long HTTP timeout for Claude Code execution (can take 2+ minutes)
    import httpx
    client = AsyncKernel(api_key=api_key, timeout=httpx.Timeout(300.0))

    # Try pool first, fall back to direct browser creation
    use_pool = True
    session_id = None
    try:
        print(f"Acquiring browser from pool '{pool_name}'...")
        acquire_result = await client.browser_pools.acquire(pool_name)
        session_id = acquire_result.session_id
        print(f"  Acquired session: {session_id}")
        print(f"  Live view: {acquire_result.browser_live_view_url}")
    except Exception as e:
        print(f"  Pool unavailable ({e}), creating browser directly...")
        use_pool = False
        browser = await client.browsers.create(
            profile={"name": "linkedin"},
            stealth=True,
            timeout_seconds=max(timeout + 120, 600),
        )
        session_id = browser.session_id
        print(f"  Created session: {session_id}")
        print(f"  Live view: {browser.browser_live_view_url}")

    try:
        # Ensure browser is ready
        print("Ensuring browser is ready...")
        await ensure_browser_ready(client, session_id)

        # Write skill file
        print("Writing skill file...")
        await write_skill_file(client, session_id, skill_content)

        # Run Claude Code
        print(f"Running Claude Code (timeout: {timeout}s)...")
        result = await run_claude_code(
            client,
            session_id,
            task_prompt,
            anthropic_api_key,
            claude_model=claude_model,
            timeout=timeout,
            interceptor_base_url=interceptor_base_url,
            interceptor_auth_token=interceptor_auth_token,
        )

        return result

    finally:
        if use_pool:
            # Release browser back to pool with reuse=True to preserve VM state
            print(f"Releasing browser {session_id} back to pool...")
            try:
                await client.browser_pools.release(pool_name, session_id=session_id, reuse=True)
                print("  Released with reuse=True")
            except Exception as e:
                print(f"  Warning: Failed to release browser: {e}")
        else:
            # Delete the directly-created browser
            print(f"Deleting browser {session_id}...")
            try:
                import httpx as _hx
                _hx.delete(
                    f"https://api.onkernel.com/browsers/{session_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                print("  Deleted")
            except Exception as e:
                print(f"  Warning: cleanup failed: {e}")
