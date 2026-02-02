"""
LLM-based verifier for browser automation task results.

Uses Claude to judge if the agent's output matches the expected result.
This supports natural language expectations, not just numeric comparisons.
"""

import json
import os
import re
from dataclasses import dataclass

import anthropic

from .tasks import Task


@dataclass
class VerificationResult:
    """Result from the LLM verifier."""

    correct: bool
    extracted_answer: str | None
    reason: str
    raw_score: float  # 1.0 if correct, 0.0 if not


async def verify_with_llm(
    task: Task,
    agent_output: str,
    api_key: str | None = None,
) -> VerificationResult:
    """
    Use Claude to verify if the agent's output matches the expected result.

    Args:
        task: The task that was executed
        agent_output: The full output from the agent
        api_key: Anthropic API key (uses env var if not provided)

    Returns:
        VerificationResult with correctness, extracted answer, and reason
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Support OpenAI fallback for verification when Anthropic credits are exhausted
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    prompt = f"""You are verifying a browser automation task result.

TASK: {task.prompt}

EXPECTED: {task.expected}

AGENT OUTPUT:
{agent_output[-8000:]}

---

Evaluate if the agent successfully completed the task and returned a result matching the expectation.

Instructions:
1. Look for an "ANSWER:" line in the agent output, or extract the final answer from context
2. Compare the extracted answer to the expected result
3. For numeric comparisons, accept values within the tolerance specified in the expected field
4. Be lenient with formatting (e.g., "39.8M" = "39,800,000" = "39800000")

Return ONLY a JSON object (no markdown, no explanation outside the JSON):
{{
  "correct": true or false,
  "extracted_answer": "the value the agent found (as a string)",
  "reason": "brief explanation of your judgment"
}}"""

    content = None
    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text.strip()
    except Exception as anthropic_err:
        # Fallback to OpenAI if Anthropic fails (e.g. credit exhaustion)
        if openai_api_key:
            print(f"  [VERIFIER] Anthropic failed ({anthropic_err}), falling back to OpenAI")
            try:
                import openai
                oai_client = openai.AsyncOpenAI(api_key=openai_api_key)
                oai_response = await oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = oai_response.choices[0].message.content.strip()
            except Exception as oai_err:
                return VerificationResult(
                    correct=False,
                    extracted_answer=None,
                    reason=f"Verification error (both Anthropic and OpenAI failed): {anthropic_err} / {oai_err}",
                    raw_score=0.0,
                )
        else:
            return VerificationResult(
                correct=False,
                extracted_answer=None,
                reason=f"Verification error: {anthropic_err}",
                raw_score=0.0,
            )

    try:
        # Try to extract JSON from the response
        # Handle cases where model might wrap in markdown code blocks
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback: try parsing the whole content
            result = json.loads(content)

        return VerificationResult(
            correct=result.get("correct", False),
            extracted_answer=result.get("extracted_answer"),
            reason=result.get("reason", "No reason provided"),
            raw_score=1.0 if result.get("correct", False) else 0.0,
        )

    except json.JSONDecodeError as e:
        return VerificationResult(
            correct=False,
            extracted_answer=None,
            reason=f"Failed to parse verifier response: {e}",
            raw_score=0.0,
        )
    except Exception as e:
        return VerificationResult(
            correct=False,
            extracted_answer=None,
            reason=f"Verification error: {e}",
            raw_score=0.0,
        )


def count_agent_steps(agent_output: str) -> int:
    """
    Count the number of agent steps/tool calls from the output.

    This is a heuristic based on common patterns in Claude Code output.
    """
    # Count occurrences of tool call patterns
    patterns = [
        r"agent-browser --cdp",  # agent-browser commands
        r"Running:",  # Claude Code tool execution
        r"Tool:",  # Tool call markers
        r"```bash\n",  # Bash code blocks (tool calls)
    ]

    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, agent_output))

    return max(1, total)  # At least 1 step


def count_sleep_commands(agent_output: str) -> tuple[int, int]:
    """
    Count sleep/wait commands and total sleep time in the agent output.
    
    Returns:
        Tuple of (number of sleep commands, total sleep milliseconds)
    """
    sleep_count = 0
    total_sleep_ms = 0
    
    # Pattern 1: agent-browser wait <ms>
    # Matches: agent-browser --cdp ... wait 2000
    wait_pattern = r"agent-browser[^\n]*wait\s+(\d+)"
    for match in re.finditer(wait_pattern, agent_output):
        sleep_count += 1
        total_sleep_ms += int(match.group(1))
    
    # Pattern 2: sleep <seconds>
    # Matches: sleep 2, sleep 5, etc.
    sleep_pattern = r"\bsleep\s+(\d+(?:\.\d+)?)"
    for match in re.finditer(sleep_pattern, agent_output):
        sleep_count += 1
        total_sleep_ms += int(float(match.group(1)) * 1000)
    
    return sleep_count, total_sleep_ms


def calculate_reward(
    correctness_score: float,
    elapsed_seconds: float,
    num_agent_steps: int,
    num_sleep_commands: int = 0,
    total_sleep_ms: int = 0,
    max_time: float = 120.0,
    max_steps: int = 20,
) -> float:
    """
    Calculate final reward with time, step, and sleep penalties.

    Reward = correctness * (1 - time_penalty - step_penalty - sleep_penalty)

    - Time penalty: 0.1 * (elapsed / max_time), capped at 0.2
    - Step penalty: 0.1 * (steps / max_steps), capped at 0.2
    - Sleep penalty: 0.02 per sleep command + 0.01 per 1000ms of sleep, capped at 0.15

    This encourages the agent to be correct, efficient, AND avoid arbitrary sleeps.

    Args:
        correctness_score: 0.0 or 1.0 from LLM judge
        elapsed_seconds: How long the task took
        num_agent_steps: Number of tool calls / steps the agent made
        num_sleep_commands: Number of sleep/wait commands used
        total_sleep_ms: Total milliseconds of explicit sleep/wait
        max_time: Reference time for penalty calculation
        max_steps: Reference step count for penalty calculation

    Returns:
        Final reward between 0.0 and 1.0
    """
    if correctness_score <= 0:
        return 0.0

    # Calculate penalties (capped individually)
    time_penalty = min(0.2, 0.1 * (elapsed_seconds / max_time))
    step_penalty = min(0.2, 0.1 * (num_agent_steps / max_steps))
    
    # Sleep penalty: penalize each sleep command and total sleep time
    # 0.02 per sleep command, 0.01 per second of sleep, capped at 0.15
    sleep_penalty = min(0.15, (num_sleep_commands * 0.02) + (total_sleep_ms / 1000 * 0.01))

    efficiency_multiplier = 1.0 - time_penalty - step_penalty - sleep_penalty
    return correctness_score * max(0.5, efficiency_multiplier)  # Floor at 0.5 if correct
