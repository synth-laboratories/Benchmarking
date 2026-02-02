"""
Analyze GEPA interceptor traces to detect skill usage patterns.

Usage:
    python scripts/analyze_traces.py /tmp/gepa_dev3.log

Fetches traces from the interceptor and checks:
1. Was the skill invoked via [tool:Skill]?
2. Was the skill content actually injected into the conversation?
3. Did the agent follow the skill's instructions (bash commands match skill patterns)?
4. How many LLM turns before/after skill invocation?

This can be integrated into GEPA as a trace-level quality signal when skills are configured.
"""

import json
import re
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TraceAnalysis:
    trace_id: str
    task: str
    candidate: str
    num_messages: int = 0
    skill_invoked: bool = False
    skill_content_injected: bool = False
    skill_args: list = field(default_factory=list)
    bash_commands: list = field(default_factory=list)
    skill_patterns_followed: list = field(default_factory=list)
    turns_before_skill: int = 0
    turns_after_skill: int = 0
    final_reward: float = 0.0

    @property
    def skill_used_correctly(self) -> bool:
        return self.skill_invoked and self.skill_content_injected and len(self.bash_commands) > 0

    @property
    def efficiency_score(self) -> float:
        """Lower is better - how many turns were wasted before invoking the skill."""
        if not self.skill_invoked:
            return 0.0
        # Ideal: skill invoked on turn 1 (turns_before_skill == 1)
        return max(0.0, 1.0 - (self.turns_before_skill - 1) * 0.2)


# Known patterns from the LinkedIn skill
SKILL_BASH_PATTERNS = [
    r"agent-browser.*--cdp.*eval.*window\.location\.href",
    r"agent-browser.*--cdp.*open.*linkedin\.com",
    r"agent-browser.*--cdp.*eval.*document\.body\.innerText",
    r"agent-browser.*--cdp.*snapshot",
    r"sleep\s+\d+",
]


def analyze_trace(data: dict, task: str, candidate: str, trace_id: str) -> TraceAnalysis:
    """Analyze a single interceptor trace for skill usage."""
    analysis = TraceAnalysis(trace_id=trace_id, task=task, candidate=candidate)

    msgs = data.get("request", {}).get("body", {}).get("messages", [])
    analysis.num_messages = len(msgs)

    skill_seen_at_turn = None
    turn = 0

    for msg in msgs:
        role = msg.get("role", "?")
        content_items = msg.get("content", [])
        if isinstance(content_items, str):
            content_items = [{"type": "text", "text": content_items}]

        if role == "assistant":
            turn += 1

        for item in content_items:
            if not isinstance(item, dict):
                continue

            # Detect tool_use
            if item.get("type") == "tool_use":
                name = item.get("name", "")
                inp = item.get("input", {})

                if name == "Skill":
                    analysis.skill_invoked = True
                    analysis.skill_args.append(inp.get("args", ""))
                    if skill_seen_at_turn is None:
                        skill_seen_at_turn = turn

                elif name == "Bash":
                    cmd = inp.get("command", "")
                    analysis.bash_commands.append(cmd)

                    # Check if command matches skill patterns
                    for pattern in SKILL_BASH_PATTERNS:
                        if re.search(pattern, cmd):
                            analysis.skill_patterns_followed.append(pattern)
                            break

            # Detect skill content injection
            text = item.get("text", "") or item.get("content", "")
            if isinstance(text, str) and "Base directory for this skill" in text:
                analysis.skill_content_injected = True

    if skill_seen_at_turn is not None:
        analysis.turns_before_skill = skill_seen_at_turn
        analysis.turns_after_skill = turn - skill_seen_at_turn

    return analysis


def fetch_trace(base_url: str, trace_id: str):
    """Fetch a trace from the interceptor."""
    url = f"{base_url}/api/interceptor/v1/trace/by-correlation/{trace_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.load(resp)
    except Exception as e:
        print(f"  Failed to fetch {trace_id}: {e}", file=sys.stderr)
        return None


def parse_log(log_path: str):
    """Parse GEPA log to extract trace_id -> (task, candidate, reward) mappings."""
    with open(log_path) as f:
        content = f.read()

    entries = []
    current_cand = None
    current_task = None
    current_trace = None

    for line in content.split("\n"):
        m = re.search(r"cand_([a-f0-9]+)_\d+/trace_([a-f0-9]+)", line)
        if m:
            current_cand = f"cand_{m.group(1)}"
            current_trace = f"trace_{m.group(2)}"

        m = re.search(r"Rollout: seed=\d+, task=(\w+)", line)
        if m:
            current_task = m.group(1)

        m = re.search(r"Final reward: ([0-9.]+)", line)
        if m:
            reward = float(m.group(1))
            if current_trace and current_task and current_cand:
                entries.append({
                    "trace_id": current_trace,
                    "task": current_task,
                    "candidate": current_cand,
                    "reward": reward,
                })

    return entries


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gepa_log_path> [interceptor_base_url]")
        sys.exit(1)

    log_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "https://infra-api-dev.usesynth.ai"

    entries = parse_log(log_path)
    print(f"Found {len(entries)} rollouts with rewards in log\n")

    # Sample: up to 5 per task type
    by_task = defaultdict(list)
    for e in entries:
        if len(by_task[e["task"]]) < 5:
            by_task[e["task"]].append(e)

    analyses = []
    for task, task_entries in by_task.items():
        for entry in task_entries:
            data = fetch_trace(base_url, entry["trace_id"])
            if data:
                a = analyze_trace(data, entry["task"], entry["candidate"], entry["trace_id"])
                a.final_reward = entry["reward"]
                analyses.append(a)

    # Summary
    print(f"\n{'='*80}")
    print("TRACE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Traces analyzed: {len(analyses)}")
    print(f"Skill invoked:           {sum(1 for a in analyses if a.skill_invoked)}/{len(analyses)}")
    print(f"Skill content injected:  {sum(1 for a in analyses if a.skill_content_injected)}/{len(analyses)}")
    print(f"Skill used correctly:    {sum(1 for a in analyses if a.skill_used_correctly)}/{len(analyses)}")
    print()

    # Per-task breakdown
    for task in sorted(by_task.keys()):
        task_analyses = [a for a in analyses if a.task == task]
        if not task_analyses:
            continue
        print(f"\n--- {task} ---")
        for a in task_analyses:
            status = "OK" if a.skill_used_correctly else "MISSING"
            patterns = len(a.skill_patterns_followed)
            print(
                f"  [{status}] {a.trace_id[:30]}... "
                f"reward={a.final_reward:.3f} "
                f"skill_turn={a.turns_before_skill} "
                f"bash_cmds={len(a.bash_commands)} "
                f"patterns={patterns}"
            )

    # Correlation: skill usage vs reward
    used = [a for a in analyses if a.skill_used_correctly]
    not_used = [a for a in analyses if not a.skill_used_correctly]
    if used:
        print(f"\nAvg reward (skill used correctly): {sum(a.final_reward for a in used)/len(used):.3f} (n={len(used)})")
    if not_used:
        print(f"Avg reward (skill NOT used):       {sum(a.final_reward for a in not_used)/len(not_used):.3f} (n={len(not_used)})")

    # Efficiency
    if used:
        avg_eff = sum(a.efficiency_score for a in used) / len(used)
        avg_turns_before = sum(a.turns_before_skill for a in used) / len(used)
        print(f"\nAvg turns before skill invocation: {avg_turns_before:.1f}")
        print(f"Avg efficiency score: {avg_eff:.2f}")


if __name__ == "__main__":
    main()
