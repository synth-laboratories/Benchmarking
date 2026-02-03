#!/usr/bin/env python3
"""
Build gold answers from Sonnet rollouts (majority vote).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rollouts(path: Path):
    records = []
    if not path.exists():
        raise SystemExit(f"Rollouts file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Build gold answers from rollouts")
    parser.add_argument("--rollouts", type=str, default="data/sonnet45_runs/rollouts.jsonl")
    parser.add_argument("--output", type=str, default="data/gold_answers.json")
    parser.add_argument("--report", type=str, default="data/gold_report.json")
    args = parser.parse_args()

    rollouts_path = Path(args.rollouts)
    records = load_rollouts(rollouts_path)

    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("task_id")].append(record)

    gold = {}
    report = {"no_consensus": {}, "total_tasks": len(grouped)}

    for task_id, runs in grouped.items():
        normalized = [r.get("normalized_answer") for r in runs if r.get("normalized_answer")]
        counts = Counter(normalized)
        if not counts:
            report["no_consensus"][task_id] = {"reason": "no_normalized_answers", "runs": runs}
            continue
        best, freq = counts.most_common(1)[0]
        if freq < 2:
            report["no_consensus"][task_id] = {"reason": "no_majority", "runs": runs}
            continue
        # pick the first run with matching normalized answer
        chosen = None
        for r in runs:
            if r.get("normalized_answer") == best:
                chosen = r.get("extracted_answer")
                break
        gold[task_id] = chosen

    Path(args.output).write_text(json.dumps(gold, indent=2, ensure_ascii=False))
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Gold answers written: {args.output}")
    print(f"No-consensus tasks: {len(report['no_consensus'])}")


if __name__ == "__main__":
    main()
