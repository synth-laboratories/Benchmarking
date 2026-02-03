#!/usr/bin/env python3
"""
Prepare LinkedIn Bench dataset for Hugging Face.

Exports all task definitions to a format suitable for uploading to Hugging Face Hub.

Usage:
    python prepare_huggingface_dataset.py
    python prepare_huggingface_dataset.py --output-dir ./hf_dataset
    python prepare_huggingface_dataset.py --push-to-hub synth-laboratories/linkedin-bench

Note: This exports task definitions only, not gold answers (which require authenticated
LinkedIn access to generate). Gold answers can be generated using:
    python scripts/run_sonnet_rollouts.py --runs-per-task 3
    python scripts/build_gold_answers.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Import tasks from the benchmark
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.linkedin_bench.tasks import TASKS, Task


def task_to_dict(task: Task, index: int) -> dict:
    """Convert a Task to a dictionary for the dataset."""
    # Determine category from task ID prefix
    category_map = {
        "commenters_": "commenters",
        "reactors_": "reactions",
        "exec_engagement_": "executive_engagement",
        "eng_leadership_": "engineering_census",
        "departures_": "departures",
        "new_hires_": "new_hires",
        "top_posts_": "top_posts",
        "hashtag_": "hashtag_frequency",
        "shared_connections_": "shared_connections",
        "top_followed_": "top_followed",
    }
    
    category = "other"
    for prefix, cat in category_map.items():
        if task.id.startswith(prefix):
            category = cat
            break
    
    return {
        "id": task.id,
        "index": index,
        "prompt": task.prompt,
        "expected": task.expected,
        "timeout": task.timeout,
        "category": category,
    }


def prepare_dataset() -> dict:
    """
    Prepare the LinkedIn Bench dataset.
    
    Schema:
    - id: str - Unique task identifier
    - index: int - Task index (for seed mapping)
    - prompt: str - What the agent should do
    - expected: str - Natural language description of expected result
    - timeout: int - Maximum seconds allowed
    - category: str - Task category
    """
    
    entries = [task_to_dict(task, i) for i, task in enumerate(TASKS)]
    
    # Count by category
    category_counts = {}
    for entry in entries:
        cat = entry["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return {
        "version": "1.0.0",
        "description": "LinkedIn Bench: Corporate monitoring benchmark for browser automation agents",
        "license": "MIT",
        "homepage": "https://github.com/synth-laboratories/Benchmarking",
        "citation": "",
        "splits": {
            "test": entries,
        },
        "metadata": {
            "total_tasks": len(entries),
            "categories": list(category_counts.keys()),
            "category_counts": category_counts,
            "default_timeout": 120,
        },
    }


def write_jsonl(entries: list[dict], output_path: Path) -> None:
    """Write entries as JSONL file."""
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def write_dataset(dataset: dict, output_dir: Path) -> None:
    """Write dataset to disk in Hugging Face format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    metadata_path = output_dir / "dataset_info.json"
    metadata = {
        "version": dataset["version"],
        "description": dataset["description"],
        "license": dataset["license"],
        "homepage": dataset["homepage"],
        "citation": dataset["citation"],
        "metadata": dataset["metadata"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    
    # Write data splits
    for split_name, entries in dataset["splits"].items():
        split_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(entries, split_path)
        print(f"  Wrote {len(entries)} entries to {split_path}")
    
    # Write README
    readme_content = f"""# LinkedIn Bench

{dataset["description"]}

## Dataset Description

LinkedIn Bench evaluates browser automation agents on corporate monitoring tasks.
Each task requires navigating LinkedIn, extracting information, and returning structured answers.

### Task Categories

| Category | Description | Tasks |
|----------|-------------|-------|
| **commenters** | Extract commenters from keyword-matched posts | {dataset["metadata"]["category_counts"].get("commenters", 0)} |
| **reactions** | Breakdown reactor titles by role | {dataset["metadata"]["category_counts"].get("reactions", 0)} |
| **executive_engagement** | Track executive activity | {dataset["metadata"]["category_counts"].get("executive_engagement", 0)} |
| **engineering_census** | Count engineering leadership | {dataset["metadata"]["category_counts"].get("engineering_census", 0)} |
| **departures** | Track employee moves between companies | {dataset["metadata"]["category_counts"].get("departures", 0)} |
| **new_hires** | Monitor hiring by role and month | {dataset["metadata"]["category_counts"].get("new_hires", 0)} |
| **top_posts** | Find highest engagement posts | {dataset["metadata"]["category_counts"].get("top_posts", 0)} |
| **hashtag_frequency** | Analyze hashtag usage | {dataset["metadata"]["category_counts"].get("hashtag_frequency", 0)} |
| **shared_connections** | Count 2nd-degree connections | {dataset["metadata"]["category_counts"].get("shared_connections", 0)} |
| **top_followed** | Find most-followed employees | {dataset["metadata"]["category_counts"].get("top_followed", 0)} |

## Data Fields

- `id`: Unique task identifier
- `index`: Task index for seed mapping
- `prompt`: Natural language instruction for the agent
- `expected`: Description of expected result (for LLM judge)
- `timeout`: Maximum seconds allowed (default: 120)
- `category`: Task category

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("synth-laboratories/linkedin-bench")

# Iterate over test tasks
for task in dataset["test"]:
    print(f"Task: {{task['id']}}")
    print(f"Prompt: {{task['prompt']}}")
    print(f"Expected: {{task['expected']}}")
```

## Reproduction

Running this benchmark requires:
1. A [Kernel](https://kernel.sh) API key for cloud browsers
2. A LinkedIn account authenticated in a Kernel profile
3. An Anthropic API key for the LLM judge

```bash
git clone https://github.com/synth-laboratories/Benchmarking
cd Benchmarking/demos/linkedin_bench
uv sync

# Create authenticated profile
kernel profiles create --name linkedin
kernel browsers create --profile-name linkedin --save-changes
# Log in to LinkedIn in the browser, then close

# Create browser pool
kernel browser-pools create --name agent-gepa --profile-name linkedin --size 10

# Build gold answers (optional, for accurate scoring)
uv run python scripts/run_sonnet_rollouts.py --runs-per-task 3
uv run python scripts/build_gold_answers.py

# Run evaluation
uv run python run_eval.py --all
```

## Reward Function

The benchmark uses a composite reward:

```python
reward = correctness * (1.0 - time_penalty - step_penalty - sleep_penalty)
```

Where:
- `correctness` = 1.0 if LLM judge says correct, 0.0 otherwise
- `time_penalty` = min(0.2, 0.1 * elapsed / max_time)
- `step_penalty` = min(0.2, 0.1 * steps / max_steps)
- `sleep_penalty` = min(0.15, 0.02 * num_sleeps + 0.01 * sleep_seconds)

This encourages efficient automation without arbitrary delays.

## Important Notes

- **Authentication required**: Tasks require a logged-in LinkedIn session
- **Rate limiting**: LinkedIn may rate-limit automated access
- **Terms of Service**: Ensure compliance with LinkedIn's ToS for your use case
- **Gold answers**: Change over time as LinkedIn data updates

## License

MIT

## Citation

```bibtex
@misc{{linkedin-bench-2026,
  title={{LinkedIn Bench: A Corporate Monitoring Benchmark for Browser Automation Agents}},
  author={{Synth Laboratories}},
  year={{2026}},
  url={{https://github.com/synth-laboratories/Benchmarking}}
}}
```
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"  Wrote README to {readme_path}")


def push_to_hub(output_dir: Path, repo_id: str) -> None:
    """Push dataset to Hugging Face Hub."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: Install datasets and huggingface_hub to push to Hub")
        print("  pip install datasets huggingface_hub")
        return
    
    print(f"\nPushing to Hugging Face Hub: {repo_id}")
    
    # Load the JSONL as a dataset
    dataset = load_dataset(
        "json",
        data_files={"test": str(output_dir / "test.jsonl")},
    )
    
    # Push to hub
    dataset.push_to_hub(repo_id, private=False)
    
    # Also upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(output_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"  Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LinkedIn Bench for Hugging Face")
    parser.add_argument("--output-dir", type=str, default="./hf_linkedin_bench", help="Output directory")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Push to HF Hub (e.g., synth-laboratories/linkedin-bench)")
    args = parser.parse_args()
    
    print("Preparing dataset...")
    dataset = prepare_dataset()
    
    print(f"\nDataset summary:")
    print(f"  Total tasks: {dataset['metadata']['total_tasks']}")
    print(f"  Categories: {len(dataset['metadata']['categories'])}")
    for cat, count in sorted(dataset['metadata']['category_counts'].items()):
        print(f"    {cat}: {count}")
    
    output_dir = Path(args.output_dir)
    print(f"\nWriting dataset to {output_dir}...")
    write_dataset(dataset, output_dir)
    
    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
