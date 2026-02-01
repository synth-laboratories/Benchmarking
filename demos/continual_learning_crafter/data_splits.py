#!/usr/bin/env python3
"""
Split definitions for Crafter continual learning.

We use progressively longer episodes and separate seed ranges to simulate
distribution shifts while staying within the same environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CrafterSplit:
    name: str
    description: str
    max_steps: int
    max_turns: int
    seed_start: int
    seed_count: int

    def seeds(self) -> List[int]:
        return list(range(self.seed_start, self.seed_start + self.seed_count))


CRAFTER_SPLITS: List[CrafterSplit] = [
    CrafterSplit(
        name="short",
        description="Short horizon episodes",
        max_steps=80,
        max_turns=20,
        seed_start=0,
        seed_count=20,
    ),
    CrafterSplit(
        name="medium",
        description="Mid-length episodes",
        max_steps=120,
        max_turns=30,
        seed_start=100,
        seed_count=20,
    ),
    CrafterSplit(
        name="long",
        description="Full horizon episodes",
        max_steps=200,
        max_turns=50,
        seed_start=200,
        seed_count=20,
    ),
]


def split_config(split: CrafterSplit) -> Dict[str, int | str]:
    return {
        "split": split.name,
        "max_steps_per_episode": split.max_steps,
        "max_turns": split.max_turns,
    }


def split_stats() -> Dict[str, Dict[str, int | str]]:
    return {
        split.name: {
            "description": split.description,
            "seed_count": split.seed_count,
            "seed_start": split.seed_start,
            "max_steps": split.max_steps,
            "max_turns": split.max_turns,
        }
        for split in CRAFTER_SPLITS
    }


if __name__ == "__main__":
    print("Crafter Continual Splits")
    print("=" * 60)
    for split in CRAFTER_SPLITS:
        print(
            f"{split.name:<6} | seeds {split.seed_start}-{split.seed_start + split.seed_count - 1} | "
            f"max_steps={split.max_steps} | max_turns={split.max_turns}"
        )
