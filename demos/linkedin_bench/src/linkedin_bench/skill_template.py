"""
LinkedIn skill template loader for GEPA optimization.

The initial skill lives in skills/linkedin.com/SKILL.md so it can be
easily diffed against optimized outputs from GEPA.

GEPA will optimize this skill content to improve reliability and speed.
"""

from pathlib import Path

# Path to the skill file relative to the project root
SKILL_FILE = Path(__file__).parent.parent.parent / "skills" / "linkedin.com" / "SKILL.md"


def get_skill_content() -> str:
    """
    Return the current skill template content.
    
    Reads from skills/linkedin.com/SKILL.md so you can:
    - Edit the skill directly as markdown
    - Diff the initial skill against GEPA-optimized versions
    """
    if SKILL_FILE.exists():
        return SKILL_FILE.read_text()
    else:
        raise FileNotFoundError(
            f"Skill file not found: {SKILL_FILE}\n"
            "Make sure skills/linkedin.com/SKILL.md exists."
        )


def get_skill_path() -> Path:
    """Return the path to the skill file."""
    return SKILL_FILE
