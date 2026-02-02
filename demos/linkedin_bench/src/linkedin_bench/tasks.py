"""
Task definitions for the LinkedIn corporate monitoring benchmark.

These tasks mirror the "corporate monitoring / research" task set from the
planning notes. Each task has:
- id: Unique identifier
- prompt: What the agent should do
- expected: Natural language description of expected result (for LLM judge)
- timeout: Maximum seconds allowed for the task
"""

from dataclasses import dataclass


@dataclass
class Task:
    """A browser automation task to evaluate."""

    id: str
    prompt: str
    expected: str
    timeout: int = 120


# The LinkedIn corporate monitoring tasks for this benchmark
TASKS = [
    Task(
        id="commenters_stripe_ai_post",
        prompt=(
            "Use your LinkedIn skill with agent-browser to find Stripe's most recent "
            "LinkedIn post that mentions 'AI'. List all commenters and their job titles. "
            "Return a bullet list formatted as 'Name — Title'."
        ),
        expected=(
            "A list of commenter names with job titles from Stripe's most recent AI-related "
            "post. Each entry should include a person's name and title."
        ),
        timeout=600,
    ),
    Task(
        id="reactors_datadog_observability",
        prompt=(
            "Use your LinkedIn skill with agent-browser to find Datadog's most recent "
            "post about observability (look for the keyword 'observability' in the post). "
            "Count how many reactors have Engineering vs Sales vs Marketing in their "
            "titles. Return counts in the format: 'Engineering: N, Sales: N, Marketing: N'."
        ),
        expected=(
            "Three numeric counts labeled Engineering, Sales, and Marketing based on the "
            "reactor list for the chosen Datadog observability post."
        ),
        timeout=600,
    ),
    Task(
        id="exec_engagement_recent_posts",
        prompt=(
            "Use your LinkedIn skill with agent-browser to scan Datadog's posts from the "
            "last 30 days. Find every post where a commenter has 'VP', 'Director', "
            "'Chief', or 'C-' in their title. Return a list of post URLs with the "
            "executive commenter name and title."
        ),
        expected=(
            "A list of Datadog post URLs from the last 30 days with the corresponding "
            "executive commenter names/titles (VP/Director/C-level)."
        ),
        timeout=600,
    ),
    Task(
        id="notion_eng_leadership_census",
        prompt=(
            "Use your LinkedIn skill with agent-browser to list current Notion employees "
            "who have 'Engineering Manager' or 'Director of Engineering' in their title. "
            "Return a list of names with titles."
        ),
        expected=(
            "Names and titles of current Notion employees whose titles include "
            "'Engineering Manager' or 'Director of Engineering'."
        ),
        timeout=600,
    ),
    Task(
        id="figma_to_canva_departures_2024",
        prompt=(
            "Use your LinkedIn skill with agent-browser to find five people who list "
            "Figma as a past employer and joined Canva in 2024. Return each person's "
            "name and current title at Canva."
        ),
        expected=(
            "Five names of former Figma employees now at Canva, each with their current "
            "Canva title. Profiles should indicate the 2024 start date."
        ),
        timeout=600,
    ),
    Task(
        id="anthropic_product_new_hires_jan_2026",
        prompt=(
            "Use your LinkedIn skill with agent-browser to list people who started at "
            "Anthropic in January 2026 with 'Product' in their title. Return names and "
            "titles."
        ),
        expected=(
            "Names and titles of Anthropic employees with 'Product' in their title whose "
            "start month is January 2026."
        ),
        timeout=600,
    ),
    Task(
        id="anthropic_top_posts_dec_2024",
        prompt=(
            "Use your LinkedIn skill with agent-browser to list Anthropic's five most "
            "engaged posts in December 2024, ranked by reaction count. Include reaction "
            "counts and a short topic summary for each post."
        ),
        expected=(
            "An ordered list of five Anthropic posts from December 2024 with reaction "
            "counts and brief topic summaries."
        ),
        timeout=600,
    ),
    Task(
        id="notion_hashtag_frequency",
        prompt=(
            "Use your LinkedIn skill with agent-browser to review Notion's last 20 posts "
            "and count how many times each hashtag appears: #AI, #hiring, #product. "
            "Return counts per hashtag."
        ),
        expected=(
            "A mapping of #AI, #hiring, and #product to counts across Notion's last 20 posts."
        ),
        timeout=600,
    ),
    Task(
        id="snowflake_shared_connections",
        prompt=(
            "Use your LinkedIn skill with agent-browser to find how many Snowflake "
            "employees are 2nd-degree connections to you. Return the count and list "
            "names of shared connections if visible."
        ),
        expected=(
            "A count of Snowflake employees that are 2nd-degree connections, plus any "
            "shared connection names surfaced by LinkedIn."
        ),
        timeout=600,
    ),
    Task(
        id="hubspot_top_followed_employees",
        prompt=(
            "Use your LinkedIn skill with agent-browser to identify the three HubSpot "
            "employees with the highest personal follower counts. Return their names "
            "and follower counts."
        ),
        expected=(
            "Three HubSpot employees with the highest follower counts and their counts."
        ),
        timeout=600,
    ),
]

# Map task IDs to tasks for easy lookup
TASK_BY_ID = {task.id: task for task in TASKS}


def get_task_by_seed(seed: int) -> Task:
    """Get a task by seed number (wraps around)."""
    return TASKS[seed % len(TASKS)]


def get_task_by_id(task_id: str) -> Task:
    """Get a task by its ID."""
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown task ID: {task_id}. Available: {list(TASK_BY_ID.keys())}")
    return TASK_BY_ID[task_id]
