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
import re


@dataclass
class Task:
    """A browser automation task to evaluate."""

    id: str
    prompt: str
    expected: str
    timeout: int = 120


# The LinkedIn corporate monitoring tasks for this benchmark

TASKS = []  # populated below


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


COMMENTER_TASKS = [
    {"company": "Stripe", "keyword": "AI"},
    {"company": "Datadog", "keyword": "observability"},
    {"company": "Notion", "keyword": "product"},
    {"company": "Figma", "keyword": "design"},
    {"company": "Canva", "keyword": "AI"},
    {"company": "Anthropic", "keyword": "research"},
    {"company": "OpenAI", "keyword": "safety"},
    {"company": "Snowflake", "keyword": "data cloud"},
    {"company": "HubSpot", "keyword": "AI"},
    {"company": "Shopify", "keyword": "commerce"},
]

REACTION_TASKS = [
    {"company": "Atlassian", "keyword": "developer"},
    {"company": "Slack", "keyword": "product"},
    {"company": "Zoom", "keyword": "AI"},
    {"company": "Asana", "keyword": "workflow"},
    {"company": "GitHub", "keyword": "Copilot"},
    {"company": "Adobe", "keyword": "design"},
    {"company": "Salesforce", "keyword": "AI"},
    {"company": "Amazon Web Services", "keyword": "cloud"},
    {"company": "Google Cloud", "keyword": "data"},
    {"company": "Microsoft Azure", "keyword": "security"},
]

EXEC_ENGAGEMENT_TASKS = [
    "Stripe",
    "Datadog",
    "Notion",
    "Figma",
    "Canva",
    "Anthropic",
    "OpenAI",
    "Snowflake",
    "HubSpot",
    "Shopify",
]

ENG_LEADERSHIP_TASKS = [
    "Notion",
    "Figma",
    "Canva",
    "Stripe",
    "Datadog",
    "OpenAI",
    "Anthropic",
    "GitHub",
    "Atlassian",
    "Shopify",
]

DEPARTURE_TASKS = [
    {"from": "Figma", "to": "Canva"},
    {"from": "Stripe", "to": "Block"},
    {"from": "Datadog", "to": "New Relic"},
    {"from": "Notion", "to": "Asana"},
    {"from": "Slack", "to": "Microsoft"},
    {"from": "Shopify", "to": "Amazon"},
    {"from": "HubSpot", "to": "Salesforce"},
    {"from": "Snowflake", "to": "Databricks"},
    {"from": "OpenAI", "to": "Anthropic"},
    {"from": "Canva", "to": "Adobe"},
]

NEW_HIRE_TASKS = [
    {"company": "Anthropic", "month": "January 2026"},
    {"company": "OpenAI", "month": "January 2026"},
    {"company": "Stripe", "month": "December 2025"},
    {"company": "Datadog", "month": "December 2025"},
    {"company": "Notion", "month": "November 2025"},
    {"company": "Figma", "month": "October 2025"},
    {"company": "Canva", "month": "September 2025"},
    {"company": "Snowflake", "month": "August 2025"},
    {"company": "HubSpot", "month": "July 2025"},
    {"company": "Shopify", "month": "June 2025"},
]

CONTENT_AUDIT_TASKS = [
    "Stripe",
    "Datadog",
    "Notion",
    "Figma",
    "Canva",
    "Anthropic",
    "OpenAI",
    "Snowflake",
    "HubSpot",
    "Shopify",
]

HASHTAG_TASKS = [
    "Stripe",
    "Datadog",
    "Notion",
    "Figma",
    "Canva",
    "Anthropic",
    "OpenAI",
    "Snowflake",
    "HubSpot",
    "Shopify",
]

SHARED_CONNECTION_TASKS = [
    "Stripe",
    "Datadog",
    "Notion",
    "Figma",
    "Canva",
    "Anthropic",
    "OpenAI",
    "Snowflake",
    "HubSpot",
    "Shopify",
]

TOP_FOLLOWED_TASKS = [
    "Stripe",
    "Datadog",
    "Notion",
    "Figma",
    "Canva",
    "Anthropic",
    "OpenAI",
    "Snowflake",
    "HubSpot",
    "Shopify",
]


TASKS = []

for spec in COMMENTER_TASKS:
    company = spec["company"]
    keyword = spec["keyword"]
    TASKS.append(
        Task(
            id=f"commenters_{_slugify(company)}_{_slugify(keyword)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to find {company}'s most recent "
                f"LinkedIn post that mentions '{keyword}'. List all commenters and their job "
                "titles. Return a bullet list formatted as 'Name — Title'."
            ),
            expected=(
                f"A list of commenter names with job titles from {company}'s most recent "
                f"post that mentions '{keyword}'. Each entry should include a person's name "
                "and title."
            ),
            timeout=600,
        )
    )

for spec in REACTION_TASKS:
    company = spec["company"]
    keyword = spec["keyword"]
    TASKS.append(
        Task(
            id=f"reactors_{_slugify(company)}_{_slugify(keyword)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to find {company}'s most recent "
                f"post about '{keyword}'. Count how many reactors have Engineering vs Sales "
                "vs Marketing in their titles. Return counts in the format: 'Engineering: N, "
                "Sales: N, Marketing: N'."
            ),
            expected=(
                "Three numeric counts labeled Engineering, Sales, and Marketing based on the "
                f"reactor list for the chosen {company} post about '{keyword}'."
            ),
            timeout=600,
        )
    )

for company in EXEC_ENGAGEMENT_TASKS:
    TASKS.append(
        Task(
            id=f"exec_engagement_{_slugify(company)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to scan {company}'s posts from "
                "the last 30 days. Find every post where a commenter has 'VP', 'Director', "
                "'Chief', or 'C-' in their title. Return a list of post URLs with the "
                "executive commenter name and title."
            ),
            expected=(
                f"A list of {company} post URLs from the last 30 days with the corresponding "
                "executive commenter names/titles (VP/Director/C-level)."
            ),
            timeout=600,
        )
    )

for company in ENG_LEADERSHIP_TASKS:
    TASKS.append(
        Task(
            id=f"eng_leadership_{_slugify(company)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to list current {company} employees "
                "who have 'Engineering Manager' or 'Director of Engineering' in their title. "
                "Return a list of names with titles."
            ),
            expected=(
                f"Names and titles of current {company} employees whose titles include "
                "'Engineering Manager' or 'Director of Engineering'."
            ),
            timeout=600,
        )
    )

for spec in DEPARTURE_TASKS:
    from_company = spec["from"]
    to_company = spec["to"]
    TASKS.append(
        Task(
            id=f"departures_{_slugify(from_company)}_to_{_slugify(to_company)}_2024",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to find five people who list "
                f"{from_company} as a past employer and joined {to_company} in 2024. Return each "
                "person's name and current title at the new company."
            ),
            expected=(
                f"Five names of former {from_company} employees now at {to_company}, each with their "
                "current title. Profiles should indicate a 2024 start date."
            ),
            timeout=600,
        )
    )

for spec in NEW_HIRE_TASKS:
    company = spec["company"]
    month = spec["month"]
    TASKS.append(
        Task(
            id=f"product_hires_{_slugify(company)}_{_slugify(month)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to list people who started at "
                f"{company} in {month} with 'Product' in their title. Return names and titles."
            ),
            expected=(
                f"Names and titles of {company} employees with 'Product' in their title whose "
                f"start month is {month}."
            ),
            timeout=600,
        )
    )

for company in CONTENT_AUDIT_TASKS:
    TASKS.append(
        Task(
            id=f"top_posts_{_slugify(company)}_last_60_days",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to list {company}'s five most engaged "
                "posts in the last 60 days, ranked by reaction count. Include reaction counts and "
                "a short topic summary for each post."
            ),
            expected=(
                f"An ordered list of five {company} posts from the last 60 days with reaction "
                "counts and brief topic summaries."
            ),
            timeout=600,
        )
    )

for company in HASHTAG_TASKS:
    TASKS.append(
        Task(
            id=f"hashtag_frequency_{_slugify(company)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to review {company}'s last 20 posts and "
                "count how many times each hashtag appears: #AI, #hiring, #product. Return counts "
                "per hashtag."
            ),
            expected=(
                f"A mapping of #AI, #hiring, and #product to counts across {company}'s last 20 posts."
            ),
            timeout=600,
        )
    )

for company in SHARED_CONNECTION_TASKS:
    TASKS.append(
        Task(
            id=f"shared_connections_{_slugify(company)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to find how many {company} employees are "
                "2nd-degree connections to you. Return the count and list names of shared connections "
                "if visible."
            ),
            expected=(
                f"A count of {company} employees that are 2nd-degree connections, plus any shared "
                "connection names surfaced by LinkedIn."
            ),
            timeout=600,
        )
    )

for company in TOP_FOLLOWED_TASKS:
    TASKS.append(
        Task(
            id=f"top_followed_{_slugify(company)}",
            prompt=(
                f"Use your LinkedIn skill with agent-browser to identify the three {company} employees "
                "with the highest personal follower counts. Return their names and follower counts."
            ),
            expected=(
                f"Three {company} employees with the highest follower counts and their counts."
            ),
            timeout=600,
        )
    )


# Map task IDs to tasks for easy lookup
TASK_BY_ID = {task.id: task for task in TASKS}



TASK_BY_ID = {task.id: task for task in TASKS}


def get_task_by_seed(seed: int) -> Task:
    """Get a task by seed number (wraps around)."""
    return TASKS[seed % len(TASKS)]


def get_task_by_id(task_id: str) -> Task:
    """Get a task by its ID."""
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown task ID: {task_id}. Available: {list(TASK_BY_ID.keys())}")
    return TASK_BY_ID[task_id]
