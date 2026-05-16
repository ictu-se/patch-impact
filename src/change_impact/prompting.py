from __future__ import annotations

from pathlib import Path
from typing import Any


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, task: dict[str, Any], condition: str) -> str:
    scoped = dict(task)
    if condition == "issue_only":
        scoped["patch_excerpt"] = ""
        scoped["test_patch_excerpt"] = ""
        scoped["repo_tree_excerpt"] = ""
    elif condition == "patch_only":
        scoped["problem_statement"] = ""
        scoped["repo_tree_excerpt"] = ""
    elif condition == "issue_plus_patch":
        scoped["repo_tree_excerpt"] = ""

    values = {
        "repo": scoped.get("repo", ""),
        "language": scoped.get("language", ""),
        "source": scoped.get("source", ""),
        "task_category": scoped.get("task_category", ""),
        "problem_statement": scoped.get("problem_statement", ""),
        "patch_excerpt": scoped.get("patch_excerpt", ""),
        "test_patch_excerpt": scoped.get("test_patch_excerpt", ""),
        "repo_tree_excerpt": scoped.get("repo_tree_excerpt", ""),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value or ""))
    return rendered
