from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import DEFAULT_MANIFEST, DEFAULT_TASKS
from .io import read_csv, write_csv, write_jsonl


def changed_files(diff_text: str) -> list[str]:
    if not isinstance(diff_text, str):
        return []
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff_text, re.M):
        path = match.group(2).strip()
        if path and path not in files:
            files.append(path)
    return files


def path_tree(paths: list[str]) -> str:
    tree: dict[str, Any] = {}
    for path in paths:
        node = tree
        for part in [part for part in path.split("/") if part]:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def walk(node: dict[str, Any], prefix: str = "", depth: int = 0) -> None:
        if depth > 4:
            return
        for name in sorted(node):
            lines.append(f"{prefix}{name}")
            walk(node[name], prefix + "  ", depth + 1)

    walk(tree)
    return "\n".join(lines)


def parse_json_list(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def clip(text: str, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def task_from_polybench(row: dict[str, str]) -> dict[str, Any] | None:
    patch = row.get("patch", "")
    test_patch = row.get("test_patch", "")
    files = changed_files(patch)
    test_files = changed_files(test_patch)
    if not files or not test_files:
        return None
    return {
        "task_id": row["instance_id"],
        "source": "SWE-PolyBench",
        "repo": row.get("repo", ""),
        "language": row.get("language", ""),
        "task_category": row.get("task_category", ""),
        "problem_statement": row.get("problem_statement", ""),
        "patch_excerpt": clip(patch, 6000),
        "test_patch_excerpt": clip(test_patch, 5000),
        "repo_tree_excerpt": path_tree(files + test_files),
        "gold_affected_files": files,
        "gold_test_files": test_files,
    }


def task_from_swebench(row: dict[str, str]) -> dict[str, Any] | None:
    patch = row.get("patch", "")
    test_patch = row.get("test_patch", "")
    files = changed_files(patch)
    test_files = changed_files(test_patch)
    fail_to_pass = parse_json_list(row.get("FAIL_TO_PASS", ""))
    if not test_files:
        test_files = sorted(
            {item.split("::", 1)[0] for item in fail_to_pass if "::" in item or "/" in item}
        )
    if not files or not test_files:
        return None
    return {
        "task_id": row["instance_id"],
        "source": "SWE-bench Lite",
        "repo": row.get("repo", ""),
        "language": "Python",
        "task_category": "bugfix",
        "problem_statement": row.get("problem_statement", ""),
        "patch_excerpt": clip(patch, 6000),
        "test_patch_excerpt": clip(test_patch, 5000),
        "repo_tree_excerpt": path_tree(files + test_files),
        "gold_affected_files": files,
        "gold_test_files": test_files,
    }


def build_tasks(poly_path: Path, swe_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_csv(manifest_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_csv(poly_path):
        task = task_from_polybench(row)
        if task:
            by_id[task["task_id"]] = task
    for row in read_csv(swe_path):
        task = task_from_swebench(row)
        if task:
            by_id[task["task_id"]] = task

    missing = [row["task_id"] for row in manifest if row["task_id"] not in by_id]
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(
            f"{len(missing)} manifest tasks were not found in benchmark CSVs: {sample}"
        )
    return [by_id[row["task_id"]] for row in sorted(manifest, key=lambda row: int(row["order"]))]


def write_task_summary(path: Path, tasks: list[dict[str, Any]]) -> None:
    fields = [
        "task_id",
        "source",
        "repo",
        "language",
        "task_category",
        "affected_files",
        "test_files",
    ]
    rows = [
        {
            "task_id": task["task_id"],
            "source": task["source"],
            "repo": task["repo"],
            "language": task["language"],
            "task_category": task["task_category"],
            "affected_files": len(task["gold_affected_files"]),
            "test_files": len(task["gold_test_files"]),
        }
        for task in tasks
    ]
    write_csv(path, rows, fields)


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--poly", required=True, type=Path, help="Path to SWE-PolyBench CSV export."
    )
    parser.add_argument(
        "--swe", required=True, type=Path, help="Path to SWE-bench Lite CSV export."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--out", default=DEFAULT_TASKS, type=Path)
    parser.add_argument("--summary", default=Path("data/change_impact_sample.csv"), type=Path)


def run_prepare(args: argparse.Namespace) -> None:
    tasks = build_tasks(args.poly, args.swe, args.manifest)
    write_jsonl(args.out, tasks)
    write_task_summary(args.summary, tasks)
    print(f"wrote {len(tasks)} tasks to {args.out}")
    print(f"wrote task summary to {args.summary}")
