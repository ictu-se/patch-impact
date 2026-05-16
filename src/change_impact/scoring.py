from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from .config import DEFAULT_RESULTS, DEFAULT_TASKS
from .io import read_jsonl, write_csv


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_path(value: Any) -> str:
    path = str(value or "").strip().strip("`'\"")
    path = path.split("::", 1)[0]
    path = path.replace("\\", "/")
    return re.sub(r"^[ab]/", "", path)


def list_field(parsed: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    value = parsed.get(key, [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("file") or item.get("path") or item.get("name") or ""
        normalized = normalize_path(item)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def path_hit(pred: str, gold: str) -> bool:
    pred = normalize_path(pred)
    gold = normalize_path(gold)
    return pred == gold or pred.endswith("/" + gold) or gold.endswith("/" + pred)


def recall_at(preds: list[str], golds: list[str], k: int) -> float | str:
    if not golds:
        return ""
    top = preds[:k]
    hits = sum(1 for gold in golds if any(path_hit(pred, gold) for pred in top))
    return hits / len(golds)


def precision_at(preds: list[str], golds: list[str], k: int) -> float:
    top = preds[:k]
    if not top:
        return 0.0
    hits = sum(1 for pred in top if any(path_hit(pred, gold) for gold in golds))
    return hits / len(top)


def tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", str(text or ""))}


def text_list(parsed: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    value = parsed.get(key, [])
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()] if isinstance(value, list) else []


def any_path_hit(preds: list[str], golds: list[str]) -> bool:
    return any(path_hit(pred, gold) for pred in preds for gold in golds)


def requirement_score(parsed: dict[str, Any] | None, task: dict[str, Any]) -> int:
    if not isinstance(parsed, dict):
        return 0
    summary = str(parsed.get("requirement_impact_summary", "")).strip()
    functionality = " ".join(text_list(parsed, "affected_functionality"))
    if not summary and not functionality:
        return 0
    impacted = list_field(parsed, "impacted_files")
    tests = list_field(parsed, "regression_test_focus")
    file_hit = any_path_hit(impacted, task.get("gold_affected_files", []))
    test_hit = any_path_hit(tests, task.get("gold_test_files", []))
    issue_overlap = len(tokens(summary + " " + functionality) & tokens(task.get("problem_statement", "")))
    patch_overlap = len(tokens(summary + " " + functionality) & tokens(task.get("patch_excerpt", "")))
    if (file_hit and test_hit) or (issue_overlap >= 4 and patch_overlap >= 3 and (file_hit or test_hit)):
        return 2
    if file_hit or test_hit or issue_overlap >= 4 or patch_overlap >= 3:
        return 1
    return 0


def risk_score(parsed: dict[str, Any] | None, task: dict[str, Any]) -> int:
    if not isinstance(parsed, dict):
        return 0
    risks = text_list(parsed, "risk_notes")
    if not risks:
        return 0
    risk_text = " ".join(risks)
    patch_text = task.get("patch_excerpt", "") + " " + task.get("test_patch_excerpt", "")
    patch_overlap = len(tokens(risk_text) & tokens(patch_text))
    issue_overlap = len(tokens(risk_text) & tokens(task.get("problem_statement", "")))
    if patch_overlap >= 4 or (patch_overlap >= 2 and issue_overlap >= 2):
        return 2
    if patch_overlap >= 1 or issue_overlap >= 2:
        return 1
    return 0


def score_outputs(outputs_jsonl: Path, tasks_path: Path, out_path: Path) -> None:
    tasks = {row["task_id"]: row for row in read_jsonl(tasks_path)}
    rows = []
    for record in read_jsonl(outputs_jsonl):
        task = tasks.get(record["task_id"], {})
        parsed = extract_json(record.get("stdout", ""))
        impacted = list_field(parsed, "impacted_files")
        tests = list_field(parsed, "regression_test_focus")
        risks = list_field(parsed, "risk_notes")
        summary = parsed.get("requirement_impact_summary", "") if isinstance(parsed, dict) else ""
        gold_files = task.get("gold_affected_files", [])
        gold_tests = task.get("gold_test_files", [])
        rows.append({
            "task_id": record.get("task_id", ""),
            "source": record.get("source", ""),
            "repo": record.get("repo", ""),
            "language": record.get("language", ""),
            "model": record.get("model", ""),
            "condition": record.get("condition", ""),
            "returncode": record.get("returncode", ""),
            "elapsed_sec": record.get("elapsed_sec", ""),
            "parse_ok": int(parsed is not None),
            "affected_file_recall_3": recall_at(impacted, gold_files, 3),
            "affected_file_recall_5": recall_at(impacted, gold_files, 5),
            "affected_file_precision_5": precision_at(impacted, gold_files, 5),
            "test_file_recall_3": recall_at(tests, gold_tests, 3),
            "test_file_recall_5": recall_at(tests, gold_tests, 5),
            "test_file_precision_5": precision_at(tests, gold_tests, 5),
            "predicted_files": len(impacted),
            "predicted_tests": len(tests),
            "risk_note_count": len(risks),
            "summary_words": len(re.findall(r"[A-Za-z0-9_]+", str(summary))),
            "gold_files": len(gold_files),
            "gold_tests": len(gold_tests),
        })
    fields = list(rows[0].keys()) if rows else []
    write_csv(out_path, rows, fields)
    print(f"wrote {len(rows)} metric rows to {out_path}")


def score_rubric(outputs_jsonl: Path, tasks_path: Path, out_path: Path) -> None:
    tasks = {row["task_id"]: row for row in read_jsonl(tasks_path)}
    rows = []
    for record in read_jsonl(outputs_jsonl):
        task = tasks.get(record["task_id"], {})
        parsed = extract_json(record.get("stdout", ""))
        rows.append({
            "task_id": record.get("task_id", ""),
            "source": record.get("source", ""),
            "repo": record.get("repo", ""),
            "language": record.get("language", ""),
            "model": record.get("model", ""),
            "condition": record.get("condition", ""),
            "requirement_impact_correctness": requirement_score(parsed, task),
            "risk_note_usefulness": risk_score(parsed, task),
        })
    fields = list(rows[0].keys()) if rows else []
    write_csv(out_path, rows, fields)
    print(f"wrote {len(rows)} rubric rows to {out_path}")


def add_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("outputs_jsonl", type=Path)
    parser.add_argument("--tasks", default=DEFAULT_TASKS, type=Path)
    parser.add_argument("--out", default=None, type=Path)


def run_score(args: argparse.Namespace) -> None:
    out = args.out or DEFAULT_RESULTS / (args.outputs_jsonl.stem + "_metrics.csv")
    score_outputs(args.outputs_jsonl, args.tasks, out)


def run_rubric(args: argparse.Namespace) -> None:
    out = args.out or DEFAULT_RESULTS / (args.outputs_jsonl.stem + "_rubric.csv")
    score_rubric(args.outputs_jsonl, args.tasks, out)
