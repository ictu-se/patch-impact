#!/usr/bin/env python3
"""Run leakage-aware path, ceiling, language, and paired-effect analyses."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from change_impact.analysis import cluster_bootstrap, extract_json_object, read_jsonl, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = PROJECT_ROOT / "data/change_impact_tasks.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = PROJECT_ROOT / "revision_results"


def changed_files(diff: str | None) -> list[str]:
    matches = re.findall(r"^diff --git a/(.*?) b/(.*?)$", diff or "", re.M)
    return list(dict.fromkeys(target for _, target in matches))


def normalize(path: Any) -> str:
    path = str(path or "").strip().strip("`'\"").replace("\\", "/")
    path = path.split("::", 1)[0]
    return re.sub(r"^[ab]/", "", path)


def hit(pred: str, gold: str) -> bool:
    pred, gold = normalize(pred), normalize(gold)
    return pred == gold or pred.endswith("/" + gold) or gold.endswith("/" + pred)


def recall(predictions: list[str], gold_paths: list[str], k: int | None = None) -> float:
    if not gold_paths:
        return 0.0
    selected = predictions if k is None else predictions[:k]
    return sum(any(hit(prediction, gold) for prediction in selected) for gold in gold_paths) / len(
        gold_paths
    )


def precision(predictions: list[str], gold_paths: list[str], k: int | None = None) -> float:
    selected = predictions if k is None else predictions[:k]
    if not selected:
        return 0.0
    hits = sum(any(hit(prediction, gold) for gold in gold_paths) for prediction in selected)
    return hits / len(selected)


def paths(obj: dict[str, Any] | None, key: str) -> list[str]:
    if not obj:
        return []
    values = obj.get(key, [])
    values = [values] if isinstance(values, str) else values
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("path") or value.get("file") or value.get("name") or ""
        value = normalize(value)
        if value and value not in out:
            out.append(value)
    return out


def repo_key(task: dict[str, Any]) -> str:
    if task.get("repo"):
        return task["repo"]
    # SWE-bench IDs have owner__repository-instance form.
    stem = task["task_id"].rsplit("-", 1)[0]
    return stem.replace("__", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks)
    if not tasks:
        raise ValueError(f"No tasks found in {args.tasks}")
    task_map = {task["task_id"]: task for task in tasks}

    sample_rows = []
    for task in tasks:
        fg, tg = task["gold_affected_files"], task["gold_test_files"]
        fp = changed_files(task["patch_excerpt"])
        tp = changed_files(task["test_patch_excerpt"])
        sample_rows.append(
            {
                "task_id": task["task_id"],
                "source": task["source"],
                "language": task["language"],
                "repo": repo_key(task),
                "gold_files": len(fg),
                "gold_tests": len(tg),
                "file_ceiling_r5": min(5, len(fg)) / len(fg),
                "test_ceiling_r5": min(5, len(tg)) / len(tg),
                "parser_file_r5": recall(fp, fg, 5),
                "parser_file_p5": precision(fp, fg, 5),
                "parser_file_recall_all": recall(fp, fg),
                "parser_test_r5": recall(tp, tg, 5),
                "parser_test_p5": precision(tp, tg, 5),
                "parser_test_recall_all": recall(tp, tg),
            }
        )
    write_csv(args.output / "parser_baseline_task.csv", sample_rows)

    language_rows = []
    for language in sorted({row["language"] for row in sample_rows}):
        group = [row for row in sample_rows if row["language"] == language]
        result = {"language": language, "tasks": len(group)}
        for key in (
            "gold_files",
            "gold_tests",
            "file_ceiling_r5",
            "test_ceiling_r5",
            "parser_file_r5",
            "parser_file_recall_all",
            "parser_test_r5",
            "parser_test_recall_all",
        ):
            result[key] = sum(row[key] for row in group) / len(group)
        language_rows.append(result)
    write_csv(args.output / "language_and_ceiling.csv", language_rows)

    llm_rows = []
    result_paths = sorted(args.results.glob("*_outputs.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"No generator outputs found in {args.results}")
    for path in result_paths:
        for record in read_jsonl(path):
            if record["condition"] == "issue_plus_patch_plus_tree":
                continue
            task = task_map[record["task_id"]]
            obj = extract_json_object(record.get("stdout", ""))
            predicted_files = paths(obj, "impacted_files")
            predicted_tests = paths(obj, "regression_test_focus")
            file_r5 = recall(predicted_files, task["gold_affected_files"], 5)
            test_r5 = recall(predicted_tests, task["gold_test_files"], 5)
            llm_rows.append(
                {
                    "task_id": record["task_id"],
                    "repo": repo_key(task),
                    "source": task["source"],
                    "language": task["language"],
                    "model": record["model"],
                    "condition": record["condition"],
                    "parse_ok": int(obj is not None),
                    "file_r5": file_r5,
                    "file_ceiling_normalized": min(
                        1.0,
                        file_r5
                        / (
                            min(5, len(task["gold_affected_files"]))
                            / len(task["gold_affected_files"])
                        ),
                    ),
                    "file_p5": precision(predicted_files, task["gold_affected_files"], 5),
                    "test_r5": test_r5,
                    "test_p5": precision(predicted_tests, task["gold_test_files"], 5),
                    "prompt_tokens": int(record.get("prompt_eval_count") or 0),
                    "output_tokens": int(record.get("eval_count") or 0),
                    "elapsed_sec": float(record.get("elapsed_sec") or 0),
                    "returncode": int(record.get("returncode") or 0),
                }
            )
    write_csv(args.output / "llm_task_metrics.csv", llm_rows)

    compact_groups = defaultdict(list)
    for row in llm_rows:
        compact_groups[
            (row["task_id"], row["repo"], row["source"], row["language"], row["condition"])
        ].append(row)
    compact_rows = []
    for (task_id, repo, source, language, condition), group in sorted(compact_groups.items()):
        compact = {
            "task_id": task_id,
            "repo": repo,
            "source": source,
            "language": language,
            "condition": condition,
            "models": len(group),
        }
        for key in (
            "parse_ok",
            "file_r5",
            "file_ceiling_normalized",
            "file_p5",
            "test_r5",
            "test_p5",
        ):
            compact[key] = sum(row[key] for row in group) / len(group)
        compact_rows.append(compact)
    write_csv(args.output / "llm_task_condition_means.csv", compact_rows)

    condition_rows = []
    parser_metric_map = {
        "file_r5": "parser_file_r5",
        "file_p5": "parser_file_p5",
        "test_r5": "parser_test_r5",
        "test_p5": "parser_test_p5",
    }
    parser_result = {
        "condition": "diff_parser",
        "models": 0,
        "tasks": len(sample_rows),
        "generations": 0,
        "parse_ok": 1.0,
        "parse_ok_ci_low": 1.0,
        "parse_ok_ci_high": 1.0,
        "file_ceiling_normalized": "",
        "file_ceiling_normalized_ci_low": "",
        "file_ceiling_normalized_ci_high": "",
    }
    for output_key, input_key in parser_metric_map.items():
        mean, low, high = cluster_bootstrap(
            sample_rows,
            input_key,
            cluster_key="repo",
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        parser_result[output_key] = mean
        parser_result[output_key + "_ci_low"] = low
        parser_result[output_key + "_ci_high"] = high
    condition_rows.append(parser_result)
    for condition in ("issue_only", "patch_only", "issue_plus_patch"):
        group = [row for row in llm_rows if row["condition"] == condition]
        result = {
            "condition": condition,
            "models": len({row["model"] for row in group}),
            "tasks": len({row["task_id"] for row in group}),
            "generations": len(group),
        }
        for key in (
            "parse_ok",
            "file_r5",
            "file_ceiling_normalized",
            "file_p5",
            "test_r5",
            "test_p5",
        ):
            mean, low, high = cluster_bootstrap(
                group,
                key,
                cluster_key="repo",
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
            result[key] = mean
            result[key + "_ci_low"] = low
            result[key + "_ci_high"] = high
        condition_rows.append(result)
    write_csv(args.output / "condition_cluster_ci.csv", condition_rows)

    diagnostic_rows = []
    for condition in ("issue_only", "patch_only", "issue_plus_patch"):
        group = [row for row in llm_rows if row["condition"] == condition]
        diagnostic_rows.append(
            {
                "condition": condition,
                "generations": len(group),
                "prompt_tokens_mean": sum(row["prompt_tokens"] for row in group) / len(group),
                "prompt_tokens_min": min(row["prompt_tokens"] for row in group),
                "prompt_tokens_max": max(row["prompt_tokens"] for row in group),
                "output_tokens_mean": sum(row["output_tokens"] for row in group) / len(group),
                "output_tokens_max": max(row["output_tokens"] for row in group),
                "failed_calls": sum(row["returncode"] != 0 for row in group),
            }
        )
    write_csv(args.output / "generation_diagnostics.csv", diagnostic_rows)

    llm_language_rows = []
    for language in sorted({row["language"] for row in llm_rows}):
        for condition in ("issue_only", "patch_only", "issue_plus_patch"):
            group = [
                row
                for row in llm_rows
                if row["language"] == language and row["condition"] == condition
            ]
            result = {"language": language, "condition": condition, "generations": len(group)}
            for key in ("file_r5", "file_ceiling_normalized", "file_p5", "test_r5", "test_p5"):
                result[key] = sum(row[key] for row in group) / len(group)
            llm_language_rows.append(result)
    write_csv(args.output / "llm_metrics_by_language.csv", llm_language_rows)

    # Paired issue+patch minus patch-only differences retain model/task matching.
    index = {(r["model"], r["task_id"], r["condition"]): r for r in llm_rows}
    differences = []
    for key, patch in index.items():
        model, task_id, condition = key
        if condition != "patch_only":
            continue
        combined_key = (model, task_id, "issue_plus_patch")
        if combined_key not in index:
            raise ValueError(f"Missing paired issue_plus_patch output for {model} on {task_id}")
        combined = index[combined_key]
        differences.append(
            {
                "model": model,
                "repo": patch["repo"],
                "file_delta": combined["file_r5"] - patch["file_r5"],
                "test_delta": combined["test_r5"] - patch["test_r5"],
            }
        )
    diff_rows = []
    for key in ("file_delta", "test_delta"):
        mean, low, high = cluster_bootstrap(
            differences,
            key,
            cluster_key="repo",
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        diff_rows.append(
            {
                "comparison": "issue_plus_patch - patch_only",
                "metric": key,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    write_csv(args.output / "paired_condition_differences.csv", diff_rows)

    model_diff_rows = []
    for model in sorted({row["model"] for row in differences}):
        group = [row for row in differences if row["model"] == model]
        model_diff_rows.append(
            {
                "model": model,
                "file_delta": sum(row["file_delta"] for row in group) / len(group),
                "test_delta": sum(row["test_delta"] for row in group) / len(group),
            }
        )
    write_csv(args.output / "model_paired_differences.csv", model_diff_rows)

    summary = {
        "tasks": len(tasks),
        "repositories": len({repo_key(t) for t in tasks}),
        "source_counts": Counter(t["source"] for t in tasks),
        "language_counts": Counter(t["language"] for t in tasks),
        "seed": 20260515,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "design_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
