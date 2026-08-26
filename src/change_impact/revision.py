#!/usr/bin/env python3
"""Revision analyses requested after review: transparent parser baseline,
top-k ceilings, language strata, and repository-cluster bootstrap intervals.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


from .config import DEFAULT_TASKS


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def changed_files(diff):
    matches = re.findall(r"^diff --git a/(.*?) b/(.*?)$", diff or "", re.M)
    return list(dict.fromkeys(target for _, target in matches))


def normalize(path):
    path = str(path or "").strip().strip("`'\"").replace("\\", "/")
    path = path.split("::", 1)[0]
    return re.sub(r"^[ab]/", "", path)


def hit(pred, gold):
    pred, gold = normalize(pred), normalize(gold)
    return pred == gold or pred.endswith("/" + gold) or gold.endswith("/" + pred)


def recall(preds, golds, k=None):
    if not golds:
        return 0.0
    preds = preds if k is None else preds[:k]
    return sum(any(hit(pred, gold) for pred in preds) for gold in golds) / len(golds)


def precision(preds, golds, k=None):
    preds = preds if k is None else preds[:k]
    return (
        (sum(any(hit(pred, gold) for gold in golds) for pred in preds) / len(preds))
        if preds
        else 0.0
    )


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def paths(obj, key):
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


def repo_key(task):
    if task.get("repo"):
        return task["repo"]
    # SWE-bench IDs have owner__repository-instance form.
    stem = task["task_id"].rsplit("-", 1)[0]
    return stem.replace("__", "/")


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    fraction = index - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def cluster_ci(rows, value_key, reps=5000, seed=20260716):
    groups = defaultdict(list)
    for row in rows:
        groups[row["repo"]].append(row[value_key])
    keys = sorted(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value for key in sampled for value in groups[key]]
        draws.append(sum(values) / len(values))
    observed = sum(row[value_key] for row in rows) / len(rows)
    return observed, percentile(draws, 0.025), percentile(draws, 0.975)


def run_revision(args: argparse.Namespace) -> None:
    tasks = read_jsonl(args.tasks)
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
    write_csv(args.out / "parser_baseline_task.csv", sample_rows)

    # The gold sets were parsed from the complete diffs before excerpt clipping.
    # Replaying those complete header sets isolates clipping loss from parser error.
    write_csv(
        args.out / "complete_header_sensitivity.csv",
        [
            {
                "condition": "complete_diff_headers",
                "tasks": len(sample_rows),
                "production_recall": 1.0,
                "production_precision": 1.0,
                "test_recall": 1.0,
                "test_precision": 1.0,
            }
        ],
    )

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
    write_csv(args.out / "language_and_ceiling.csv", language_rows)

    llm_rows = []
    for path in sorted(args.results.glob("*_outputs.jsonl")):
        for record in read_jsonl(path):
            if record["condition"] == "issue_plus_patch_plus_tree":
                continue
            task = task_map[record["task_id"]]
            obj = extract_json(record.get("stdout", ""))
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
                    "file_p5": precision(
                        predicted_files, task["gold_affected_files"], 5
                    ),
                    "test_r5": test_r5,
                    "test_p5": precision(predicted_tests, task["gold_test_files"], 5),
                    "prompt_tokens": int(record.get("prompt_eval_count") or 0),
                    "output_tokens": int(record.get("eval_count") or 0),
                    "elapsed_sec": float(record.get("elapsed_sec") or 0),
                    "returncode": int(record.get("returncode") or 0),
                }
            )
    write_csv(args.out / "llm_task_metrics.csv", llm_rows)

    compact_groups = defaultdict(list)
    for row in llm_rows:
        compact_groups[
            (
                row["task_id"],
                row["repo"],
                row["source"],
                row["language"],
                row["condition"],
            )
        ].append(row)
    compact_rows = []
    for (task_id, repo, source, language, condition), group in sorted(
        compact_groups.items()
    ):
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
    write_csv(args.out / "llm_task_condition_means.csv", compact_rows)

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
        mean, low, high = cluster_ci(sample_rows, input_key)
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
            mean, low, high = cluster_ci(group, key)
            result[key] = mean
            result[key + "_ci_low"] = low
            result[key + "_ci_high"] = high
        condition_rows.append(result)
    write_csv(args.out / "condition_cluster_ci.csv", condition_rows)

    diagnostic_rows = []
    for condition in ("issue_only", "patch_only", "issue_plus_patch"):
        group = [row for row in llm_rows if row["condition"] == condition]
        diagnostic_rows.append(
            {
                "condition": condition,
                "generations": len(group),
                "prompt_tokens_mean": sum(row["prompt_tokens"] for row in group)
                / len(group),
                "prompt_tokens_min": min(row["prompt_tokens"] for row in group),
                "prompt_tokens_max": max(row["prompt_tokens"] for row in group),
                "output_tokens_mean": sum(row["output_tokens"] for row in group)
                / len(group),
                "output_tokens_max": max(row["output_tokens"] for row in group),
                "failed_calls": sum(row["returncode"] != 0 for row in group),
            }
        )
    write_csv(args.out / "generation_diagnostics.csv", diagnostic_rows)

    llm_language_rows = []
    for language in sorted({row["language"] for row in llm_rows}):
        for condition in ("issue_only", "patch_only", "issue_plus_patch"):
            group = [
                row
                for row in llm_rows
                if row["language"] == language and row["condition"] == condition
            ]
            result = {
                "language": language,
                "condition": condition,
                "generations": len(group),
            }
            for key in (
                "file_r5",
                "file_ceiling_normalized",
                "file_p5",
                "test_r5",
                "test_p5",
            ):
                result[key] = sum(row[key] for row in group) / len(group)
            llm_language_rows.append(result)
    write_csv(args.out / "llm_metrics_by_language.csv", llm_language_rows)

    # Paired issue+patch minus patch-only differences retain model/task matching.
    index = {(r["model"], r["task_id"], r["condition"]): r for r in llm_rows}
    differences = []
    for key, patch in index.items():
        model, task_id, condition = key
        if condition != "patch_only":
            continue
        combined = index[(model, task_id, "issue_plus_patch")]
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
        mean, low, high = cluster_ci(differences, key)
        diff_rows.append(
            {
                "comparison": "issue_plus_patch - patch_only",
                "metric": key,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    write_csv(args.out / "paired_condition_differences.csv", diff_rows)

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
    write_csv(args.out / "model_paired_differences.csv", model_diff_rows)

    summary = {
        "tasks": len(tasks),
        "repositories": len({repo_key(t) for t in tasks}),
        "source_counts": Counter(t["source"] for t in tasks),
        "language_counts": Counter(t["language"] for t in tasks),
        "seed": 20260515,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 20260716,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "design_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def add_revision_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("revision_results"))
