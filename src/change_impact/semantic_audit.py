#!/usr/bin/env python3
"""Blind, repeated LLM-as-a-judge semantic audit for the revision.

The sample is paired across the three defensible evidence conditions, balanced
across the ten generators, and includes one task from each language per model.
"""

import argparse
import csv
import json
import random
import re
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from .config import DEFAULT_TASKS

CONDITIONS = ("issue_only", "patch_only", "issue_plus_patch")
JUDGES = ("deepseek-coder:6.7b", "mistral:7b")
METRICS = (
    "functionality_correctness",
    "requirement_summary_correctness",
    "risk_usefulness",
    "unsupported_claim_severity",
    "overall_triage_utility",
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def valid_judgment(value):
    return isinstance(value, dict) and all(
        isinstance(value.get(key), int) and 0 <= value[key] <= 2 for key in METRICS
    )


def call_ollama(model, prompt, endpoint, timeout=180):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.9, "num_predict": 240},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode(errors="replace"))
    return data.get("response", ""), round(time.time() - started, 3)


def judge_prompt(task, candidate):
    return f"""You are independently auditing a software change report. Do not infer quality from model identity or input condition; neither is shown. Judge only against the issue and diffs below.

Rubric (integer scores only):
- functionality_correctness: 0 incorrect/unrelated; 1 partly correct or vague; 2 specific and supported.
- requirement_summary_correctness: 0 incorrect/unsupported; 1 captures part of the intent; 2 accurately captures the behavioral requirement.
- risk_usefulness: 0 misleading/unsupported; 1 plausible but generic or incomplete; 2 specific, actionable, and grounded in the change.
- unsupported_claim_severity: 0 none; 1 minor speculation; 2 material fabricated behavior, component, or consequence.
- overall_triage_utility: 0 unusable; 1 useful with verification; 2 useful and substantially evidence-grounded.

Calibration anchors: empty/unrelated prose receives correctness and utility 0; a paraphrase supported by issue and changed code can receive 2 without copying words; plausible but non-evidenced risk claims receive risk 1 and unsupported severity at least 1. File names alone do not establish semantic correctness.

Return exactly one JSON object with those five keys plus "rationale" (maximum 35 words). No Markdown.

ISSUE:
{task['problem_statement']}

PRODUCTION DIFF:
{task['patch_excerpt']}

TEST DIFF:
{task['test_patch_excerpt']}

CANDIDATE REPORT:
{candidate}
"""


def select_items(tasks, outputs):
    by_language = defaultdict(list)
    for task in tasks:
        by_language[task["language"]].append(task)
    for language in by_language:
        by_language[language].sort(key=lambda task: task["task_id"])
    models = sorted({record["model"] for record in outputs})
    output_index = {
        (record["model"], record["condition"], record["task_id"]): record
        for record in outputs
    }
    items = []
    languages = ("Java", "JavaScript", "Python", "TypeScript")
    for model_index, model in enumerate(models):
        for language_index, language in enumerate(languages):
            group = by_language[language]
            task = group[(model_index * 3 + language_index) % len(group)]
            pair_id = f"P{model_index:02d}-{language_index}"
            for condition in CONDITIONS:
                record = output_index[(model, condition, task["task_id"])]
                items.append(
                    {
                        "pair_id": pair_id,
                        "model": model,
                        "condition": condition,
                        "task": task,
                        "candidate": record.get("stdout", ""),
                    }
                )
    return items


def weighted_kappa(a, b):
    n = len(a)
    if not n:
        return 0.0
    observed = sum(((x - y) ** 2) / 4 for x, y in zip(a, b)) / n
    ca = [a.count(i) / n for i in range(3)]
    cb = [b.count(i) / n for i in range(3)]
    expected = sum(
        ca[i] * cb[j] * ((i - j) ** 2) / 4 for i in range(3) for j in range(3)
    )
    return 1 - observed / expected if expected else 1.0


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def bootstrap_mean(values, reps=5000, seed=20260716):
    rng = random.Random(seed)
    draws = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(reps)]
    return sum(values) / len(values), percentile(draws, 0.025), percentile(draws, 0.975)


def cluster_bootstrap(rows, metric, cluster="pair_id", reps=5000, seed=20260716):
    groups = defaultdict(list)
    for row in rows:
        groups[row[cluster]].append(row[metric])
    keys = sorted(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        selected = [rng.choice(keys) for _ in keys]
        values = [value for key in selected for value in groups[key]]
        draws.append(sum(values) / len(values))
    observed = [row[metric] for row in rows]
    return (
        sum(observed) / len(observed),
        percentile(draws, 0.025),
        percentile(draws, 0.975),
    )


def run_semantic_audit(args: argparse.Namespace) -> None:
    tasks = read_jsonl(args.tasks)
    outputs = [
        record
        for path in args.results.glob("*_outputs.jsonl")
        for record in read_jsonl(path)
        if record["condition"] in CONDITIONS
    ]
    items = select_items(tasks, outputs)
    raw_path = args.out / "semantic_judgments.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.resume and raw_path.exists():
        for row in read_jsonl(raw_path):
            if row.get("parse_ok") and valid_judgment(row.get("judgment")):
                completed[
                    (row["judge"], row["model"], row["condition"], row["task_id"])
                ] = row

    jobs = []
    judges_to_run = (args.run_judge,) if args.run_judge else JUDGES
    for item in items:
        for judge in judges_to_run:
            key = (judge, item["model"], item["condition"], item["task"]["task_id"])
            if key not in completed:
                jobs.append((judge, item))
    lock = threading.Lock()

    def run(job):
        judge, item = job
        response, elapsed = call_ollama(
            judge,
            judge_prompt(item["task"], item["candidate"]),
            args.ollama_url,
            args.timeout,
        )
        parsed = extract_json(response)
        if not valid_judgment(parsed):
            retry_prompt = f"""Your prior audit response used a missing or non-integer score. Convert it to exactly one JSON object with integer values 0, 1, or 2 for functionality_correctness, requirement_summary_correctness, risk_usefulness, unsupported_claim_severity, and overall_triage_utility, plus a rationale of at most 35 words. Never use decimals: choose 1 for useful with verification and 2 for substantially grounded. Preserve your prior assessment; do not add commentary.\n\nPRIOR RESPONSE:\n{response}"""
            response, retry_elapsed = call_ollama(
                judge, retry_prompt, args.ollama_url, args.timeout
            )
            elapsed += retry_elapsed
            parsed = extract_json(response)
        return {
            "pair_id": item["pair_id"],
            "task_id": item["task"]["task_id"],
            "language": item["task"]["language"],
            "model": item["model"],
            "condition": item["condition"],
            "judge": judge,
            "elapsed_sec": elapsed,
            "parse_ok": int(parsed is not None),
            "judgment": parsed,
            "raw": response,
        }

    mode = "a" if completed else "w"
    with (
        raw_path.open(mode, encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [pool.submit(run, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            if count % 10 == 0 or count == len(futures):
                print(f"completed {count}/{len(futures)} new judgments", flush=True)

    raw_rows = read_jsonl(raw_path)
    selected_keys = {
        (judge, item["model"], item["condition"], item["task"]["task_id"])
        for item in items
        for judge in JUDGES
    }
    raw_rows = [
        row
        for row in raw_rows
        if (row["judge"], row["model"], row["condition"], row["task_id"])
        in selected_keys
    ]
    # Interrupted/resumed executions may complete an in-flight duplicate. Keep
    # one judgment for each stable evaluation key.
    deduplicated = {}
    for row in raw_rows:
        key = (row["judge"], row["model"], row["condition"], row["task_id"])
        deduplicated[key] = row
    rows = list(deduplicated.values())
    metrics = METRICS
    clean = []
    for row in rows:
        judgment = row.get("judgment") or {}
        if not all(
            isinstance(judgment.get(key), int) and 0 <= judgment[key] <= 2
            for key in metrics
        ):
            continue
        clean.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "pair_id",
                        "task_id",
                        "language",
                        "model",
                        "condition",
                        "judge",
                    )
                },
                **{key: judgment[key] for key in metrics},
                "rationale": str(judgment.get("rationale", "")).strip(),
            }
        )
    with (args.out / "semantic_judgments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(clean)

    summary = []
    for condition in CONDITIONS:
        group = [row for row in clean if row["condition"] == condition]
        result = {
            "condition": condition,
            "outputs": len({(r["model"], r["task_id"]) for r in group}),
            "judgments": len(group),
        }
        for metric in metrics:
            mean, low, high = cluster_bootstrap(group, metric)
            result[metric] = mean
            result[metric + "_ci_low"] = low
            result[metric + "_ci_high"] = high
        summary.append(result)
    with (args.out / "semantic_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    agreements = []
    index = defaultdict(dict)
    for row in clean:
        index[(row["model"], row["condition"], row["task_id"])][row["judge"]] = row
    for metric in metrics:
        paired = [
            judges for judges in index.values() if all(j in judges for j in JUDGES)
        ]
        a = [judges[JUDGES[0]][metric] for judges in paired]
        b = [judges[JUDGES[1]][metric] for judges in paired]
        agreements.append(
            {
                "metric": metric,
                "pairs": len(paired),
                "exact_agreement": sum(x == y for x, y in zip(a, b)) / len(a),
                "quadratic_weighted_kappa": weighted_kappa(a, b),
            }
        )
    with (args.out / "semantic_agreement.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(agreements[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(agreements)

    semantic_differences = []
    clean_index = {
        (row["judge"], row["pair_id"], row["condition"]): row for row in clean
    }
    for metric in metrics:
        differences = []
        for judge in JUDGES:
            for pair_id in sorted({row["pair_id"] for row in clean}):
                patch = clean_index.get((judge, pair_id, "patch_only"))
                combined = clean_index.get((judge, pair_id, "issue_plus_patch"))
                if patch and combined:
                    differences.append(
                        {
                            "pair_id": pair_id,
                            "difference": combined[metric] - patch[metric],
                        }
                    )
        mean, low, high = cluster_bootstrap(differences, "difference")
        semantic_differences.append(
            {
                "comparison": "issue_plus_patch - patch_only",
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    with (args.out / "semantic_paired_differences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(semantic_differences[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(semantic_differences)

    judge_sensitivity = []
    for judge in JUDGES:
        for metric in metrics:
            differences = []
            for pair_id in sorted({row["pair_id"] for row in clean}):
                patch = clean_index.get((judge, pair_id, "patch_only"))
                combined = clean_index.get((judge, pair_id, "issue_plus_patch"))
                if patch and combined:
                    differences.append(
                        {
                            "pair_id": pair_id,
                            "difference": combined[metric] - patch[metric],
                        }
                    )
            mean, low, high = cluster_bootstrap(differences, "difference")
            judge_sensitivity.append(
                {
                    "judge": judge,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    with (args.out / "semantic_judge_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(judge_sensitivity[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(judge_sensitivity)
    print(f"usable judgments: {len(clean)}/{len(rows)}")


def add_semantic_audit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("revision_results"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-judge",
        choices=JUDGES,
        default=None,
        help="Run one judge; final agreement still requires both judges' stored results.",
    )
