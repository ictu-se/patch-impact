#!/usr/bin/env python3
"""Blind, repeated LLM-as-a-judge semantic audit for the revision.

The sample is paired across the three defensible evidence conditions, balanced
across the ten generators, and includes one task from each language per model.
"""

import argparse
import json
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from change_impact.analysis import cluster_bootstrap, extract_json_object, read_jsonl, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = PROJECT_ROOT / "data/change_impact_tasks.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = PROJECT_ROOT / "revision_results"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
CONDITIONS = ("issue_only", "patch_only", "issue_plus_patch")
JUDGES = ("deepseek-coder:6.7b", "mistral:7b")
METRICS = (
    "functionality_correctness",
    "requirement_summary_correctness",
    "risk_usefulness",
    "unsupported_claim_severity",
    "overall_triage_utility",
)


def valid_judgment(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(key), int) and 0 <= value[key] <= 2 for key in METRICS
    )


def call_ollama(
    model: str,
    prompt: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 180,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 0.9, "num_predict": 240},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode(errors="replace"))
    return data.get("response", ""), round(time.time() - started, 3)


def judge_prompt(task: dict[str, Any], candidate: str) -> str:
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
{task["problem_statement"]}

PRODUCTION DIFF:
{task["patch_excerpt"]}

TEST DIFF:
{task["test_patch_excerpt"]}

CANDIDATE REPORT:
{candidate}
"""


def select_items(
    tasks: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_language = defaultdict(list)
    for task in tasks:
        by_language[task["language"]].append(task)
    for language in by_language:
        by_language[language].sort(key=lambda task: task["task_id"])
    models = sorted({record["model"] for record in outputs})
    output_index = {
        (record["model"], record["condition"], record["task_id"]): record for record in outputs
    }
    items = []
    languages = ("Java", "JavaScript", "Python", "TypeScript")
    for model_index, model in enumerate(models):
        for language_index, language in enumerate(languages):
            group = by_language.get(language, [])
            if not group:
                raise ValueError(f"No tasks available for required language {language}")
            task = group[(model_index * 3 + language_index) % len(group)]
            pair_id = f"P{model_index:02d}-{language_index}"
            for condition in CONDITIONS:
                output_key = (model, condition, task["task_id"])
                if output_key not in output_index:
                    raise ValueError(
                        "Missing generator output for "
                        f"model={model}, condition={condition}, task={task['task_id']}"
                    )
                record = output_index[output_key]
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


def weighted_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return 0.0
    observed = sum(((x - y) ** 2) / 4 for x, y in zip(a, b)) / n
    ca = [a.count(i) / n for i in range(3)]
    cb = [b.count(i) / n for i in range(3)]
    expected = sum(ca[i] * cb[j] * ((i - j) ** 2) / 4 for i in range(3) for j in range(3))
    return 1 - observed / expected if expected else 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_716)
    parser.add_argument(
        "--run-judge",
        choices=JUDGES,
        default=None,
        help="Run only one judge while retaining both judges in final agreement analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks)
    result_paths = sorted(args.results.glob("*_outputs.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"No generator outputs found in {args.results}")
    outputs = [
        record
        for path in result_paths
        for record in read_jsonl(path)
        if record["condition"] in CONDITIONS
    ]
    items = select_items(tasks, outputs)
    raw_path = args.output / "semantic_judgments.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.resume and raw_path.exists():
        for row in read_jsonl(raw_path):
            if row.get("parse_ok") and valid_judgment(row.get("judgment")):
                completed[(row["judge"], row["model"], row["condition"], row["task_id"])] = row

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
            base_url=args.ollama_url,
            timeout=args.timeout,
        )
        parsed = extract_json_object(response)
        if not valid_judgment(parsed):
            retry_prompt = f"""Your prior audit response used a missing or non-integer score. Convert it to exactly one JSON object with integer values 0, 1, or 2 for functionality_correctness, requirement_summary_correctness, risk_usefulness, unsupported_claim_severity, and overall_triage_utility, plus a rationale of at most 35 words. Never use decimals: choose 1 for useful with verification and 2 for substantially grounded. Preserve your prior assessment; do not add commentary.\n\nPRIOR RESPONSE:\n{response}"""
            response, retry_elapsed = call_ollama(
                judge,
                retry_prompt,
                base_url=args.ollama_url,
                timeout=args.timeout,
            )
            elapsed += retry_elapsed
            parsed = extract_json_object(response)
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
            isinstance(judgment.get(key), int) and 0 <= judgment[key] <= 2 for key in metrics
        ):
            continue
        clean.append(
            {
                **{
                    key: row[key]
                    for key in ("pair_id", "task_id", "language", "model", "condition", "judge")
                },
                **{key: judgment[key] for key in metrics},
                "rationale": str(judgment.get("rationale", "")).strip(),
            }
        )
    write_csv(args.output / "semantic_judgments.csv", clean)

    summary = []
    for condition in CONDITIONS:
        group = [row for row in clean if row["condition"] == condition]
        result = {
            "condition": condition,
            "outputs": len({(r["model"], r["task_id"]) for r in group}),
            "judgments": len(group),
        }
        for metric in metrics:
            mean, low, high = cluster_bootstrap(
                group,
                metric,
                cluster_key="pair_id",
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
            result[metric] = mean
            result[metric + "_ci_low"] = low
            result[metric + "_ci_high"] = high
        summary.append(result)
    write_csv(args.output / "semantic_summary.csv", summary)

    agreements = []
    index = defaultdict(dict)
    for row in clean:
        index[(row["model"], row["condition"], row["task_id"])][row["judge"]] = row
    for metric in metrics:
        paired = [judges for judges in index.values() if all(j in judges for j in JUDGES)]
        a = [judges[JUDGES[0]][metric] for judges in paired]
        b = [judges[JUDGES[1]][metric] for judges in paired]
        agreements.append(
            {
                "metric": metric,
                "pairs": len(paired),
                "exact_agreement": (sum(x == y for x, y in zip(a, b)) / len(a) if a else ""),
                "quadratic_weighted_kappa": weighted_kappa(a, b) if a else "",
            }
        )
    write_csv(args.output / "semantic_agreement.csv", agreements)

    semantic_differences = []
    clean_index = {(row["judge"], row["pair_id"], row["condition"]): row for row in clean}
    for metric in metrics:
        differences = []
        for judge in JUDGES:
            for pair_id in sorted({row["pair_id"] for row in clean}):
                patch = clean_index.get((judge, pair_id, "patch_only"))
                combined = clean_index.get((judge, pair_id, "issue_plus_patch"))
                if patch and combined:
                    differences.append(
                        {"pair_id": pair_id, "difference": combined[metric] - patch[metric]}
                    )
        mean, low, high = cluster_bootstrap(
            differences,
            "difference",
            cluster_key="pair_id",
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        semantic_differences.append(
            {
                "comparison": "issue_plus_patch - patch_only",
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    write_csv(args.output / "semantic_paired_differences.csv", semantic_differences)
    print(f"usable judgments: {len(clean)}/{len(rows)}")


if __name__ == "__main__":
    main()
