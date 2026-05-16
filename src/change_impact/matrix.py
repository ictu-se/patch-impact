from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import CONDITIONS, DEFAULT_RESULTS, DEFAULT_TASKS, MODELS, count_jsonl_rows, safe_model_name


def run_logged(cmd: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=log, check=True)


def run_job(model: str, condition: str, args: argparse.Namespace, expected_rows: int) -> tuple[str, str]:
    results_dir = args.results_dir
    safe = safe_model_name(model)
    out = results_dir / f"{safe}_{condition}_outputs.jsonl"
    metrics = results_dir / f"{safe}_{condition}_outputs_metrics.csv"
    rubric = results_dir / f"{safe}_{condition}_outputs_rubric.csv"
    log_path = results_dir / f"{safe}_{condition}.log"

    if out.exists() and count_jsonl_rows(out) < expected_rows:
        out.unlink()

    if not out.exists() or count_jsonl_rows(out) < expected_rows:
        cmd = [
            sys.executable,
            "-m",
            "change_impact.cli",
            "run",
            "--model",
            model,
            "--condition",
            condition,
            "--tasks",
            str(args.tasks),
            "--timeout",
            str(args.timeout),
            "--out",
            str(out),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        run_logged(cmd, log_path)

    run_logged([sys.executable, "-m", "change_impact.cli", "score", str(out), "--tasks", str(args.tasks), "--out", str(metrics)], log_path)
    run_logged([sys.executable, "-m", "change_impact.cli", "rubric", str(out), "--tasks", str(args.tasks), "--out", str(rubric)], log_path)
    return model, condition


def add_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--tasks", default=DEFAULT_TASKS, type=Path)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS, type=Path)
    parser.add_argument("--workers", type=int, default=2)


def run_matrix(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = args.limit or count_jsonl_rows(args.tasks)
    jobs = [(model, condition) for model in args.models for condition in args.conditions]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, model, condition, args, expected_rows) for model, condition in jobs]
        for future in as_completed(futures):
            model, condition = future.result()
            print(f"done {model} {condition}", flush=True)
