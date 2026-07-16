from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import CONDITIONS, DEFAULT_PROMPT, DEFAULT_RESULTS, DEFAULT_TASKS, safe_model_name
from .io import iter_jsonl
from .prompting import load_template, render_prompt


def call_ollama(model: str, prompt: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 700},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    return {
        "returncode": 0,
        "stdout": data.get("response", ""),
        "stderr": "",
        "elapsed_sec": round(time.time() - started, 3),
        "eval_count": data.get("eval_count", ""),
        "prompt_eval_count": data.get("prompt_eval_count", ""),
    }


def run_generation(
    *,
    model: str,
    condition: str,
    tasks_path: Path,
    template_path: Path,
    out_path: Path,
    timeout: int,
    limit: int = 0,
) -> None:
    template = load_template(template_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for task in iter_jsonl(tasks_path):
            prompt = render_prompt(template, task, condition)
            try:
                result = call_ollama(model, prompt, timeout)
            except TimeoutError:
                result = {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "timeout",
                    "elapsed_sec": timeout,
                }
            except urllib.error.URLError as exc:
                result = {
                    "returncode": -2,
                    "stdout": "",
                    "stderr": str(exc),
                    "elapsed_sec": timeout,
                }
            except http.client.RemoteDisconnected as exc:
                result = {
                    "returncode": -3,
                    "stdout": "",
                    "stderr": str(exc),
                    "elapsed_sec": timeout,
                }

            record = {
                "task_id": task["task_id"],
                "source": task["source"],
                "repo": task["repo"],
                "language": task["language"],
                "condition": condition,
                "model": model,
                "eval_count": "",
                "prompt_eval_count": "",
                **result,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            count += 1
            print(
                f"{count}: {task['task_id']} rc={record['returncode']} elapsed={record['elapsed_sec']}",
                flush=True,
            )
            if limit and count >= limit:
                break
    print(f"wrote {count} outputs to {out_path}")


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--tasks", default=DEFAULT_TASKS, type=Path)
    parser.add_argument("--template", default=DEFAULT_PROMPT, type=Path)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None, type=Path)


def run_single(args: argparse.Namespace) -> None:
    safe = safe_model_name(args.model)
    out = args.out or DEFAULT_RESULTS / f"{safe}_{args.condition}_outputs.jsonl"
    run_generation(
        model=args.model,
        condition=args.condition,
        tasks_path=args.tasks,
        template_path=args.template,
        out_path=out,
        timeout=args.timeout,
        limit=args.limit,
    )
