from __future__ import annotations

from pathlib import Path


CONDITIONS = [
    "issue_only",
    "patch_only",
    "issue_plus_patch",
    "issue_plus_patch_plus_tree",
]

MODELS = [
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "gemma3:4b",
    "granite3.2-vision:latest",
    "llama3.2-vision:11b",
    "qwen2.5vl:3b",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = PROJECT_ROOT / "data" / "change_impact_tasks.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "task_manifest.csv"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "change_impact_prompt.md"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_FIGURES = PROJECT_ROOT / "figures"


def safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
