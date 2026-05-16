from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import CONDITIONS, DEFAULT_FIGURES, DEFAULT_RESULTS
from .io import read_csv, write_csv


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values) -> float | None:
    vals = [value for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def model_condition_from_metrics_path(path: Path) -> tuple[str, str]:
    name = path.name.replace("_outputs_metrics.csv", "")
    for condition in sorted(CONDITIONS, key=len, reverse=True):
        suffix = "_" + condition
        if name.endswith(suffix):
            return name[: -len(suffix)].replace("_", ":", 1), condition
    return "", ""


def quality_score(row: dict[str, Any]) -> float:
    affected = as_float(row["affected_file_recall_5"]) or 0.0
    tests = as_float(row["test_file_recall_5"]) or 0.0
    parse = as_float(row["parse_ok"]) or 0.0
    file_precision = as_float(row["affected_file_precision_5"]) or 0.0
    test_precision = as_float(row["test_file_precision_5"]) or 0.0
    precision = (file_precision + test_precision) / 2.0
    risk_present = min((as_float(row["risk_note_count"]) or 0.0), 3.0) / 3.0
    summary_words = as_float(row["summary_words"]) or 0.0
    summary_present = 1.0 if 15 <= summary_words <= 100 else 0.0
    return 100.0 * (
        0.28 * affected
        + 0.28 * tests
        + 0.16 * precision
        + 0.14 * parse
        + 0.07 * risk_present
        + 0.07 * summary_present
    )


def load_metric_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_outputs_metrics.csv")):
        model, condition = model_condition_from_metrics_path(path)
        for row in read_csv(path):
            row["model"] = row.get("model") or model
            row["condition"] = row.get("condition") or condition
            row["quality_score"] = quality_score(row)
            row["overclaim_file_proxy"] = 1.0 - (as_float(row["affected_file_precision_5"]) or 0.0)
            row["overclaim_test_proxy"] = 1.0 - (as_float(row["test_file_precision_5"]) or 0.0)
            rows.append(row)
    return rows


def write_summaries(results_dir: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = defaultdict(list)
    source_groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"])].append(row)
        source_groups[(row["model"], row["condition"], row["source"])].append(row)

    metric_names = [
        "parse_ok",
        "affected_file_recall_3",
        "affected_file_recall_5",
        "affected_file_precision_5",
        "test_file_recall_3",
        "test_file_recall_5",
        "test_file_precision_5",
        "risk_note_count",
        "summary_words",
        "elapsed_sec",
        "quality_score",
        "overclaim_file_proxy",
        "overclaim_test_proxy",
    ]
    fields = ["model", "condition", "rows"] + metric_names

    summary = []
    for (model, condition), group in sorted(groups.items()):
        out = {"model": model, "condition": condition, "rows": len(group)}
        for name in metric_names:
            out[name] = mean(as_float(row.get(name)) for row in group)
        summary.append(out)
    write_csv(results_dir / "change_impact_summary_by_condition.csv", summary, fields)

    by_model = defaultdict(list)
    for row in summary:
        by_model[row["model"]].append(row)
    best_rows = [
        max(candidates, key=lambda row: row["quality_score"] if row["quality_score"] is not None else -math.inf)
        for _, candidates in sorted(by_model.items())
    ]
    write_csv(results_dir / "change_impact_best_per_model.csv", best_rows, fields)

    source_summary = []
    for (model, condition, source), group in sorted(source_groups.items()):
        out = {"model": model, "condition": condition, "source": source, "rows": len(group)}
        for name in metric_names:
            out[name] = mean(as_float(row.get(name)) for row in group)
        source_summary.append(out)
    write_csv(results_dir / "change_impact_summary_by_source.csv", source_summary, ["model", "condition", "source", "rows"] + metric_names)

    task_index = {(row["model"], row["condition"], row["task_id"]): row for row in rows}
    deltas = []
    for model in sorted(by_model):
        issue_rows = [row for row in rows if row["model"] == model and row["condition"] == "issue_only"]
        for base in issue_rows:
            paired = task_index.get((model, "issue_plus_patch", base["task_id"]))
            if not paired:
                continue
            deltas.append({
                "model": model,
                "task_id": base["task_id"],
                "source": base["source"],
                "affected_file_recall_5_delta": (as_float(paired["affected_file_recall_5"]) or 0.0) - (as_float(base["affected_file_recall_5"]) or 0.0),
                "test_file_recall_5_delta": (as_float(paired["test_file_recall_5"]) or 0.0) - (as_float(base["test_file_recall_5"]) or 0.0),
                "quality_score_delta": (as_float(paired["quality_score"]) or 0.0) - (as_float(base["quality_score"]) or 0.0),
            })
    write_csv(results_dir / "change_impact_paired_deltas.csv", deltas, ["model", "task_id", "source", "affected_file_recall_5_delta", "test_file_recall_5_delta", "quality_score_delta"])
    return summary, best_rows, deltas


def write_figures(figures_dir: Path, summary: list[dict[str, Any]], best_rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    best_sorted = sorted(best_rows, key=lambda row: row["quality_score"], reverse=True)
    labels = [row["model"].replace("qwen2.5-coder", "qwen-coder").replace(":", "\n") for row in best_sorted]
    scores = [row["quality_score"] for row in best_sorted]
    runtimes = [row["elapsed_sec"] for row in best_sorted]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(range(len(labels)), scores, color="#3b6ea8")
    axes[0].set_ylabel("Screening score")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylim(0, max(scores) * 1.18 if scores else 1)
    axes[1].scatter(runtimes, scores, s=70, color="#b34d4d")
    for row in best_sorted:
        axes[1].annotate(row["model"].split(":")[0].replace("qwen2.5-coder", "qwen-coder"), (row["elapsed_sec"], row["quality_score"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("Mean runtime (s)")
    axes[1].set_ylabel("Screening score")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_model_screening_runtime.pdf")
    plt.close(fig)

    patch = sorted([row for row in summary if row["condition"] == "issue_plus_patch"], key=lambda row: row["model"])
    labels = [row["model"].replace("qwen2.5-coder", "qwen-coder").replace(":", "\n") for row in patch]
    file_recalls = [row["affected_file_recall_5"] for row in patch]
    test_recalls = [row["test_file_recall_5"] for row in patch]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.bar([i - 0.18 for i in x], file_recalls, width=0.36, label="Affected files", color="#2c7a7b")
    ax.bar([i + 0.18 for i in x], test_recalls, width=0.36, label="Test files", color="#d08c2e")
    ax.set_ylabel("Recall@5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(file_recalls + test_recalls) * 1.22 if file_recalls or test_recalls else 1)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_patch_condition_recall.pdf")
    plt.close(fig)

    delta_groups = defaultdict(list)
    for row in deltas:
        delta_groups[row["model"]].append(row)
    labels = []
    file_d = []
    test_d = []
    for model in sorted(delta_groups):
        group = delta_groups[model]
        labels.append(model.replace("qwen2.5-coder", "qwen-coder").replace(":", "\n"))
        file_d.append(mean(as_float(row["affected_file_recall_5_delta"]) for row in group))
        test_d.append(mean(as_float(row["test_file_recall_5_delta"]) for row in group))
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.bar([i - 0.18 for i in x], file_d, width=0.36, label="Affected files", color="#577590")
    ax.bar([i + 0.18 for i in x], test_d, width=0.36, label="Test files", color="#f3722c")
    ax.set_ylabel("Issue+patch minus issue-only")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_patch_gain_by_model.pdf")
    plt.close(fig)


def add_summarize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS, type=Path)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES, type=Path)
    parser.add_argument("--skip-figures", action="store_true")


def run_summarize(args: argparse.Namespace) -> None:
    rows = load_metric_rows(args.results_dir)
    summary, best_rows, deltas = write_summaries(args.results_dir, rows)
    if not args.skip_figures:
        write_figures(args.figures_dir, summary, best_rows, deltas)
    print(f"wrote {len(summary)} condition rows and {len(best_rows)} best rows")
