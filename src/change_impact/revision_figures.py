#!/usr/bin/env python3
"""Generate manuscript figures from the released revision result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "Diff parser": "#1b4965",
    "Issue only": "#9c755f",
    "Patch only": "#2a9d8f",
    "Issue + patch": "#e76f51",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}")
    plt.close(fig)


def path_recovery(results: Path, output: Path) -> None:
    data = pd.read_csv(results / "condition_cluster_ci.csv")
    labels = {
        "diff_parser": "Diff parser",
        "issue_only": "Issue only",
        "patch_only": "Patch only",
        "issue_plus_patch": "Issue + patch",
    }
    data["label"] = data["condition"].map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharey=True)
    y = np.arange(len(data))[::-1]
    for ax, prefix, title in zip(
        axes, ("file", "test"), ("Changed production paths", "Changed test paths")
    ):
        for index, row in data.iterrows():
            mean = row[f"{prefix}_r5"]
            low = row[f"{prefix}_r5_ci_low"]
            high = row[f"{prefix}_r5_ci_high"]
            ax.errorbar(
                mean,
                y[index],
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                ms=5,
                capsize=2.5,
                color=COLORS[row["label"]],
                ecolor=COLORS[row["label"]],
            )
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Recall@5 with 95% CI")
        ax.set_title(title)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, data["label"])
    fig.subplots_adjust(wspace=0.12)
    save(fig, output, "path_recovery_ci")


def language_recovery(results: Path, output: Path) -> None:
    parser = pd.read_csv(results / "language_and_ceiling.csv")
    llm = pd.read_csv(results / "llm_metrics_by_language.csv")
    patch = llm[llm["condition"] == "patch_only"].set_index("language")
    languages = parser["language"].tolist()
    x = np.arange(len(languages))
    width = 0.30
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    ax = axes[0]
    ax.bar(
        x - width / 2,
        parser["parser_file_r5"],
        width,
        label="Diff parser",
        color="#1b4965",
    )
    raw = np.array([patch.loc[v, "file_r5"] for v in languages])
    normalized = np.array([patch.loc[v, "file_ceiling_normalized"] for v in languages])
    ax.bar(x + width / 2, raw, width, label="Patch-only LLM", color="#2a9d8f")
    for i, ceiling in enumerate(parser["file_ceiling_r5"]):
        ax.plot(
            [i - 0.38, i + 0.38],
            [ceiling, ceiling],
            color="#555555",
            linestyle="--",
            linewidth=1,
        )
    ax.set_xticks(x, languages)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Production-path R@5")
    ax.set_title("(a) Raw recall and task ceiling")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    for i, language in enumerate(languages):
        ax.plot([raw[i], normalized[i]], [i, i], color="#999999", linewidth=1.5)
        ax.scatter(
            raw[i], i, color="#2a9d8f", marker="o", label="Raw R@5" if i == 0 else None
        )
        ax.scatter(
            normalized[i],
            i,
            color="#e9c46a",
            marker="s",
            label="Ceiling-normalized" if i == 0 else None,
        )
    ax.set_yticks(np.arange(len(languages)), languages)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Patch-only production recall")
    ax.set_title("(b) Effect of ceiling normalization")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(wspace=0.35)
    save(fig, output, "language_recovery")


def model_deltas(results: Path, output: Path) -> None:
    deltas = pd.read_csv(results / "model_paired_differences.csv").set_index("model")
    deltas = deltas.rename(columns={"file_delta": "production", "test_delta": "test"})
    deltas = deltas.sort_values("test")
    labels = [
        v.replace(":latest", "")
        .replace("qwen2.5-coder", "Qwen2.5-Coder")
        .replace("qwen2.5vl", "Qwen2.5-VL")
        .replace("qwen2.5", "Qwen2.5")
        .replace("llama3.2", "Llama3.2")
        .replace("gemma3", "Gemma3")
        .replace("granite3.2-vision", "Granite3.2-Vision")
        for v in deltas.index
    ]
    y = np.arange(len(deltas))
    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.scatter(
        deltas["production"],
        y + 0.13,
        color="#1b4965",
        marker="o",
        label="Production paths",
    )
    ax.scatter(
        deltas["test"], y - 0.13, color="#e76f51", marker="s", label="Test paths"
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean paired change in R@5: issue + patch minus patch only")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    save(fig, output, "model_paired_deltas")


def semantic_results(results: Path, output: Path) -> None:
    summary = pd.read_csv(results / "semantic_summary.csv")
    effects = pd.read_csv(results / "semantic_paired_differences.csv")
    dimensions = [
        ("functionality_correctness", "Functionality"),
        ("requirement_summary_correctness", "Requirement"),
        ("risk_usefulness", "Risk"),
        ("unsupported_claim_severity", "Unsupported"),
        ("overall_triage_utility", "Utility"),
    ]
    labels = {
        "issue_only": "Issue only",
        "patch_only": "Patch only",
        "issue_plus_patch": "Issue + patch",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    x = np.arange(len(dimensions))
    offsets = (-0.18, 0, 0.18)
    for offset, (_, row) in zip(offsets, summary.iterrows()):
        label = labels[row["condition"]]
        means = [row[key] for key, _ in dimensions]
        lows = [row[f"{key}_ci_low"] for key, _ in dimensions]
        highs = [row[f"{key}_ci_high"] for key, _ in dimensions]
        axes[0].errorbar(
            x + offset,
            means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            fmt="o",
            capsize=2,
            color=COLORS[label],
            label=label,
        )
    axes[0].set_xticks(x, [label for _, label in dimensions], rotation=25, ha="right")
    axes[0].set_ylim(-0.05, 2.08)
    axes[0].set_ylabel("Mean rubric score (0-2)")
    axes[0].set_title("(a) Condition profiles")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)

    effect_names = {
        "functionality_correctness": "Functionality",
        "requirement_summary_correctness": "Requirement",
        "risk_usefulness": "Risk",
        "unsupported_claim_severity": "Unsupported",
        "overall_triage_utility": "Utility",
    }
    effects["label"] = effects["metric"].map(effect_names)
    y = np.arange(len(effects))[::-1]
    axes[1].axvline(0, color="#555555", linewidth=0.9)
    for index, row in effects.iterrows():
        axes[1].errorbar(
            row["mean"],
            y[index],
            xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt="o",
            capsize=2.5,
            color="#e76f51",
        )
    axes[1].set_yticks(y, effects["label"])
    axes[1].set_xlabel("Paired mean difference with 95% CI")
    axes[1].set_title("(b) Issue + patch minus patch only")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.6)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.38)
    save(fig, output, "semantic_results")


def run_revision_figures(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    path_recovery(args.results, args.output)
    language_recovery(args.results, args.output)
    model_deltas(args.results, args.output)
    semantic_results(args.results, args.output)


def add_revision_figure_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=Path("revision_results"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
