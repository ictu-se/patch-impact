from __future__ import annotations

import argparse

from .dataset import add_prepare_args, run_prepare
from .matrix import add_matrix_args, run_matrix
from .ollama_runner import add_run_args, run_single
from .scoring import add_score_args, run_rubric, run_score
from .summarize import add_summarize_args, run_summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="change-impact")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build the 100-task JSONL from external benchmark CSVs.")
    add_prepare_args(prepare)
    prepare.set_defaults(func=run_prepare)

    run = sub.add_parser("run", help="Run one model and one evidence condition through Ollama.")
    add_run_args(run)
    run.set_defaults(func=run_single)

    matrix = sub.add_parser("matrix", help="Run and score a model/condition matrix.")
    add_matrix_args(matrix)
    matrix.set_defaults(func=run_matrix)

    score = sub.add_parser("score", help="Score file/test alignment metrics for one output JSONL.")
    add_score_args(score)
    score.set_defaults(func=run_score)

    rubric = sub.add_parser("rubric", help="Score automatic requirement/risk rubric proxies for one output JSONL.")
    add_score_args(rubric)
    rubric.set_defaults(func=run_rubric)

    summarize = sub.add_parser("summarize", help="Aggregate metrics and write summary CSVs/figures.")
    add_summarize_args(summarize)
    summarize.set_defaults(func=run_summarize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
