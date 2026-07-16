# patch-impact

Reproducibility artifact for **Patch-Grounded Software Change Reports with Local Large Language Models: What Remains Beyond Diff Parsing?**

The revised study separates changed-path extraction from semantic change-report evaluation. It compares 10 Ollama-served generators with a deterministic diff parser, reports repository-cluster bootstrap intervals and top-five ceilings, and performs a blind semantic audit with two independent local LLM judges.

## Contents

- `src/change_impact/`: reusable dataset preparation, prompting, model execution, and scoring package
- `scripts/revision_analysis.py`: parser baseline, language/ceiling analysis, paired contrasts, and clustered bootstrap intervals
- `scripts/semantic_llm_judge.py`: deterministic paired sampling, blind judging, agreement, and semantic contrasts
- `scripts/plot_revision_figures.py`: data-derived publication figures from the released result tables
- `prompts/`: structured change-report prompt
- `data/revision_task_manifest.csv`: exact task selection and repository clusters
- `revision_results/`: per-task measurements, semantic judgments, and aggregate tables used by the revised study

Generated patches and benchmark text are not redistributed. Manuscripts, submission files, local caches, and build products are excluded.

## Requirements

- Python 3.9+
- Ollama 0.6+ running locally at `http://127.0.0.1:11434`
- SWE-bench Lite and SWE-PolyBench benchmark exports
- approximately 48 GB unified/system memory for reproducing the largest generator locally

Install the package:

```bash
git clone https://github.com/ictu-se/patch-impact.git
cd patch-impact
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

`requirements.txt` lists every direct Python runtime dependency. The editable package install exposes the `change-impact` command; Ollama and its model weights are external system dependencies.

## Rebuild the Fixed Task Set

Prepare the benchmark records from the two external exports. The revision manifest records the exact 100 task identifiers selected with seed `20260515`.

```bash
change-impact prepare \
  --poly /path/to/SWE-PolyBench_500/test.csv \
  --swe /path/to/SWE-bench_Lite/test.csv \
  --out data/change_impact_tasks.jsonl \
  --summary data/change_impact_sample.csv
```

Confirm that the generated identifiers match `data/revision_task_manifest.csv` before running the matrix.

## Run the Generator Matrix

Install the exact model tags listed in `revision_results/model_manifest.csv`, then run:

```bash
change-impact matrix \
  --conditions issue_only patch_only issue_plus_patch \
  --workers 3 --timeout 300
```

The revised primary analysis uses `issue_only`, `patch_only`, and `issue_plus_patch`. The original `issue_plus_patch_plus_tree` condition is retained only for transparency because its tree was derived from changed paths.

## Reproduce the Leakage-Aware Analysis

After the matrix finishes:

```bash
python scripts/revision_analysis.py
```

This command extracts visible diff paths, computes the deterministic baseline, changed-path recall and precision, task-specific top-five ceilings, language strata, repository-cluster bootstrap intervals, and the paired issue-plus-patch versus patch-only contrast. Bootstrap seed `20260716` and 5,000 replicates are the CLI defaults. Paths and bootstrap settings can be overridden explicitly:

```bash
python scripts/revision_analysis.py \
  --tasks data/change_impact_tasks.jsonl \
  --results results \
  --output revision_results \
  --bootstrap-replicates 5000 \
  --bootstrap-seed 20260716
```

Regenerate all result figures without rerunning the models:

```bash
python scripts/plot_revision_figures.py \
  --results revision_results \
  --output generated_figures
```

## Reproduce the Semantic Audit

Install the two judge models and run:

```bash
ollama pull deepseek-coder:6.7b
ollama pull mistral:7b
python scripts/semantic_llm_judge.py --workers 3 --resume --run-judge deepseek-coder:6.7b
ollama stop deepseek-coder:6.7b
python scripts/semantic_llm_judge.py --workers 4 --resume --run-judge mistral:7b
```

The script selects 40 model-task pairs without using output quality, evaluates all three primary conditions, runs both judges at temperature zero, retries invalid structured responses, and writes 240 usable judgments when complete. It also reports exact agreement, quadratic-weighted kappa, output-cluster bootstrap intervals, and paired semantic contrasts. Use `--ollama-url` and `--timeout` when Ollama is not served at the default local endpoint.

## Reported Evidence

The tracked revision results include:

- task-level parser and LLM path measurements;
- aggregate condition intervals and paired contrasts;
- language-specific ceilings and scores;
- prompt/output token diagnostics and frozen model identifiers;
- clean semantic judgments, condition summaries, judge agreement, and paired semantic differences.

These files are sufficient to regenerate every reported numeric table without retaining publication source or submission material in the repository.

## Validation

Run the lightweight checks before analysis or after modifying the code:

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/plot_revision_figures.py \
  --results revision_results \
  --output generated_figures
```

## Reproducibility Notes

- Generator decoding: temperature `0.1`, top-p `0.9`, maximum `700` output tokens.
- Semantic judges: temperature `0`, fixed five-dimension 0--2 rubric, two independent model families.
- Path precision divides hits by the actual number of emitted predictions among the first five.
- A path outside the completed patch is reported as a non-patch prediction, not automatically as a hallucination.
- Runtime varies with hardware, cache state, quantization, and concurrent workers and is not used as a controlled ranking outcome.
