# patch-impact

Reproducibility artifact for **Patch-Grounded Software Change Impact Analysis and Regression Test Triage with Large Language Models**.

This repository contains only the code, prompt template, and task manifest needed to rerun the experiment. It intentionally excludes generated model outputs, metrics, figures, and manuscript files.

## What This Reproduces

The full experiment evaluates 10 Ollama-served models on 100 repository tasks under four evidence conditions:

- `issue_only`
- `patch_only`
- `issue_plus_patch`
- `issue_plus_patch_plus_tree`

The pipeline rebuilds the task JSONL from external benchmark CSVs, runs the model matrix, scores file/test alignment, computes automatic rubric proxies, and writes aggregate CSVs and figures.

## Requirements

- Python 3.9+
- Ollama running locally at `http://127.0.0.1:11434`
- Benchmark CSV exports:
  - SWE-PolyBench 500 test CSV
  - SWE-bench Lite test CSV

The benchmark CSVs are external research artifacts and are not redistributed here.

## Install

```bash
git clone https://github.com/ictu-se/patch-impact.git
cd patch-impact
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

Install the Ollama models used in the paper:

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull granite3.2-vision:latest
ollama pull llama3.2-vision:11b
ollama pull qwen2.5vl:3b
```

## Prepare the 100 Tasks

Use the included `data/task_manifest.csv` to select the exact 100 task IDs used in the paper:

```bash
change-impact prepare \
  --poly /path/to/SWE-PolyBench_500/test.csv \
  --swe /path/to/SWE-bench_Lite/test.csv \
  --out data/change_impact_tasks.jsonl \
  --summary data/change_impact_sample.csv
```

The generated `data/change_impact_tasks.jsonl` contains patch excerpts and gold file/test oracles derived from the external CSVs. It is ignored by Git.

## Run a Smoke Test

```bash
change-impact matrix \
  --models qwen2.5-coder:1.5b \
  --conditions issue_only issue_plus_patch \
  --limit 2 \
  --workers 1
```

Then summarize:

```bash
change-impact summarize
```

## Run the Full Matrix

Start Ollama, then run:

```bash
change-impact matrix --workers 3 --timeout 240
change-impact summarize
```

Outputs are written to:

- `results/` for raw model outputs, per-run metrics, rubric CSVs, and aggregate CSVs
- `figures/` for PDF figures

Both directories are ignored by Git.

## Main Commands

```bash
change-impact prepare    # create task JSONL from benchmark CSVs
change-impact run        # run one model/condition
change-impact matrix     # run and score many model/condition pairs
change-impact score      # score file/test alignment for one output file
change-impact rubric     # score automatic requirement/risk proxies
change-impact summarize  # aggregate metrics and generate figures
```

## Reproducibility Notes

The task sample is fixed by `data/task_manifest.csv`. The generation settings are fixed in code: temperature `0.1`, top-p `0.9`, and `700` output tokens. Exact wall-clock runtime may vary with hardware, Ollama version, quantization, and concurrent workers.
