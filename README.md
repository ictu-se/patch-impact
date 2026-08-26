# patch-impact

Reproducibility artifact for **Patch-Grounded Software Change Reports with Local Large Language Models: What Remains Beyond Diff Parsing?**

This repository contains the refactored experiment code, prompt template, fixed task manifest, released model outputs, and analysis results needed to verify the reported tables and regenerate the figures. Manuscript sources and submission files are intentionally excluded.

## What This Reproduces

The reported experiment evaluates 10 Ollama-served models on 100 repository tasks under three evidence conditions:

- `issue_only`
- `patch_only`
- `issue_plus_patch`

The pipeline rebuilds the task JSONL from external benchmark CSVs, runs the model matrix, computes changed-path metrics and deterministic parser baselines, estimates repository-cluster bootstrap intervals and paired effects, performs a blinded two-judge semantic audit, and regenerates the four empirical figures.

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

The semantic audit additionally uses:

```bash
ollama pull deepseek-coder:6.7b
ollama pull mistral:7b
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
  --conditions patch_only issue_plus_patch \
  --limit 2 \
  --workers 1
```

Then summarize:

```bash
change-impact revision \
  --tasks data/change_impact_tasks.jsonl \
  --results results \
  --out revision_results
```

## Run the Full Matrix

Start Ollama, then run:

```bash
change-impact matrix --workers 3 --timeout 240
change-impact revision --results results --out revision_results
change-impact semantic-audit \
  --results results \
  --out revision_results \
  --workers 4
change-impact revision-figures \
  --results revision_results \
  --output figures
```

Outputs are written to:

- `results/` for raw model outputs and per-run measurements
- `revision_results/` for parser baselines, clustered intervals, paired effects, and semantic-audit tables
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
change-impact revision   # parser baseline, ceilings, clustered CIs, paired effects
change-impact semantic-audit    # blinded two-judge semantic evaluation
change-impact revision-figures  # regenerate the four reported empirical figures
```

## Verify the Released Results

The committed `released_results/` directory contains all 3,000 reports used by
the revised three-condition analysis and the derived tables. After preparing
`data/change_impact_tasks.jsonl`, recompute the deterministic and LLM path
analyses into a temporary directory:

```bash
change-impact revision \
  --tasks data/change_impact_tasks.jsonl \
  --results released_results/model_outputs \
  --out reproduced_revision
```

To recompute the published semantic tables from the committed raw judgments
without issuing new model calls, first copy the raw judgment file and resume:

```bash
mkdir -p reproduced_semantic
cp released_results/revision/semantic_judgments.jsonl reproduced_semantic/
change-impact semantic-audit \
  --tasks data/change_impact_tasks.jsonl \
  --results released_results/model_outputs \
  --out reproduced_semantic \
  --resume
```

Regenerate the figures directly from the released tables:

```bash
change-impact revision-figures \
  --results released_results/revision \
  --output reproduced_figures
```

## Reproducibility Notes

The task sample is fixed by `data/task_manifest.csv`. Generation uses temperature `0.1`, top-p `0.9`, and 700 output tokens. Semantic judging uses temperature `0`, top-p `0.9`, and 240 output tokens with explicit calibration anchors. The audit selects one task from each of four languages per generator, evaluates all three conditions, and uses two judges for 240 judgments. Exact wall-clock runtime may vary with hardware, Ollama version, quantization, and concurrent workers.
