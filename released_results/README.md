# Released experimental results

This directory contains the non-manuscript artifacts needed to verify the
reported tables and regenerate the empirical figures.

- `model_outputs/` contains the 3,000 reports from 10 generators, 100 tasks,
  and the three analyzed evidence conditions.
- `revision/` contains task-level metrics, repository-cluster bootstrap
  intervals, paired effects, blinded semantic judgments, agreement estimates,
  and judge-sensitivity results.

The generated task dataset is not redistributed because it is rebuilt from the
two upstream benchmark exports. Use `change-impact prepare` with the included
task manifest before rerunning the analyses.
