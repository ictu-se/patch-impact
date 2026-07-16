"""Shared utilities for deterministic replication analyses."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read a JSON Lines file and report malformed records with line numbers."""
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            records.append(value)
    return records


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write mappings to CSV using the first row as the stable schema."""
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_json_object(text: str | None) -> JsonObject | None:
    """Extract one JSON object from plain text or a fenced model response."""
    candidate = (text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)
    attempts = [candidate]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start >= 0 and end > start:
        attempts.append(candidate[start : end + 1])
    for attempt in attempts:
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("Percentiles require at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    cluster_key: str,
    replicates: int = 5_000,
    seed: int = 20_260_716,
) -> tuple[float, float, float]:
    """Estimate a mean and percentile interval by resampling whole clusters."""
    if not rows:
        raise ValueError(f"No rows available for bootstrap metric {value_key!r}")
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")

    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[cluster_key])].append(float(row[value_key]))
    cluster_ids = sorted(groups)
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled_ids = [generator.choice(cluster_ids) for _ in cluster_ids]
        sample = [value for cluster_id in sampled_ids for value in groups[cluster_id]]
        draws.append(sum(sample) / len(sample))

    observed = [float(row[value_key]) for row in rows]
    mean = sum(observed) / len(observed)
    return mean, percentile(draws, 0.025), percentile(draws, 0.975)
