#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

change-impact matrix \
  --conditions issue_only patch_only issue_plus_patch \
  --workers "${WORKERS:-3}" \
  --timeout "${TIMEOUT:-300}"
python scripts/revision_analysis.py
