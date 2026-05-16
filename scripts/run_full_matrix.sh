#!/usr/bin/env bash
set -euo pipefail

change-impact matrix --workers "${WORKERS:-3}" --timeout "${TIMEOUT:-240}"
change-impact summarize
