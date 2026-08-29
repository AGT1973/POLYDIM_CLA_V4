#!/usr/bin/env bash
set -euo pipefail
PROJECT="${1:-.}"
export POLYDIM_PROJECT="$(cd "$PROJECT" && pwd)"
cd "$(dirname "$0")/.."
python -m pytest -q scientific_suite/tests --tb=short
