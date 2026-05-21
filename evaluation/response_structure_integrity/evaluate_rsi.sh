#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash evaluate_rsi.sh <golden_path> <model_output_path> <eval_results_path>
#
# Example:
#   bash evaluate_rsi.sh data/golden.jsonl data/pred.jsonl results/eval_results.jsonl

if [[ $# -ne 3 ]]; then
  echo "Usage: bash evaluate_rsi.sh <golden_path> <model_output_path> <eval_results_path>"
  exit 1
fi

GOLDEN_PATH="$1"
MODEL_OUTPUT_PATH="$2"
EVAL_RESULTS_PATH="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/evaluate_rsi.py" \
  --golden-path "${GOLDEN_PATH}" \
  --model-output-path "${MODEL_OUTPUT_PATH}" \
  --eval-results-path "${EVAL_RESULTS_PATH}"
