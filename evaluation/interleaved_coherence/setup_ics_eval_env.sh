#!/usr/bin/env bash

set -euo pipefail

# Interleaved coherence environment setup script
#
# Usage:
#   bash setup_ics_eval_env.sh
#   bash setup_ics_eval_env.sh <env_name>
#
# Example:
#   bash setup_ics_eval_env.sh unim_ics

ENV_NAME="${1:-unim_ics}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/ics_requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] ics_requirements.txt not found: ${REQ_FILE}"
  exit 1
fi

echo "[INFO] Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip

echo "[INFO] Installing Python dependencies from ics_requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

cat <<EOF

[DONE] Environment setup finished.

To activate the environment:
  conda activate ${ENV_NAME}

Then set the required environment variables
  export OPENAI_API_KEY="your_openai_api_key"

Optional: for local audio/video captioning server
  export VLLM_API_BASE="http://localhost:8000/v1"
  export VLLM_API_KEY="EMPTY"

If you need audio/video evaluation, please start an OpenAI-compatible local server
such as vLLM before running the evaluator.
EOF
