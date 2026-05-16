#!/usr/bin/env bash

set -euo pipefail

# Generate Quality environment setup script
#
# Usage:
#   bash setup_gq_eval_env.sh
#   bash setup_gq_eval_env.sh <env_name>
#
# Example:
#   bash setup_gq_eval_env.sh unim_gq

ENV_NAME="${1:-unim_gq}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/gq_requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] gq_requirements.txt not found: ${REQ_FILE}"
  exit 1
fi

echo "[INFO] Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip

echo "[INFO] Installing Python dependencies from gq_requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

cat <<EOF

[DONE] Environment setup finished.

To activate the environment:
  conda activate ${ENV_NAME}

Required for OpenAI-based text/code/document evaluation:
  export OPENAI_API_KEY="your_openai_api_key"

Extra runtime notes
  1. Document evaluation needs the system Tesseract OCR binary.
  2. Video evaluation additionally expects a DOVER checkout at:
       ../../DOVER

EOF
