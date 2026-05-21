#!/usr/bin/env bash

set -euo pipefail

# Semantic Correctness environment setup script
#
# Usage:
#   bash setup_sc_eval_env.sh
#   bash setup_sc_eval_env.sh <env_name>
#
# Example:
#   bash setup_sc_eval_env.sh unim_sc

ENV_NAME="${1:-unim_sc}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/sc_requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] sc_requirements.txt not found: ${REQ_FILE}"
  exit 1
fi

echo "[INFO] Please make sure these system dependencies are installed:"
echo "       ffmpeg, poppler, cairo, jpeg-turbo, tesseract"

echo "[INFO] Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip

echo "[INFO] Installing Python dependencies from sc_requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

cat <<EOF

[DONE] Environment setup finished.

To activate the environment:
  conda activate ${ENV_NAME}

Required runtime variables:
  export OPENAI_API_KEY="your_openai_api_key"

Optional runtime variables:
  export VLLM_API_KEY="EMPTY"

Extra runtime notes:
  1. Audio/video captioning expects an OpenAI-compatible vLLM endpoint.
  2. The default vLLM model is Qwen/Qwen2.5-Omni-3B.
  3. Please make sure ffmpeg, poppler, cairo, jpeg-turbo, and tesseract are installed.
  4. CUDA is not installed by this script. If you need GPU PyTorch, install the
     matching CUDA-enabled PyTorch build separately for your machine.
  5. Real 3D captioning in evaluate_sc.py additionally requires PointLLM and a valid
     PointLLM model path. Please install PointLLM separately from its official repository.

EOF
