# Interleaved Coherence Evaluation

This directory contains the **Interleaved Coherence Score (ICS)** pipeline used in **UniM** to evaluate whether multimodal outputs remain coherent after modality placeholders are expanded into dense text descriptions.

The pipeline works in two stages:

1. It converts each referenced modality file into text.
2. It replaces placeholders such as `<image1>` or `<audio2>` in the generated response, then asks a judge model to score the final interleaved text on:
   - `coherence`
   - `style_consistency`

Both scores are integers in the range **1-5**.

## Files

- `evaluate_ics.py`: main evaluation entry point for ICS scoring
- `x2text_gpt5.py`: local modality-to-text helpers
- `ics_requirements.txt`: Python dependencies for the ICS pipeline
- `setup_ics_eval_env.sh`: conda environment creation and dependency installation script


## Installation

We provide two environment installation methods.

(1) We recommend using the provided setup script:

```bash
bash setup_ics_eval_env.sh
```

This script will:

- create a conda environment `unim_ics`
- install all dependencies listed in `ics_requirements.txt`

You can also specify a custom environment name:

```bash
bash setup_ics_eval_env.sh unim_ics
```

After installation, activate the environment:

```bash
conda activate unim_ics
```

(2) If you prefer not to use the script, you can install the same environment manually:

```bash
conda create -n unim_ics python=3.10 pip
conda activate unim_ics
pip install -r ics_requirements.txt
```


## API Keys and Runtime Setup

### 1. OpenAI API

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. Local vLLM server for audio and video

For `audio` and `video` modalities, this pipeline expects a local **OpenAI-compatible** endpoint, typically served with `vLLM`.

Set the endpoint variables:

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
export VLLM_API_KEY="EMPTY"
```

Example server launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Omni-3B \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enable-log-requests \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --h11-max-incomplete-event-size 200000000 \
  --task generate \
  --limit-mm-per-prompt '{"video": 1}'
```


## Input Example Format

The input file is a JSONL file. Each record should contain:

- `domain`
- `subdomain`
- `id`
- `difficulty_level`
- `output.content`: text containing modality placeholders
- `output.modal`: mapping from placeholder names to relative local file paths

Example:

```json
{
  "domain": "natural_science",
  "subdomain": "computer_science",
  "id": "501",
  "difficulty_level": 1,
  "output": {
    "content": "The figure in <image1> supports the main claim.",
    "modal": {
      "image1": "image/example.png"
    }
  }
}
```

## Usage

Run the evaluation script with:

```bash
python evaluate_ics.py \
  -i /path/to/input.jsonl \
  -o /path/to/output.jsonl \
  -d /path/to/data
```

Arguments:

- `-i, --input`: input JSONL file
- `-o, --output`: output JSONL file
- `-d, --base_dir`: root directory containing the local modality files

The output file is cleared at the start of each run.


## Output Format

Each output record contains:

- `content`: placeholder-expanded text
- `modal`: dense caption generated for each placeholder
- `score.coherence`
- `score.style_consistency`

Example:

```json
{
  "domain": "natural_science",
  "subdomain": "computer_science",
  "id": "501",
  "content": "The figure in <image1:A technical illustration showing the core system components and their interactions.> supports the main claim.",
  "modal": {
    "image1": "A technical illustration showing the core system components and their interactions."
  },
  "difficulty_level": 1,
  "score": {
    "coherence": 5,
    "style_consistency": 4
  }
}
```
