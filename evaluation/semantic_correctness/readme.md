# Semantic Correctness Evaluation

This directory contains the **Semantic Correctness (SC)** evaluation pipeline used in **UniM**.

The current entry point is:

- `evaluate_sc.py`

The script performs three core steps in a single pipeline:

1. Read model responses and ground-truth annotations.
2. Replace multimodal placeholders such as `<image1>` and `<audio2>` with generated captions or transcriptions.
3. Score semantic correctness between the caption-expanded response and the reference answer.

## Files

- `evaluate_sc.py`: semantic correctness evaluation entry point
- `sc_requirements.txt`: Python dependencies for this module
- `setup_sc_eval_env.sh`: conda environment creation and dependency installation script
- `readme.md`: usage documentation

## Installation

### Option 1: Recommended setup script

```bash
bash setup_sc_eval_env.sh unim_sc
```

Then activate the environment:

```bash
conda activate unim_sc
```

### Option 2: Manual installation

```bash
conda create -n unim_sc python=3.11 pip
conda activate unim_sc
pip install -r sc_requirements.txt
```

## Runtime Requirements

### 1. OpenAI API

This script uses `gpt-5-mini` for:

- image captioning
- document captioning
- semantic correctness scoring

Set your API key before running:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. vLLM endpoint for audio/video captioning

If your responses contain `audio` or `video` outputs, you also need an OpenAI-compatible vLLM endpoint.

The default script configuration expects:

```text
http://127.0.0.1:8009/v1
```

Example:

```bash
vllm serve Qwen/Qwen2.5-Omni-3B \
  --dtype auto \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port 8009
```

If your endpoint requires an API key, you can optionally set:

```bash
export VLLM_API_KEY="EMPTY"
```

### 3. System dependencies

Please make sure these system packages are already installed:

- `ffmpeg`
- `poppler`
- `cairo`
- `jpeg-turbo`
- `tesseract`

For the current `evaluate_sc.py` workflow, the only system tool directly used in code is:

- `ffmpeg`: used for `.webm` conversion and video cleanup before captioning

The setup script prints these dependency reminders but does not install them automatically.

## Usage

Run the SC evaluator with:

```bash
python evaluate_sc.py \
  --response-path input.jsonl \
  --ground-truth-path input_gt.jsonl \
  --assets-root . \
  --vllm-endpoint http://127.0.0.1:8009/v1 \
  --vllm-model-name Qwen/Qwen2.5-Omni-3B
```

Optional arguments:

- `--traverse-mode response`: iterate through response items first
- `--traverse-mode gt`: iterate through ground-truth items first; this is the default
- `--pointllm-model-path`: currently kept only for interface compatibility

## Expected Directory Structure

A simple and recommended layout is:

```text
project_root/
├── response.jsonl
├── ground_truth.jsonl
├── image/
│   ├── img_0001_01.jpg
│   └── ...
├── audio/
│   ├── aud_0001_01.mp3
│   └── ...
├── video/
│   ├── vid_0001_01.mp4
│   └── ...
├── document/
│   ├── doc_0001_01.pdf
│   └── ...
└── threeD/
    ├── obj_0001_01.ply
    └── ...
```

In this setup, the relative paths stored in `output.modal` are resolved against `--assets-root`.

For example, if a response item contains:

```json
{
  "output": {
    "modal": {
      "image2": "image/img_0001_01.jpg",
      "audio2": "audio/aud_0001_01.mp3"
    }
  }
}
```

then you can run:

```bash
python evaluate_sc.py \
  --response-path project_root/response.jsonl \
  --ground-truth-path project_root/ground_truth.jsonl \
  --assets-root project_root \
  --vllm-endpoint http://127.0.0.1:8009/v1 \
  --vllm-model-name Qwen/Qwen2.5-Omni-3B
```

If you set `--assets-root .`, then the resource folders should be organized under the current working root accordingly.

## Traversal Modes

### `gt` mode (default)

- Iterate over ground-truth items
- Match responses by `id`
- Skip unmatched ground-truth entries
- Save results to:

```text
<response_path>.caption_scored.traverse_gt.jsonl
```

### `response` mode

- Iterate over response items
- Still generate caption-expanded outputs even if no matching ground truth is found
- Use `score: null` when no reference is available
- Save results to:

```text
<response_path>.caption_scored.jsonl
```

## Input Format

The reader accepts either:

- a JSON array
- concatenated multi-line JSON objects

Each response item should contain at least:

- `id`
- `output.modal`
- `output.content`

Each ground-truth item should contain at least:

- `id`
- `output.content`

Example response item:

```json
{
  "domain": "natural_science",
  "subdomain": "math",
  "id": "1",
  "output": {
    "modal": {
      "audio2": "audio/aud_0001_02.mp3"
    },
    "content": "The missing shape number is <audio2>."
  }
}
```

Example ground-truth item:

```json
{
  "domain": "natural_science",
  "subdomain": "math",
  "id": "1",
  "output": {
    "content": "The missing shape number is three."
  }
}
```



## Important Notes

- Relative paths inside `output.modal` are resolved against `--assets-root`
- A convenient setup is to place the resource folders under the same project root as `response.jsonl` and `ground_truth.jsonl`
- URLs are also supported for multimodal assets
- `.webm` audio/video files are converted before being sent to the captioning backend
- The script writes output as a pretty-printed JSON list, even though the file suffix remains `.jsonl`
- `threeD` captioning requires `PointLLM`, `open3d`, and a CUDA-enabled environment
- Remote 3D assets are downloaded to a temporary local file before captioning when needed

## Output Format

The script copies the original response item and adds:

- `output.captioned_response`
- `output.score`

Example:

```json
{
  "domain": "natural_science",
  "subdomain": "math",
  "id": "1",
  "output": {
    "modal": {
      "audio2": "audio/aud_0001_02.mp3"
    },
    "content": "The missing shape number is <audio2>.",
    "captioned_response": "The missing shape number is <audio2: The audio explains that the answer is number three.>",
    "score": 5
  }
}
```
