# Generate Quality Evaluation

This directory contains the **Generate Quality (GQ)** evaluation pipeline used in **UniM** to assess the quality of generated outputs across multiple modalities.

The current entry point is:

- `evaluate_gq.py`

It reads a JSONL file, detects the modality type of each item in `output.modal`, computes a quality score for each generated output, and writes the scores back to a new JSONL file.


## Files

- `evaluate_gq.py`: unified generation-quality evaluation entry point
- `gq_requirements.txt`: Python dependencies for this module
- `setup_gq_eval_env.sh`: conda environment creation and dependency installation script
- `audio.py`: handcrafted statistical audio-quality scoring
- `image.py`: image-quality scoring based on simplified BRISQUE / NIQE-style logic
- `video.py`: standalone DOVER-based video-quality evaluation helper
- `threeD.py`: no-reference 3D quality scoring
- `document.py`: OCR + OpenAI-based document-quality evaluation helper
- `code.py`: standalone code-evaluation example script

## Installation

### Option 1: Recommended setup script

```bash
bash setup_gq_eval_env.sh unim_gq
```

You can also provide a custom conda environment name:


Then activate the environment:

```bash
conda activate unim_gq
```

### Option 2: Manual installation

```bash
conda create -n unim_gq python=3.10 pip
conda activate unim_gq
pip install -r gq_requirements.txt
```

## Extra Runtime Dependencies

### 1. OpenAI API

Set your API key before running:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. Tesseract OCR

Document evaluation uses `pytesseract`, so the system-level **Tesseract OCR binary** must also be installed.

Examples:

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### 3. DOVER for video evaluation

Video evaluation depends on a separate DOVER checkout.  
The current code expects DOVER at:

```text
../../DOVER
```

relative to this directory.

More specifically, `evaluate_gq.py` expects:

- `../../DOVER/dover.yml`
- pretrained DOVER weights referenced by that config
- DOVER Python modules importable from that checkout

If DOVER is missing, `evaluate_gq.py` will skip video scoring and return `null` for video outputs instead of crashing the entire run.

## Usage

Run the GQ evaluation script with:

```bash
python evaluate_gq.py --input input.jsonl --output output.jsonl
```

Arguments:

- `--input`: path to the input JSONL file
- `--output`: optional output JSONL path

If `--output` is omitted, the script writes to:

```text
<input_dir>/<input_name>_output.jsonl
```

## Input Format

The input file should be a JSONL file in multi-line expanded JSON format.

Each record is expected to contain at least:

- `domain`
- `subdomain`
- `id`
- `difficulty_level`
- `output.modal`
- `output.content`

Example:

```json
{
  "domain": "general_domain",
  "subdomain": "architecture",
  "id": "1",
  "input": {
    "modal": {
      "image1": "image/img_0001_01.jpg",
      "audio1": "audio/aud_0001_01.mp3"
    },
    "content": "Description of the input..."
  },
  "output": {
    "modal": {
      "image2": "image/img_0001_02.png",
      "audio2": "audio/aud_0001_02.mp3",
      "text1": "This is a text answer...",
      "code1": "def example(): pass"
    },
    "content": "Description of the output..."
  },
  "difficulty_level": 3
}
```

## Important Input Notes

- File paths inside `output.modal` are resolved relative to the input JSONL file directory.
- `text` and `code` are evaluated directly from their raw string content.
- The current unified script evaluates only `output.modal`, not `input.modal`.
- For file-based modalities (`image`, `audio`, `video`, `threeD`, `document`), missing files are logged and the corresponding score becomes `null`.

## Output Format

The output JSONL preserves the original record and adds a new `scores` field:

```json
{
  "domain": "general_domain",
  "subdomain": "architecture",
  "id": "1",
  "output": {
    "modal": {
      "image2": "image/img_0001_02.png",
      "audio2": "audio/aud_0001_02.mp3"
    },
    "content": "..."
  },
  "scores": {
    "image2": 85.5,
    "audio2": 78.3
  },
  "difficulty_level": 3
}
```

## Score Range

- All modality scores are reported on a `0-100` scale
- Higher is better
- `null` means the modality evaluation failed or was skipped

## How Each Modality Is Scored

### Image

- Implemented in `evaluate_gq.py` via `image.py`
- Uses simplified BRISQUE / NIQE-style feature scoring

### Audio

- Implemented in `evaluate_gq.py` via `audio.py`
- Uses handcrafted statistical features such as SNR, loudness, silence ratio, dynamic range, and spectral structure

### Video

- Implemented in `evaluate_gq.py` with DOVER
- Fuses DOVER technical and aesthetic quality outputs into a final `0-100` score

### 3D

- Implemented in `evaluate_gq.py` via `threeD.py`
- Uses no-reference topology / geometry / sampling quality scoring

### Document

- Implemented directly inside `evaluate_gq.py`
- Opens the file with PIL, extracts OCR text with `pytesseract`, converts it into a rough markdown table, then asks `gpt-5-mini` for an overall quality score

### Text

- Implemented directly inside `evaluate_gq.py`
- Uses `gpt-5-mini` to score clarity, coherence, informativeness, conciseness, and style, then maps the average from `1-5` to `0-100`

### Code

- Implemented directly inside `evaluate_gq.py`
- Uses `gpt-5-mini` to score correctness, readability, design, performance, security, and testability, and returns the model-reported overall score
