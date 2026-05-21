# Response Structure Integrity Evaluation

This directory contains the **Response Structure Integrity** evaluation pipeline used in **UniM**.

The current entry point is:

- `evaluate_rsi.py`

This module evaluates whether a model response follows the expected structural format of the reference output, focusing on modality type matching and modality-count consistency.

It reports two scores:

- `StS`: Strict Structure Score
- `LeS`: Lenient Structure Score

## Files

- `evaluate_rsi.py`: response structure integrity evaluation entry point
- `evaluate_rsi.sh`: shell wrapper for convenient command-line usage


## Installation

This module uses only Python standard-library packages.

You can run it directly with your Python environment:


## Usage

### Python

```bash
python evaluate_rsi.py \
  --golden-path /path/to/golden.jsonl \
  --model-output-path /path/to/model_output.jsonl \
  --eval-results-path /path/to/eval_results.jsonl
```

### Shell wrapper

```bash
bash evaluate_rsi.sh \
  /path/to/golden.jsonl \
  /path/to/model_output.jsonl \
  /path/to/eval_results.jsonl
```

## Input Format

The loader accepts either:

- a JSON array
- concatenated multi-line JSON objects

Each record is expected to contain at least:

- `id`
- `output.content`

Example golden item:

```json
{
  "id": "1",
  "output": {
    "content": "The answer is <image1> and <audio1>."
  }
}
```

Example model output item:

```json
{
  "id": "1",
  "output": {
    "content": "The answer is <image1>."
  }
}
```


## Output Format

The output file is written in standard JSONL format, one JSON object per line.

Each output record is copied from the model output record and extended with:

- `eval_score.StS`
- `eval_score.LeS`

Example:

```json
{
  "id": "1",
  "output": {
    "content": "The answer is <image1>."
  },
  "eval_score": {
    "StS": 0.5,
    "LeS": 0.5
  }
}
```

## Missing Predictions

If a golden record exists but the corresponding model output is missing:

- the record is still included in evaluation
- an empty response structure is used
- the resulting structure score will typically be `0.0`

This behavior ensures that missing outputs are not silently skipped.

## Important Notes

- Matching is done by `id`
- Records in the golden file without `id` are skipped with a warning
- Extra model output records without a matching golden `id` are not evaluated, and a warning is printed
- Only `output.content` is used for structure evaluation
- This module does not evaluate semantic quality or interleaved coherence
