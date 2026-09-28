# Unified Interleave Multimodal Agent (UNIMA)

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="xxxx"
export QWEN_OMNI_CACHE_DIR="xxxx"
```

The Qwen2.5-Omni setup follows the official repo/model:
`https://github.com/QwenLM/Qwen2.5-Omni`
`https://huggingface.co/Qwen/Qwen2.5-Omni-7B`

If your environment already has a conflicting `transformers` version, run:

```bash
pip uninstall -y transformers
pip install transformers==4.52.3
pip install qwen-omni-utils[decord]==0.0.10
pip install accelerate
```

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:7860`.

The local Qwen2.5-Omni model loads only when an audio or video input is submitted.
Set `QWEN_OMNI_WARMUP=1` to load it at startup. The cache defaults to `HF_HOME`
or `~/.cache/huggingface` when `QWEN_OMNI_CACHE_DIR` is unset. Generated files
are stored in a separate directory for each run under `runs/`.
