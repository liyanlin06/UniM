"""Generation Quality Evaluation Script.
This script evaluates GQ using local files.
Evaluates quality across 7 modalities: image, audio, video, text, code, document, 3D
"""

import argparse
import json
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global OpenAI client (initialized once)
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client singleton"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            # Get API key from environment variable
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required but not set.")
            _openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    return _openai_client


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified multi-modal quality evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: input_dir/input_name_output.jsonl)"
    )
    return parser.parse_args()


def read_multiline_jsonl(file_path):
    """
    Read multi-line expanded JSONL format
    Returns list of JSON objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by }{ pattern to separate objects
    objects = []
    depth = 0
    start = 0

    for i, char in enumerate(content):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                obj_str = content[start:i+1]
                try:
                    obj = json.loads(obj_str)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON object: {e}")

    return objects


def write_multiline_jsonl(objects, file_path):
    """
    Write multi-line expanded JSONL format
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for obj in objects:
            json_str = json.dumps(obj, ensure_ascii=False, indent=2)
            f.write(json_str + '\n')


def detect_modal_type(key):
    """
    Detect modal type from key prefix
    Returns: modal_type string or None
    """
    key_lower = key.lower()

    if key_lower.startswith('image'):
        return 'image'
    elif key_lower.startswith('audio'):
        return 'audio'
    elif key_lower.startswith('video'):
        return 'video'
    elif key_lower.startswith('threed'):
        return 'threeD'
    elif key_lower.startswith('document'):
        return 'document'
    elif key_lower.startswith('text'):
        return 'text'
    elif key_lower.startswith('code'):
        return 'code'
    else:
        return None


def evaluate_image(file_path):
    """Evaluate image quality using image.py logic"""
    try:
        import cv2
        import numpy as np
        from scipy.special import gamma as gamma_func

        # Import image evaluation functions
        sys.path.insert(0, os.path.dirname(__file__))
        from image import calculate_brisque, calculate_niqe

        brisque = calculate_brisque(file_path)
        niqe = calculate_niqe(file_path)

        if brisque is None or niqe is None:
            return None

        # Normalize to 0-100 scale (lower is better, so invert)
        # BRISQUE: 0-100 (lower better) -> normalize to 0-100 (higher better)
        score = 100 - brisque
        return round(max(0, min(100, score)), 2)

    except Exception as e:
        logger.error(f"Image evaluation failed: {e}")
        return None


def evaluate_audio(file_path):
    """Evaluate audio quality using audio.py logic"""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from audio import score_audio

        result = score_audio(file_path)
        return result.get("quality_0_100")

    except Exception as e:
        logger.error(f"Audio evaluation failed: {e}")
        return None


def evaluate_video(file_path):
    """Evaluate video quality using DOVER model"""
    try:
        import torch
        import yaml
        import numpy as np

        # Check if DOVER is available
        dover_path = os.path.join(os.path.dirname(__file__), '../../DOVER')
        if not os.path.exists(dover_path):
            logger.warning(f"DOVER directory not found at {dover_path}, skipping video evaluation")
            return None

        sys.path.append(dover_path)
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from dover.models import DOVER

        # Load config
        opt_path = os.path.join(dover_path, 'dover.yml')
        if not os.path.exists(opt_path):
            logger.warning(f"DOVER config not found at {opt_path}, skipping video evaluation")
            return None

        with open(opt_path, 'r') as f:
            opt = yaml.safe_load(f)

        # Fix model weight path
        if opt["test_load_path"].startswith("./"):
            opt["test_load_path"] = os.path.join(dover_path, opt["test_load_path"][2:])

        # Check if model weights exist
        if not os.path.exists(opt["test_load_path"]):
            logger.warning(f"DOVER model weights not found, skipping video evaluation")
            return None

        # Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model
        evaluator = DOVER(**opt["model"]["args"]).to(device)
        evaluator.load_state_dict(
            torch.load(opt["test_load_path"], map_location=device)
        )
        evaluator.eval()

        # Setup samplers
        dopt = opt["data"]["val-l1080p"]["args"]
        temporal_samplers = {}
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        # Process video
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])

        views, _ = spatial_temporal_view_decomposition(
            file_path, dopt["sample_types"], temporal_samplers
        )

        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(device)
            )

        # Get scores
        with torch.no_grad():
            results = [r.mean().item() for r in evaluator(views)]

        # Fuse results (TQE + AQE)
        x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
            results[1] + 0.08285
        ) / 0.03774 * 0.3896
        fused_score = 1 / (1 + np.exp(-x))

        # Convert to 0-100 scale
        score = fused_score * 100
        return round(score, 2)

    except Exception as e:
        logger.error(f"Video evaluation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def evaluate_threeD(file_path):
    """Evaluate 3D model quality using threeD.py logic"""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from threeD import load_any, score_object, total_score

        obj = load_any(file_path)
        scores = score_object(obj)
        total = total_score(scores)
        return round(total, 2)

    except Exception as e:
        logger.error(f"3D evaluation failed: {e}")
        return None


def evaluate_document(file_path):
    """Evaluate document quality using document.py logic"""
    try:
        from PIL import Image
        import pytesseract
        import re

        # Get OpenAI client
        client = get_openai_client()
        if client is None:
            logger.error("OpenAI client not available")
            return None

        # Load image
        img = Image.open(file_path)

        # OCR
        text = pytesseract.image_to_string(img)

        # Convert to markdown table
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rows = []
        for line in lines:
            parts = re.split(r"\s{2,}|\t", line)
            rows.append(parts)

        md_lines = []
        if rows:
            header = "| " + " | ".join(rows[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            md_lines.append(header)
            md_lines.append(separator)
            for row in rows[1:]:
                md_lines.append("| " + " | ".join(row) + " |")
        doc_md = "\n".join(md_lines)

        # Build prompt
        prompt = f"""
You are a professional document quality assessor.
Below is the Markdown table from OCR extraction results:
{doc_md}

Please evaluate and provide an overall score (0-100).
Output ONLY a JSON object with this format:
{{"overall_score_100": <int>}}
"""

        # Call gpt-5-mini
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a document quality evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)
        return result_json.get("overall_score_100")

    except Exception as e:
        logger.error(f"Document evaluation failed: {e}")
        return None


def evaluate_text(content):
    """Evaluate text quality using gpt-5-mini"""
    try:
        # Get OpenAI client
        client = get_openai_client()
        if client is None:
            logger.error("OpenAI client not available")
            return None

        prompt = f"""
You are an expert evaluator of open-ended text responses.
Evaluate the TEXT according to the following dimensions (1-5 each):
1. Clarity
2. Coherence & Logic
3. Informativeness & Specificity
4. Conciseness
5. Style & Tone

Compute the Overall Score as the average of the five scores, then convert to 0-100 scale.

Output format (JSON only):
{{
  "clarity": <int>,
  "coherence": <int>,
  "informativeness": <int>,
  "conciseness": <int>,
  "style": <int>,
  "overall": <float>
}}

### TEXT:
{content}
"""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)
        overall = result_json.get("overall", 3.0)
        # Convert from 1-5 scale to 0-100 scale
        score = ((overall - 1) / 4) * 100
        return round(score, 2)

    except Exception as e:
        logger.error(f"Text evaluation failed: {e}")
        return None


def evaluate_code(content):
    """Evaluate code quality using gpt-5-mini"""
    try:
        # Get OpenAI client
        client = get_openai_client()
        if client is None:
            logger.error("OpenAI client not available")
            return None

        code_md = f"```\n{content}\n```"

        prompt = f"""
You are a strict code reviewer.
Evaluate a piece of code in Markdown format.

## Evaluation Dimensions (each scored 0-100):
1. correctness: Syntax soundness, logic consistency
2. readability: Clarity of naming, structure
3. design: Modularity, maintainability
4. performance: Efficiency
5. security: Avoid obvious risks
6. testability: Code can be tested

## Requirements:
- Output ONLY valid JSON
- Do NOT wrap in markdown code blocks
- Keys: "correctness", "readability", "design", "performance", "security", "testability", "overall_score"
- "overall_score" must be the average of all six dimensions

Now evaluate the following code:

{code_md}
"""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)
        return result_json.get("overall_score")

    except Exception as e:
        logger.error(f"Code evaluation failed: {e}")
        return None


def evaluate_modal(modal_type, value, base_dir):
    """
    Evaluate a single modal resource

    Args:
        modal_type: Type of modal (image, audio, video, text, code, document, threeD)
        value: File path or content string
        base_dir: Base directory for resolving relative paths

    Returns:
        Score (float) or None if failed
    """
    logger.info(f"Evaluating {modal_type}: {value[:50] if isinstance(value, str) else value}...")

    try:
        if modal_type in ['image', 'audio', 'video', 'threeD', 'document']:
            # These modals need file paths
            if value.startswith('http://') or value.startswith('https://'):
                file_path = value
            else:
                file_path = os.path.join(base_dir, value)

            if not os.path.exists(file_path) and not value.startswith('http'):
                logger.error(f"File not found: {file_path}")
                return None

            # Call corresponding evaluation function
            if modal_type == 'image':
                return evaluate_image(file_path)
            elif modal_type == 'audio':
                return evaluate_audio(file_path)
            elif modal_type == 'video':
                return evaluate_video(file_path)
            elif modal_type == 'threeD':
                return evaluate_threeD(file_path)
            elif modal_type == 'document':
                return evaluate_document(file_path)

        elif modal_type == 'text':
            return evaluate_text(value)

        elif modal_type == 'code':
            return evaluate_code(value)

        else:
            logger.warning(f"Unknown modal type: {modal_type}")
            return None

    except Exception as e:
        logger.error(f"Evaluation failed for {modal_type}: {e}")
        return None


def process_jsonl(input_file, output_file):
    """
    Process input JSONL file and generate output with scores
    """
    # Read input
    logger.info(f"Reading input file: {input_file}")
    objects = read_multiline_jsonl(input_file)
    logger.info(f"Found {len(objects)} objects to process")

    # Get base directory for resolving relative paths
    base_dir = os.path.dirname(os.path.abspath(input_file))

    # Process each object
    for idx, obj in enumerate(objects):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing object {idx + 1}/{len(objects)} (id: {obj.get('id', 'unknown')})")
        logger.info(f"{'='*60}")

        # Get output.modal
        output_modal = obj.get('output', {}).get('modal', {})

        if not output_modal:
            logger.warning(f"No output.modal found in object {idx + 1}")
            obj['scores'] = {}
            continue

        # Evaluate each modal resource
        scores = {}
        for key, value in output_modal.items():
            modal_type = detect_modal_type(key)

            if modal_type is None:
                logger.warning(f"Unknown modal type for key: {key}")
                scores[key] = None
                continue

            score = evaluate_modal(modal_type, value, base_dir)
            scores[key] = score

            if score is not None:
                logger.info(f"✓ {key} ({modal_type}): {score}")
            else:
                logger.warning(f"✗ {key} ({modal_type}): evaluation failed")

        # Add scores to object
        obj['scores'] = scores

    # Write output
    logger.info(f"\nWriting output file: {output_file}")
    write_multiline_jsonl(objects, output_file)
    logger.info(f"✓ Processing complete! Output saved to: {output_file}")


def main():
    """Main entry point"""
    args = parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Determine output file
    if args.output is None:
        input_path = Path(args.input)
        output_name = input_path.stem + '_output' + input_path.suffix
        args.output = str(input_path.parent / output_name)

    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")

    # Process
    try:
        process_jsonl(args.input, args.output)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
