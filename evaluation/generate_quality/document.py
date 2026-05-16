import pytesseract
from PIL import Image
import requests
from openai import OpenAI
import io
import json
import re
import os

# ============ Configuration Area ============
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Specify Tesseract path if needed
# =================================

def load_image(path_or_url):
    """Load image from local path or URL and return PIL.Image object"""
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        # Download from URL
        response = requests.get(path_or_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        # Load from local path
        return Image.open(path_or_url)

def image_to_text(img):
    """Extract text using OCR"""
    return pytesseract.image_to_string(img)

def text_to_markdown_table(text):
    """Convert OCR text roughly to Markdown table"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []
    for line in lines:
        parts = re.split(r"\s{2,}|\t", line)  # Split by multiple spaces/tabs
        rows.append(parts)

    md_lines = []
    if rows:
        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        md_lines.append(header)
        md_lines.append(separator)
        for row in rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)

def build_prompt(doc_md, heuristics=""):
    """Construct the prompt"""
    return f"""
You are a professional document quality assessor, specializing in evaluating the expression and presentation quality of tabular documents.
You will not evaluate whether the facts are correct, nor will you evaluate pixel clarity of images, but only focus on the expression quality of the tabular document itself.

Below is the Markdown table from OCR extraction results:
[DOC_MD]
{doc_md}

Below are some heuristic detection results:
[HEURISTICS]
{heuristics}

Please evaluate according to the following requirements:
- Score on six dimensions, 0-5 points for each dimension, and provide brief reasons:
  1) Clarity: How immediately understandable are headers/row labels/values? Are terms unambiguous and concise?
  2) Structure: Presence and correctness of title, header row, logical row/column grouping, and overall organization.
  3) Consistency: Uniformity of units, decimal places, capitalization, punctuation, and naming conventions.
  4) Readability: Visual alignment/readability as inferred from Markdown structure (e.g., clear headers, spacing, grouping cues).
  5) Data Presentation: Comparability and numeric hygiene: appropriate decimal precision, column alignment, units close to values, concise notation.
  6) Usefulness: Self-containment for the intended audience: enough context (title/labels/notes) to use the table standalone.
- Finally provide an overall score (0-100) using weighted average:
  Clarity 0.20, Structure 0.15, Consistency 0.20, 
  Readability 0.15, Data Presentation 0.20, Usefulness 0.10
- Output must be in JSON format, containing:
  - overall_score_100
  - dimensions (each dimension contains score_5 and rationale)
  - evidence_notes
"""

def call_4o(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",   # or gpt-4o-mini
        messages=[
            {"role": "system", "content": "You are a document quality evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def main(image_path):
    # 1. Load image (supports both local path and URL)
    img = load_image(image_path)

    # 2. OCR
    text = image_to_text(img)
    print("🔍 OCR extraction results:\n", text)

    # 3. Convert to Markdown
    doc_md = text_to_markdown_table(text)
    print("\n📑 Converted Markdown table:\n", doc_md)

    # 4. Heuristic detection
    heuristics = []
    decimals = re.findall(r"\d+\.\d+", text)
    if decimals:
        lengths = [len(d.split(".")[1]) for d in decimals]
        if len(set(lengths)) > 1:
            heuristics.append("Inconsistent decimal places")
    if not re.search(r"(g|kcal|mg|kg)", text, re.IGNORECASE):
        heuristics.append("Numerical values lack unit specification")
    heuristics_text = "; ".join(heuristics) if heuristics else "No obvious issues found"

    # 5. Construct prompt
    prompt = build_prompt(doc_md, heuristics_text)

    # 6. Call 4o
    result = call_4o(prompt)

    # 7. Output JSON
    try:
        result_json = json.loads(result)
    except Exception:
        print("⚠️ Model did not return valid JSON, raw output:")
        print(result)
        return
    print("\n✅ Quality assessment results (JSON):")
    print(json.dumps(result_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Use command line argument
        image_path = sys.argv[1]
    else:
        # Default example (URL)
        image_path = "https://raw.githubusercontent.com/Shize-ZHANG/Any2Any-Interleaved-Data-Pipline/main/original_data/document/doc_0791_01.png"

    main(image_path)
