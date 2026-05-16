"""Interleaved Coherence Evaluation Script.

This script evaluates ICS score using local files.

Example:
python evaluate_ics.py \
  -i "/path/to/input.jsonl" \
  -o "/path/to/output.jsonl" \
  -d "/path/to/data"
"""

import os
import json
import re
import torch
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Import modality-to-text helpers
from x2text_gpt5 import replace_placeholders, image_to_text, document_to_text, code_to_text, video_to_text, audio_to_text, threed_to_text

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def build_modal_path(base_dir, tag, path, processed_domain, subdomain):
    """Build the absolute local path for one modality file.
    """
    # Normalize the root directory
    base_dir = os.path.abspath(base_dir)
    
    # Build the full path
    full_path = os.path.join(base_dir, processed_domain, subdomain, path)
        
    return full_path


def generate_optimized_single_tag_samples():
    samples = {
        # ===== Holistic Coherence examples  =====
        "coherence_5": (
            "A serene forest scene <image:Highly detailed forest with sunlight perfectly filtering through leaves, mossy ground, winding path, birds and river depicted naturally; every element harmonious; line, color, and visual style fully consistent; wording perfectly aligned with visuals; overall presentation flawless; semantic and logical order extremely clear>."
        ),
        "coherence_4": (
            "Forest scene mostly coherent <image:Green trees with uneven sunlight, ground and path roughly sketched, birds slightly small; some minor inconsistencies in colors and line style; wording mostly aligns with visuals; local repetitions or small jumps exist but understanding not affected; overall coherent>."
        ),
        "coherence_3": (
            "Forest scene partially consistent <image:Trees roughly drawn, uneven proportions; birds and river in slightly different style; colors partially inconsistent; wording partially deviates from image tone; local logic jumps present; requires some reasoning to fully understand>."
        ),
        "coherence_2": (
            "Forest scene low coherence <image:Trees bright neon green, birds cartoonish, river pink; ground unrealistic; terminology mixed; visual elements conflicting; reading experience difficult; multiple contradictions>."
        ),
        "coherence_1": (
            "Completely incoherent scene <image:A cityscape with unrelated desert, river, and random birds; style chaotic; terminology wrong; wording and visual style entirely uncoordinated; logic collapsed; almost impossible to understand>."
        ),

        # ===== Stylistic Harmony examples =====
        "style_5": (
            "A serene forest scene <image:Highly detailed forest with tall green trees, sunlight perfectly filtering through leaves, mossy ground, winding path, birds and river depicted naturally; all elements harmonious; colors balanced and realistic; line work clean and consistent; wording precisely aligned with visuals; expression and visual style fully coordinated; overall immersive and smooth>."
        ),
        "style_4": (
            "Forest scene mostly consistent in style <image:Green trees under partly cloudy sunlight, ground and path clearly visible; birds perched naturally; river reflecting surroundings; brushstroke thickness varies slightly; some colors slightly stylized; minor deviations in shading style; terminology mostly consistent; overall reading experience coherent>."
        ),
        "style_3": (
            "Forest scene partially consistent <image:Trees unevenly shaped, some branches disproportionate; river and birds drawn simpler than trees; colors slightly clashing; wording occasionally deviates in tone from image; visual style not uniform; terminology partially mixed; overall understandable but smoothness reduced>."
        ),
        "style_2": (
            "Forest scene inconsistent in style <image:Trees neon green, river tinted pink; birds cartoonish while trees semi-realistic; ground unrealistic with flat shading; terminology not uniform; visual style conflicting; wording and visuals partially misaligned; comprehension affected>."
        ),
        "style_1": (
            "Completely inconsistent style <image:A cityscape mixed with desert, random river and cartoon birds; all elements in clashing styles; terminology entirely wrong; wording and image unaligned; expression and visual style chaotic; presentation almost impossible to interpret>."
        ),
    }
    return samples

def evaluate_with_gpt(text_content, model="gpt-5-mini"):
    """Call the judge model to score Holistic Coherence and Stylistic Harmony.
    """
    samples = generate_optimized_single_tag_samples()
    
    prompt = f"""
You are a strict evaluation assistant. Evaluate the following text according to the criteria below.

[Evaluation Dimensions & Scoring Guidelines]

1. Coherence (1–5 points)
- 5: 
  1. Output **highly matches** the input, multimodal references **accurate and specific**;
  2. Different modality information **complements each other** without noticeable contradictions;
  3. **Logic is rigorous**, structure is reasonable, overall semantic/logical order is **clear**;
  4. Interleaved multimodal tags **naturally ordered**, reading/understanding experience **smooth**.
- 4: 
  1. Output **mostly matches** the input, references **generally reasonable**;
  2. Different modality information **mostly complementary**, minor omissions or redundancy;
  3. Overall logic **coherent**, local repetitions or jumps minor, not affecting understanding;
  4. Interleaved tags **mostly ordered**, minor adjustments do not affect understanding.
- 3:
  1. Output has **general relation** to input, references are **vague or partially unclear**;
  2. Some modality blocks **repeated or missing**, minor contradictions exist;
  3. Local logic jumps or slight contradictions, **requires extra reasoning**;
  4. Interleaved tags **partially disordered**, local understanding may be difficult.
- 2:
  1. Output **lowly matches** the input, most references **vague or incorrect**;
  2. Cross-modal information **repetitive or conflicting**;
  3. Logic **obviously chaotic**, many contradictions;
  4. Interleaved tag order **clearly disordered**, understanding difficult.
- 1:
  1. Output is **almost irrelevant** to input, references **missing or wrong**;
  2. Different modalities **barely complement**, major contradictions;
  3. Logic **completely collapsed**, frequent contradictions;
  4. Interleaved tag order **completely disordered**, hard to understand.

2. Style Consistency (1–5 points)
- 5:
  1. Style **highly consistent**, cross-modal narration **uniform**, tone and sentence structures **without deviation**;
  2. Terminology **fully consistent**, key concepts, tags, and naming **completely aligned**;
  3. Expression and visual style **fully aligned**, wording, sentence structures, and rhetorical/visual style **coordinated**, overall experience **smooth**.
- 4:
  1. Style **mostly consistent**, minor deviations;
  2. Terminology **generally consistent**, key concepts, tags, and naming **mostly aligned**;
  3. Expression and visual style **mostly aligned**, minor deviations, overall experience **unaffected**.
- 3:
  1. Style **partially consistent**, noticeable differences in narration;
  2. Terminology **partially mixed**, key concepts, tags, naming **sometimes inconsistent**;
  3. Expression and visual style **partially aligned**, wording, sentence structures, or rhetorical/visual style **obviously deviate**, reading/watching experience affected.
- 2:
  1. Style **inconsistent**, cross-modal narration **uncoordinated**;
  2. Terminology **not uniform**, key concepts, tags, naming **frequently mixed**;
  3. Expression and visual style **not aligned**, wording, sentence structures, or rhetorical/visual style **conflicting**, understanding affected.
- 1:
  1. Style **completely inconsistent**, cross-modal narration **chaotic**;
  2. Terminology **completely wrong**, key concepts, tags, naming **incorrect or inconsistent**;
  3. Expression and visual style **completely chaotic**, wording, sentence structures, or rhetorical/visual style **extremely uncoordinated**, almost impossible to understand.

[Sample References for Scoring]

Coherence Samples:
5: {samples['coherence_5']}
4: {samples['coherence_4']}
3: {samples['coherence_3']}
2: {samples['coherence_2']}
1: {samples['coherence_1']}

Style Consistency Samples:
5: {samples['style_5']}
4: {samples['style_4']}
3: {samples['style_3']}
2: {samples['style_2']}
1: {samples['style_1']}

[Task Requirement]
Strictly assign a score (1–5) for each dimension. 
Return JSON only, e.g.:
{{
  "coherence": 4,
  "style_consistency": 5
}}

[Text to Evaluate]
{text_content}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evaluation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_output = response.choices[0].message.content.strip()

        # Strip optional fenced JSON wrappers 
        raw_output = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())

        # Parse the model output as JSON 
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            print("Warning: could not parse JSON from model output:", raw_output)
            return {"raw": raw_output}
    except Exception as e:
        print("Error calling GPT-5-mini:", e)
        return None

def stream_json_objects(file_path):
    """Yield JSON objects from a JSONL-like file, allowing multi-line items.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
            buffer += line
            try:
                # Try to parse one complete object
                item = json.loads(buffer)
                yield item
                buffer = ""
            except json.JSONDecodeError:
                continue

def extract_caption(full_text):
    """Extract the useful caption body from model output.
    """
    marker = "assistant\n"
    if marker in full_text:
        caption = full_text.split(marker, 1)[1]
        return caption.strip()  # Trim leading/trailing whitespace
    else:
        # Fall back to the original text
        return full_text.strip()


if __name__ == "__main__":
    # Map placeholder prefixes to caption functions
    fn_map = {
        "image": image_to_text,
        "document": document_to_text,
        "code": code_to_text,
        "video": video_to_text,
        "audio": audio_to_text,
        "threeD": threed_to_text
    }
    # Default captioning model for each modality
    model_map = {
        "image": "gpt-5-mini",
        "document": "gpt-5-mini",
        "code": "gpt-5-mini",
        "video": "Qwen/Qwen2.5-Omni-3B",
        "audio": "Qwen/Qwen2.5-Omni-3B",
        "threeD": "gpt-5-mini",
    }

    parser = argparse.ArgumentParser(description="Evaluate multimodal content for coherence and style consistency")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--base_dir", "-d", required=True, help="Base directory for local modality files")
    args = parser.parse_args()

    file_path = args.input
    output_file = args.output
    base_data_dir = args.base_dir
    
    open(output_file, "w").close()  # Reset output file

    for item in stream_json_objects(file_path):
        domain = item.get("domain")
        subdomain = item.get("subdomain")
        rec_id = item.get("id")
        output_content = item.get("output", {}).get("content", "")
        modal_map_raw = item.get("output", {}).get("modal", {})
        original_domain = item.get("domain")
        processed_domain = original_domain

        # Resolve local file paths for each placeholder
        modal_map = {}
        for tag, path in modal_map_raw.items():
            modal_map[tag] = build_modal_path(base_data_dir, tag, path, processed_domain, subdomain)
            
            # Record missing files instead of stopping the whole run
            if not os.path.exists(modal_map[tag]):
                print(f"[ERROR] Local file not found for {tag}: {modal_map[tag]}. Skipping.")
                modal_map[tag] = None 

        # Generate dense captions for each placeholder
        modal_text_map = {}
        for tag, local_path in modal_map.items():
            if local_path is None:
                modal_text_map[tag] = f"File {tag} not found at specified path."
                continue

            prefix_match = re.match(r"[a-zA-Z]+", tag)
            if prefix_match:
                prefix = prefix_match.group(0)
                fn = fn_map.get(prefix)
                 
                # The 3D captioner benefits from extra context
                if prefix == "threeD":
                    print(f"Processing {tag} ({prefix}) -> {local_path} with subdomain={subdomain}")
                    with torch.no_grad():
                        raw_caption = fn(
                            local_path,
                            model_map.get(prefix),
                            subdomain=subdomain,
                            content=output_content 
                        )
                    torch.cuda.empty_cache()
                    modal_text_map[tag] = extract_caption(raw_caption)
                elif fn:
                    print(f"Processing {tag} ({prefix}) -> {local_path}")
                    with torch.no_grad(): 
                        raw_caption = fn(local_path, model_map.get(prefix))
                    torch.cuda.empty_cache() 
                    modal_text_map[tag] = extract_caption(raw_caption)
                else:
                    modal_text_map[tag] = local_path 
            else:
                modal_text_map[tag] = local_path

        # Replace placeholders with generated captions
        final_text = replace_placeholders(
            text=output_content,
            code_map=modal_text_map,
            fn_map=None,
            model_map=None,
            modal_map=None
        )

        # Ask the judge model for final scores
        score = evaluate_with_gpt(final_text)

        # Fall back to conservative defaults on malformed outputs
        if not isinstance(score, dict) or not score:
            score = {"coherence": 1, "style_consistency": 1}
        else:
            # Fill missing score fields
            score.setdefault("coherence", 1)
            score.setdefault("style_consistency", 1)

        # Write one result record
        result_obj = {
            "domain": domain,
            "subdomain": subdomain,
            "id": rec_id,
            "content": final_text,
            "modal": modal_text_map,
            "difficulty_level": item.get("difficulty_level"),
            "score": score
        }
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result_obj, ensure_ascii=False, indent=4) + "\n")

        print(f"[Written] domain={domain}, subdomain={subdomain}, id={rec_id}")
