"""X-to-text utilities for interleaved coherence evaluation.
all x_to_text functions operate on local files and return text descriptions.
"""

import os
import base64
import warnings
import logging
import io
import re
import gc
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import easyocr
import open3d as o3d
from openai import OpenAI

# ==========================================================
# Global configuration
# ==========================================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().setLevel(logging.ERROR)
load_dotenv()

# OpenAI-compatible endpoint served by vLLM
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
client_vllm = OpenAI(api_key=os.getenv("VLLM_API_KEY", "EMPTY"), base_url=VLLM_API_BASE)

# OpenAI client for GPT-based captioning and summarization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

reader = easyocr.Reader(['en'], gpu=False)

# ==========================================================
# OCR and helper utilities
# ==========================================================
def ocr_extract_text(local_path):
    """Run OCR on a local file when needed"""
    try:
        text_list = reader.readtext(local_path, detail=0)
        return ' '.join(text_list)
    except Exception as e:
        print(f"OCR failed for {local_path}: {e}")
        return ""

# ==========================================================
# Image -> text
# ==========================================================
def image_to_text(image_path, model_name="gpt-5-mini"):
    """Generate a dense caption for one image"""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_payload = f"data:image/png;base64,{b64}"

        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Please generate a dense caption for this image."},
                        {"type": "input_image", "image_url": image_payload}
                    ]
                }
            ],
            max_output_tokens=500
        )

        caption = getattr(response, "output_text", "")
        return caption or "[WARN] No caption found"
    except Exception as e:
        print("Error generating image caption:", e)
        return ""

# ==========================================================
# CSV -> rendered table image -> text
# ==========================================================
def csv_to_png_base64(csv_path):
    """Render a CSV file as a PNG table and return base64"""
    try:
        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(min(12, len(df.columns) * 1.2), 6))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print("Error converting CSV to PNG:", e)
        return None

# ==========================================================
# Document -> text 
# ==========================================================
def document_to_text(local_path, model_name="gpt-5-mini"):
    """Generate a dense caption for a document-like file"""
    file_ext = Path(local_path).suffix.lower()
    extracted_text = ""
    img_base64 = None

    if file_ext == ".csv":
        img_base64 = csv_to_png_base64(local_path)
    elif file_ext not in {".pdf", ".docx", ".txt"}:
        extracted_text = ocr_extract_text(local_path)

    prompt = f"""
You are a visual and document captioning assistant.
Below is a document and its extracted text.
Generate a **dense description** covering layout and content.
[File] {local_path}
[Extracted Text] {extracted_text if extracted_text else "N/A"}
Return only the dense caption.
"""
    messages = [{"role": "system", "content": "You are a strict visual captioning assistant."},
                {"role": "user", "content": prompt}]
    if img_base64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "Here is the CSV rendered as an image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]})

    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error generating dense caption:", e)
        return ""

# ==========================================================
# Code -> text
# ==========================================================
def code_to_text(code_snippet, model="gpt-5-mini"):
    """Summarize a code snippet or source file content"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise programming assistant."},
                {"role": "user", "content": code_snippet}
            ],
            temperature=0.2, max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Failed to summarize code: {e}"

# ==========================================================
# Video -> text
# ==========================================================
def video_to_text(video_path, model_name="Qwen/Qwen2.5-Omni-3B", summary=True, chunk_size_mb=8):
    """Caption a video via a local vLLM endpoint, with optional chunking.
    - auto-splits large payloads 
    - captions each chunk independently
    - optionally summarizes the merged result
    """

    try:
        if not os.path.exists(video_path):
            return f"[ERROR] Video not found: {video_path}"

        # Step 1: encode the video as base64
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode("utf-8")
        total_size_mb = len(b64) / 1e6
        print(f"[Encode] Video encoded to base64 ({total_size_mb:.2f} MB text)")

        # Step 2: split large payloads into chunks
        chunk_len = chunk_size_mb * 1_000_000
        chunks = [b64[i:i + int(chunk_len)] for i in range(0, len(b64), int(chunk_len))]
        print(f"[Chunk] Split into {len(chunks)} chunks of ~{chunk_size_mb}MB each.")

        partial_captions = []

        # Step 3: caption each chunk independently
        for idx, chunk in enumerate(chunks):
            print(f"[Infer] Processing chunk {idx+1}/{len(chunks)} ...")
            video_payload = f"data:video/mp4;base64,{chunk}"

            try:
                resp = client_vllm.chat.completions.create(
                    model=model_name,
                    max_completion_tokens=1024,
                    temperature=0.2,
                    messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Provide a highly detailed and coherent description of this full video strictly in English. "
                                    "Describe all visible scenes, actions, people, objects, emotions, environments, and transitions comprehensively, preserving the full meaning and visual context of the video. "
                                    "Do not omit important information about movement, mood, or setting. "
                                    "Do not use any non-English characters. "
                                    "Do not use any quotation marks (' or \"). "
                                    "Do not use any angle brackets (< or >). "
                                    "Do not use double quotes under any circumstances. "
                                    "Avoid punctuation, symbols, or formatting that are not part of standard English writing. "
                                    "Write in complete, fluent, and natural English sentences only."
                                )
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": video_payload}
                            }
                        ]
                    }
                ]
                )

                caption = resp.choices[0].message.content.strip()
                print(f"[OK] Received caption for chunk {idx+1} ({len(caption)} chars)")
                partial_captions.append(f"[Segment {idx+1}] {caption}")

                # Release GPU cache between chunks
                if "torch" in globals():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"[WARN] Chunk {idx+1} failed: {e}")
                partial_captions.append(f"[Segment {idx+1}] [ERROR: {e}]")

        # Step 4: merge partial captions
        dense_caption = "\n".join(partial_captions)
        print(f"[Merge] Merged {len(partial_captions)} partial captions (total {len(dense_caption)} chars).")

        # Step 5: optionally compress into one sentence
        if summary and dense_caption.strip():
            try:
                resp2 = client.chat.completions.create(
                model="gpt-5-mini",
                messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise English summarizer. "
                        "You must produce a single, grammatically correct English sentence summarizing the input description. "
                        "Do not use any non-English characters. "
                        "Do not use any quotation marks (' or \"). "
                        "Do not use any angle brackets (< or >). "
                        "Do not use double quotes under any circumstances. "
                        "Avoid punctuation or symbols that are not part of standard English writing."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following description clearly and naturally in one English sentence:\n\n{dense_caption}"
                    )
                }
            ]
            )
                summary_text = resp2.choices[0].message.content.strip()
                print("[Summary] ✅ Successfully generated concise summary.")
                return summary_text
            except Exception as e:
                print(f"[WARN] Summary step failed: {e}")
                return dense_caption
        else:
            return dense_caption

    except Exception as e:
        print("Error in video_to_text (chunked):", e)
        return f"[ERROR] Failed to process video: {e}"


# ==========================================================
# Audio -> text
# ==========================================================
def audio_to_text(audio_path, model_name="Qwen/Qwen2.5-Omni-3B"):
    """Transcribe and optionally summarize a local audio file"""
    try:
        if not os.path.exists(audio_path):
            return f"[ERROR] Audio not found: {audio_path}"

        with open(audio_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode("utf-8")
        print(f"[Encode] Audio encoded to base64 ({len(b64_audio)/1e6:.2f} MB)")
        audio_payload = f"data:audio/mp3;base64,{b64_audio}"

        resp = client_vllm.chat.completions.create(
            model=model_name,
            max_completion_tokens=1024,
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the speech in this audio clip into complete English text accurately and fully. "
                            "Write exactly what is spoken, preserving every word and sentence from the speaker's voice. "
                            "Do not summarize, paraphrase, or interpret the meaning. "
                            "Do not include any information about background noise, music, sound effects, or non-speech elements. "
                            "Ignore filler sounds such as 'uh', 'um', or background chatter unless they are part of the spoken content. "
                            "Do not use any non-English characters. "
                            "Do not use any quotation marks (' or \"). "
                            "Do not use any angle brackets (< or >). "
                            "Do not use double quotes under any circumstances. "
                            "Avoid punctuation, symbols, or formatting that are not part of standard English writing. "
                            "Output only the transcribed English speech in continuous, natural sentences."
                        )
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_payload}
                    }
                ]
            }
        ]
    )
        dense_caption = resp.choices[0].message.content.strip()
        try:
            resp2 = client.chat.completions.create(
                model="gpt-5-mini",
                messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise English summarizer. "
                        "You must produce a single, grammatically correct English sentence summarizing the input description. "
                        "Focus primarily on preserving the full semantic meaning and key message of the original speech or audio content. "
                        "Ensure that essential ideas, context, and intent are fully retained. "
                        "Tone and emotion may be reflected briefly only if they are necessary to convey meaning, "
                        "but they must not override the semantic accuracy of the summary. "
                        "Do not use any non-English characters. "
                        "Do not use any quotation marks (' or \"). "
                        "Do not use any angle brackets (< or >). "
                        "Do not use double quotes under any circumstances. "
                        "Avoid punctuation or symbols that are not part of standard English writing."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following description clearly and naturally in one English sentence:\n\n{dense_caption}"
                    )
                }
            ]
            )
            summary_text = resp2.choices[0].message.content.strip()
            return summary_text

        except Exception as e:
            print("[WARN] Summary step failed:", e)
            return dense_caption
    except Exception as e:
        print("Error in audio_to_text:", e)
        return f"[ERROR] Failed to process audio: {e}"


# ==========================================================
# 3D file -> text
# ==========================================================
def threed_to_text(local_path, model_name="gpt-4o-mini", subdomain=None, content=None):
    """Render a 3D file into projections and caption it"""
    file_ext = Path(local_path).suffix.lower()
    try:
        # Step 1: read point cloud or mesh
        if file_ext == ".ply":
            pcd = o3d.io.read_point_cloud(local_path)
            points = np.asarray(pcd.points)
        elif file_ext == ".off":
            mesh = o3d.io.read_triangle_mesh(local_path)
            mesh.compute_vertex_normals()
            points = np.asarray(mesh.vertices)
        else:
            return f"[WARN] Unsupported file type: {file_ext}"

        # Step 2: render three orthogonal projections
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap="viridis")
        axs[1].scatter(points[:, 0], points[:, 2], s=1, c=points[:, 1], cmap="viridis")
        axs[2].scatter(points[:, 1], points[:, 2], s=1, c=points[:, 0], cmap="viridis")
        for a, title in zip(axs, ["XY", "XZ", "YZ"]):
            a.set_title(f"{title} Projection"); a.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=72, bbox_inches="tight")
        plt.close()

        # Step 3: convert the rendered image to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_payload = f"data:image/png;base64,{img_base64}"

        # Step 4: build the captioning prompt
        user_prompt = f"""
You are a visual captioning assistant.
Given the image below (a three-view projection of a 3D object), generate a **highly detailed and dense caption**.

- The object type is: {subdomain}.
- Use the following extra context if useful: {content}.
- Do not describe raw points or dots. Instead, infer the real-world object this point cloud represents.
- Write a coherent paragraph that clearly conveys the object's identity, shape, components, spatial structure, and possible function.
- Return only the caption text; no extra explanation or markup.
        """

        # Step 5: call the OpenAI vision model
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_payload}
                    ]
                }
            ],
            max_output_tokens=500
        )

        # Step 6: parse the SDK response
        if hasattr(response, "output_text") and response.output_text:
            caption = response.output_text.strip()
        else:
            for item in getattr(response, "output", []) or []:
                if hasattr(item, "content") and item.content:
                    for c in item.content:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            text_piece = getattr(c, "text", "")
                            if isinstance(text_piece, str) and text_piece.strip():
                                caption += text_piece.strip() + "\n"

        if not caption.strip():
            print("[WARN] No caption found, raw response:")
            print(response.model_dump_json(indent=2))
        return caption.strip() or "[WARN] Empty caption"

    except Exception as e:
        return f"[ERROR] 3D caption failed: {e}"

# ==========================================================
# Placeholder replacement / 占位符替换
# ==========================================================
def replace_placeholders(text, code_map=None, fn_map=None, model_map=None, modal_map=None):
    """Replace placeholders like <image1> with generated captions / 将占位符替换为生成好的描述文本。"""
    if code_map is None:
        code_map = {}
    if modal_map is None:
        modal_map = {}

    def replacer(match):
        tag = match.group(0).strip("<> \n")
        if tag in code_map:
            return f"<{tag}:{code_map[tag]}>"
        if tag in modal_map:
            path_or_url = modal_map[tag]
            prefix = re.match(r"[a-zA-Z]+", tag)
            if prefix and fn_map and prefix.group(0) in fn_map:
                fn = fn_map[prefix.group(0)]
                model_name = model_map.get(prefix.group(0)) if model_map else None
                caption = fn(path_or_url, model_name)
                return f"<{tag}:{caption}>"
            return f"<{tag}:{path_or_url}>"
        return match.group(0)

    return re.sub(r"<(code|image|document|audio|video|threeD)\d+>", replacer, text)
