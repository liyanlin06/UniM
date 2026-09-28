from __future__ import annotations

import datetime as dt
import html
import os
import queue
import re
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from urllib.parse import quote


def _configure_gradio_temp_dir() -> None:
    """Force Gradio temp data into a writable project-local directory."""
    project_temp = Path(__file__).resolve().parent / ".gradio_tmp"
    configured = Path(os.getenv("GRADIO_TEMP_DIR", str(project_temp))).expanduser()

    chosen = configured
    try:
        configured.mkdir(parents=True, exist_ok=True)
        probe = configured / ".write_probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
    except Exception:
        project_temp.mkdir(parents=True, exist_ok=True)
        chosen = project_temp

    os.environ["GRADIO_TEMP_DIR"] = str(chosen.resolve())


_configure_gradio_temp_dir()

import gradio as gr

from unima import PipelineCallbacks, run_interleave_session, warmup_local_av_model

INPUT_TOKEN_RE = re.compile(r"(\[(?:IMG|AUD|VID)\d+\])")
INPUT_TOKEN_FULL_RE = re.compile(r"\[(?:IMG|AUD|VID)\d+\]")
OUTPUT_TAG_RE = re.compile(r"(<(?:image|audio|video)\d+>)")
OUTPUT_TAG_FULL_RE = re.compile(r"<(?:image|audio|video)\d+>")

ICON_BY_MODALITY = {
    "image": "🖼",
    "audio": "🎵",
    "video": "🎬",
}

PREFIX_BY_MODALITY = {
    "image": "IMG",
    "audio": "AUD",
    "video": "VID",
}

EMPTY_COUNTERS = {"image": 0, "audio": 0, "video": 0}

CSS = """
:root {
  --bg: #070b16;
  --panel: #0e1426;
  --panel-soft: #121b33;
  --line: #2c3a5f;
  --text: #f4f7ff;
  --muted: #9fb0d6;
  --primary: #22d3ee;
  --user: #12325d;
  --assistant: #171d2e;
}

html, body, .gradio-container {
  background: radial-gradient(circle at 15% -10%, #1e293b 0%, var(--bg) 42%, #060a14 100%);
  color: var(--text);
}

.panel-card {
  background: linear-gradient(180deg, #121a31 0%, #0d1527 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px;
}

.chat-shell {
  height: 62vh !important;
  max-height: 62vh !important;
  overflow-y: auto;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px;
  background: #0a1020;
  scroll-behavior: smooth;
}

#chat_view {
  min-height: 100%;
}

.chat-empty {
  color: var(--muted);
  text-align: center;
  margin: 26px 0;
}

.chat-msg {
  display: flex;
  margin: 10px 0;
}

.chat-msg.user {
  justify-content: flex-end;
}

.chat-msg.assistant {
  justify-content: flex-start;
}

.bubble {
  max-width: min(920px, 88%);
  border-radius: 14px;
  border: 1px solid var(--line);
  padding: 10px 12px;
  color: var(--text);
  overflow: visible;
}

.chat-msg.user .bubble {
  background: linear-gradient(180deg, #194173 0%, var(--user) 100%);
}

.chat-msg.assistant .bubble {
  background: linear-gradient(180deg, #1d2740 0%, var(--assistant) 100%);
}

.bubble-meta {
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 6px;
}

.txt-seg {
  white-space: pre-wrap;
  line-height: 1.5;
}

.mm-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  margin: 2px 4px 2px 0;
  border-radius: 999px;
  border: 1px solid #315387;
  background: #1a2a48;
  font-size: 12px;
  line-height: 1.4;
  color: #dbeafe;
  vertical-align: middle;
}

.mm-chip.image {
  border-color: #2563eb;
  background: #172554;
}

.mm-chip.audio {
  border-color: #0ea5e9;
  background: #0f2f4a;
}

.mm-chip.video {
  border-color: #14b8a6;
  background: #11343a;
}

.mm-chip.pending {
  border-color: #64748b;
  background: #1e293b;
}

.mm-pop {
  display: inline-block;
  position: relative;
  margin-right: 4px;
}

.mm-pop > summary {
  list-style: none;
  cursor: pointer;
}

.mm-pop > summary::-webkit-details-marker {
  display: none;
}

.mm-pop-body {
  position: relative;
  width: min(360px, 100%);
  margin-top: 8px;
  background: #0b1223;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
}

.mm-pop-body .media-link {
  display: inline-block;
  margin-top: 8px;
  font-size: 12px;
  color: #93c5fd;
  text-decoration: none;
}

.mm-pop-body .media-link:hover {
  text-decoration: underline;
}

.mm-pop-body img,
.mm-pop-body video {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #334155;
}

.mm-pop-body audio {
  width: 100%;
}

.status-badge {
  color: #dbeafe;
  background: #1d2f52;
  border: 1px solid #365a93;
  border-radius: 10px;
  padding: 8px 10px;
}
"""

JS_INSERT_CURSOR = """
(text, media_state, counters, file_value) => {
  try {
    const root = document.querySelector('#draft_textarea');
    const ta = root ? root.querySelector('textarea') : null;
    const raw = text || "";
    if (!ta) {
      return [raw, media_state, counters, file_value];
    }
    const start = Number.isInteger(ta.selectionStart) ? ta.selectionStart : raw.length;
    const end = Number.isInteger(ta.selectionEnd) ? ta.selectionEnd : start;
    const next = raw.slice(0, start) + "[CURSOR]" + raw.slice(end);
    return [next, media_state, counters, file_value];
  } catch (e) {
    return [text || "", media_state, counters, file_value];
  }
}
"""

JS_FORCE_ENGLISH_UPLOAD_HINTS = r"""
() => {
  const zhToEn = new Map([
    ["\u5c06\u56fe\u7247\u62d6\u81f3\u5230\u6b64\u5904 \u6216\u70b9\u51fb\u4e0a\u4f20", "Drop image here or click to upload"],
    ["\u5c06\u97f3\u9891\u62d6\u81f3\u5230\u6b64\u5904 \u6216\u70b9\u51fb\u4e0a\u4f20", "Drop audio here or click to upload"],
    ["\u5c06\u89c6\u9891\u62d6\u81f3\u5230\u6b64\u5904 \u6216\u70b9\u51fb\u4e0a\u4f20", "Drop video here or click to upload"],
  ]);

  const normalize = (s) => (s || "").replace(/\s+/g, " ").trim();

    const replaceInNode = (node) => {
        if (!node) return;
        if (node.nodeType === Node.TEXT_NODE) {
            const text = node.nodeValue || "";
            const normalized = normalize(text);
            if (zhToEn.has(normalized)) {
                node.nodeValue = zhToEn.get(normalized);
            }
            return;
        }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    for (const child of node.childNodes) {
      replaceInNode(child);
    }
  };

  const run = () => replaceInNode(document.body);
  run();
  const obs = new MutationObserver(run);
  obs.observe(document.body, { childList: true, subtree: true, characterData: true });
}
"""


def _new_counters() -> Dict[str, int]:
    return dict(EMPTY_COUNTERS)


def _escape(text: str) -> str:
    return html.escape(text or "")


def _extract_file_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name"):
            v = value.get(key)
            if isinstance(v, str):
                return v
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                return item
    return ""


def _modality_from_input_token(token: str) -> str:
    if token.startswith("[IMG"):
        return "image"
    if token.startswith("[AUD"):
        return "audio"
    if token.startswith("[VID"):
        return "video"
    return ""


def _modality_from_output_tag(tag: str) -> str:
    if tag.startswith("<image"):
        return "image"
    if tag.startswith("<audio"):
        return "audio"
    if tag.startswith("<video"):
        return "video"
    return ""


def _file_url(path: str) -> str:
    return "/gradio_api/file=" + quote(str(Path(path).expanduser().resolve()), safe="/")


def _media_chip_html(modality: str, label: str, filename: str = "", pending: bool = False) -> str:
    icon = ICON_BY_MODALITY.get(modality, "📎")
    cls = f"mm-chip {modality}"
    if pending:
        cls += " pending"
    title = _escape(filename) if filename else ""
    safe_label = _escape(label)
    suffix = f" · {_escape(filename)}" if filename else ""
    return f"<span class='{cls}' title='{title}'>{icon} {safe_label}{suffix}</span>"


def _media_popup_html(modality: str, label: str, path: str, filename: str = "") -> str:
    url = _file_url(path)
    chip = _media_chip_html(modality, label, filename=filename)
    if modality == "image":
        preview = f"<img src='{url}' alt='{_escape(label)}' />"
    elif modality == "video":
        preview = f"<video src='{url}' controls preload='metadata'></video>"
    elif modality == "audio":
        preview = f"<audio src='{url}' controls></audio>"
    else:
        preview = f"<div class='txt-seg'>{_escape(path)}</div>"
    link = f"<a class='media-link' href='{url}' target='_blank' rel='noopener'>Open media in new tab</a>"
    return (
        "<details class='mm-pop'>"
        f"<summary>{chip}</summary>"
        f"<div class='mm-pop-body'>{preview}{link}</div>"
        "</details>"
    )


def _render_input_interleave(text: str, media_state: Dict[str, Dict[str, str]], *, with_popup: bool) -> str:
    parts = INPUT_TOKEN_RE.split(text or "")
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if INPUT_TOKEN_FULL_RE.fullmatch(part):
            info = media_state.get(part)
            modality = _modality_from_input_token(part)
            if info and modality:
                filename = info.get("name", "")
                path = info.get("path", "")
                if with_popup and path:
                    out.append(_media_popup_html(modality, part, path, filename=filename))
                else:
                    out.append(_media_chip_html(modality, part, filename=filename))
            else:
                out.append(_media_chip_html(modality or "video", part, pending=True))
            continue
        if part.strip():
            out.append(f"<span class='txt-seg'>{_escape(part)}</span>")
    if not out:
        return "<span class='txt-seg'>(empty)</span>"
    return "".join(out)


def _render_output_interleave(text: str, artifact_map: Dict[str, Dict[str, str]] | None) -> str:
    parts = OUTPUT_TAG_RE.split(text or "")
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if OUTPUT_TAG_FULL_RE.fullmatch(part):
            modality = _modality_from_output_tag(part)
            info = (artifact_map or {}).get(part)
            if info:
                p = Path(str(info["path"])).expanduser().resolve()
                if p.is_file():
                    out.append(_media_popup_html(modality, part, str(p), filename=p.name))
                else:
                    out.append(_media_chip_html(modality, f"{part} [missing]", pending=True))
            else:
                # Only render tags as media chips if they map to generated artifacts.
                out.append(f"<span class='txt-seg'>{_escape(part)}</span>")
            continue
        if part.strip():
            out.append(f"<span class='txt-seg'>{_escape(part)}</span>")
    if not out:
        return "<span class='txt-seg'>(no output)</span>"
    return "".join(out)


def _render_chat(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return "<div class='chat-empty'>Start a new chat. Click insert buttons to add media tokens.</div>"

    chunks = []
    for msg in messages:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        label = "UNIMA" if role == "assistant" else "You"
        content_html = msg.get("html", "") or ""
        chunks.append(
            f"<div class='chat-msg {role}'>"
            f"<div class='bubble'><div class='bubble-meta'>{label}</div>{content_html}</div>"
            "</div>"
        )
    return "".join(chunks)


def _render_draft_preview(text: str, media_state: Dict[str, Dict[str, str]]) -> str:
    if not (text or "").strip() and not media_state:
        return "<div class='chat-empty'>Draft preview appears here.</div>"
    body = _render_input_interleave(text, media_state, with_popup=True)
    return f"<div class='chat-msg user'><div class='bubble'><div class='bubble-meta'>Draft</div>{body}</div></div>"


def _insert_media_token(
    draft_text: str,
    media_state: Dict[str, Dict[str, str]],
    counters: Dict[str, int],
    file_value: Any,
    modality: str,
) -> Tuple[str, Dict[str, Dict[str, str]], Dict[str, int], str, str]:
    path_raw = _extract_file_path(file_value)
    if not path_raw:
        return (
            draft_text,
            dict(media_state or {}),
            dict(counters or _new_counters()),
            _render_draft_preview(draft_text, media_state or {}),
            f"No {modality} file detected.",
        )

    path = str(Path(path_raw).expanduser().resolve())
    media = dict(media_state or {})
    ctr = dict(counters or _new_counters())
    ctr.setdefault(modality, 0)
    ctr[modality] += 1
    token = f"[{PREFIX_BY_MODALITY[modality]}{ctr[modality]}]"

    media[token] = {
        "modality": modality,
        "path": path,
        "name": Path(path).name,
    }

    base = draft_text or ""
    if "[CURSOR]" in base:
        next_text = base.replace("[CURSOR]", token, 1)
    elif base and not base.endswith((" ", "\n")):
        next_text = base + " " + token
    else:
        next_text = base + token

    preview = _render_draft_preview(next_text, media)
    status = f"Inserted {token} for file {Path(path).name}"
    return next_text, media, ctr, preview, status


def insert_image_token(
    draft_text: str,
    media_state: Dict[str, Dict[str, str]],
    counters: Dict[str, int],
    image_value: Any,
):
    text, media, ctr, preview, status = _insert_media_token(
        draft_text,
        media_state,
        counters,
        image_value,
        "image",
    )
    return text, media, ctr, preview, f"<div class='status-badge'>{_escape(status)}</div>", None


def insert_audio_token(
    draft_text: str,
    media_state: Dict[str, Dict[str, str]],
    counters: Dict[str, int],
    audio_value: Any,
):
    text, media, ctr, preview, status = _insert_media_token(
        draft_text,
        media_state,
        counters,
        audio_value,
        "audio",
    )
    return text, media, ctr, preview, f"<div class='status-badge'>{_escape(status)}</div>", None


def insert_video_token(
    draft_text: str,
    media_state: Dict[str, Dict[str, str]],
    counters: Dict[str, int],
    video_value: Any,
):
    text, media, ctr, preview, status = _insert_media_token(
        draft_text,
        media_state,
        counters,
        video_value,
        "video",
    )
    return text, media, ctr, preview, f"<div class='status-badge'>{_escape(status)}</div>", None


def clear_draft():
    return (
        "",
        {},
        _new_counters(),
        _render_draft_preview("", {}),
        "<div class='status-badge'>Draft cleared.</div>",
        None,
        None,
        None,
    )


def new_chat():
    return (
        [],
        _render_chat([]),
        "",
        "",
        {},
        _new_counters(),
        _render_draft_preview("", {}),
        "<div class='status-badge'>Started a new chat.</div>",
        None,
        None,
        None,
    )


def _append_log(base: str, line: str, max_chars: int = 70000) -> str:
    txt = (line or "").strip()
    if not txt:
        return base
    merged = f"{base}\n{txt}" if base else txt
    if len(merged) <= max_chars:
        return merged
    return "...\n" + merged[-max_chars:]


def _build_input_blocks(draft_text: str, media_state: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    parts = INPUT_TOKEN_RE.split(draft_text or "")

    for part in parts:
        if not part:
            continue
        if INPUT_TOKEN_FULL_RE.fullmatch(part):
            info = media_state.get(part)
            if not info:
                raise ValueError(f"Media token {part} has no attached file. Reinsert the file before sending.")
            modality = str(info.get("modality", "")).strip().lower()
            path = str(info.get("path", "")).strip()
            if modality not in {"image", "audio", "video"} or not path:
                raise ValueError(f"Media token {part} has an invalid attachment.")
            if not Path(path).expanduser().is_file():
                raise ValueError(f"Media token {part} points to a missing file. Reinsert the file before sending.")
            blocks.append({"modality": modality, "value": path})
            continue

        text = part.strip()
        if text:
            blocks.append({"modality": "text", "value": text})

    return blocks


def send_message(
    draft_text: str,
    media_state: Dict[str, Dict[str, str]],
    counters: Dict[str, int],
    chat_state: List[Dict[str, str]],
    planner_model: str,
    writer_model: str,
    sora_model: str,
    video_seconds: str,
    video_size: str,
    output_root: str,
) -> Iterator[Tuple[str, Dict[str, Dict[str, str]], Dict[str, int], str, str, List[Dict[str, str]], str, str, None, None, None]]:
    chat = list(chat_state or [])
    media = dict(media_state or {})
    ctr = dict(counters or _new_counters())

    try:
        blocks = _build_input_blocks(draft_text, media)
    except ValueError as exc:
        yield (
            draft_text,
            media,
            ctr,
            _render_draft_preview(draft_text, media),
            f"<div class='status-badge'>{_escape(str(exc))}</div>",
            chat,
            _render_chat(chat),
            "",
            None,
            None,
            None,
        )
        return
    if not blocks:
        yield (
            draft_text,
            media,
            ctr,
            _render_draft_preview(draft_text, media),
            "<div class='status-badge'>Enter text or insert at least one media token.</div>",
            chat,
            _render_chat(chat),
            "",
            None,
            None,
            None,
        )
        return

    user_html = _render_input_interleave(draft_text, media, with_popup=True)
    chat.append({"role": "user", "html": user_html})
    chat.append({"role": "assistant", "html": "<span class='txt-seg'>Generating...</span>"})

    logs = "[app] submitted task"
    status_html = "<div class='status-badge'>Running: streaming generation started.</div>"
    chat_html = _render_chat(chat)

    # Clear draft immediately after submit, like chat apps.
    yield (
        "",
        {},
        _new_counters(),
        _render_draft_preview("", {}),
        status_html,
        chat,
        chat_html,
        logs,
        None,
        None,
        None,
    )

    output_base = Path((output_root or "./runs")).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"run_{dt.datetime.now():%Y%m%d_%H%M%S}_", dir=output_base))
    logs = _append_log(logs, f"[app] run_dir={run_dir}")

    events: queue.Queue[Tuple[str, str]] = queue.Queue()
    holder: Dict[str, Any] = {}

    callbacks = PipelineCallbacks(
        on_log=lambda text: events.put(("log", text)),
        on_qwen_delta=lambda text: events.put(("qwen", text)),
        on_understanding_delta=lambda text: events.put(("understanding", text)),
        on_generation_delta=lambda text: events.put(("generation", text)),
    )

    def _worker() -> None:
        try:
            holder["result"] = run_interleave_session(
                blocks,
                output_dir=str(run_dir),
                planner_model=planner_model.strip() or "gpt-5",
                writer_model=writer_model.strip() or "gpt-5-mini",
                sora_model=sora_model.strip() or "sora-2",
                default_video_seconds=int(video_seconds),
                default_video_size=video_size.strip() or "1280x720",
                callbacks=callbacks,
            )
        except Exception as exc:
            holder["error"] = exc
            holder["error_trace"] = traceback.format_exc()
        finally:
            events.put(("done", ""))

    started = time.time()
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    assistant_stream = ""
    while True:
        try:
            kind, payload = events.get(timeout=0.25)
        except queue.Empty:
            if worker.is_alive():
                continue
            break

        if kind == "log":
            logs = _append_log(logs, payload)
        elif kind == "qwen":
            logs = _append_log(logs, f"[qwen] {payload}")
        elif kind == "understanding":
            logs = _append_log(logs, f"[understanding] {payload}")
        elif kind == "generation":
            logs = _append_log(logs, f"[generation] {payload}")
            assistant_stream += payload
            chat[-1]["html"] = _render_output_interleave(assistant_stream, artifact_map=None)
            chat_html = _render_chat(chat)

        if kind == "done":
            break

        yield (
            "",
            {},
            _new_counters(),
            _render_draft_preview("", {}),
            status_html,
            chat,
            chat_html,
            logs,
            None,
            None,
            None,
        )

    worker.join(timeout=1.0)

    if "error" in holder:
        err = holder["error"]
        status_html = f"<div class='status-badge'>Run failed: {_escape(type(err).__name__)}: {_escape(repr(err))}</div>"
        logs = _append_log(logs, f"[app] failed: {type(err).__name__}: {err!r}")
        trace = str(holder.get("error_trace", "") or "").strip()
        if trace:
            logs = _append_log(logs, "[app] traceback (last 24 lines):")
            for line in trace.splitlines()[-24:]:
                logs = _append_log(logs, f"[trace] {line}")
        chat[-1]["html"] = f"<span class='txt-seg'>Run failed: {_escape(type(err).__name__)}: {_escape(str(err))}</span>"
        chat_html = _render_chat(chat)
        yield (
            "",
            {},
            _new_counters(),
            _render_draft_preview("", {}),
            status_html,
            chat,
            chat_html,
            logs,
            None,
            None,
            None,
        )
        return

    result = holder.get("result")
    if result is None:
        status_html = "<div class='status-badge'>Run failed: internal error, missing result.</div>"
        logs = _append_log(logs, "[app] failed: result missing")
        chat[-1]["html"] = "<span class='txt-seg'>Run failed: internal error.</span>"
        chat_html = _render_chat(chat)
        yield (
            "",
            {},
            _new_counters(),
            _render_draft_preview("", {}),
            status_html,
            chat,
            chat_html,
            logs,
            None,
            None,
            None,
        )
        return

    artifact_map = {
        item.tag: {
            "modality": item.modality,
            "path": item.path,
        }
        for item in result.output_artifacts
    }
    chat[-1]["html"] = _render_output_interleave(result.final_text or "", artifact_map=artifact_map)
    chat_html = _render_chat(chat)

    elapsed = time.time() - started
    img_n = sum(1 for a in result.output_artifacts if a.modality == "image")
    aud_n = sum(1 for a in result.output_artifacts if a.modality == "audio")
    vid_n = sum(1 for a in result.output_artifacts if a.modality == "video")
    logs = _append_log(logs, f"[app] finished in {elapsed:.1f}s")
    logs = _append_log(logs, f"[app] artifacts image={img_n}, audio={aud_n}, video={vid_n}")

    status_html = (
        "<div class='status-badge'>"
        f"Completed in {elapsed:.1f}s; output dir: {_escape(str(run_dir))}; "
        f"image={img_n}, audio={aud_n}, video={vid_n}"
        "</div>"
    )

    yield (
        "",
        {},
        _new_counters(),
        _render_draft_preview("", {}),
        status_html,
        chat,
        chat_html,
        logs,
        None,
        None,
        None,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="UniMA Chat", js=JS_FORCE_ENGLISH_UPLOAD_HINTS) as demo:
        chat_state = gr.State([])
        media_state = gr.State({})
        counters_state = gr.State(_new_counters())

        gr.Markdown(
            """
# UniMA Chat Studio
"""
        )

        with gr.Row(equal_height=True):
            status_md = gr.HTML("<div class='status-badge'>Ready.</div>")
            new_chat_btn = gr.Button("New Chat", variant="secondary")

        with gr.Row():
            with gr.Column(scale=8, elem_classes=["panel-card"]):
                chat_html = gr.HTML(_render_chat([]), elem_id="chat_view", elem_classes=["chat-shell"])
                with gr.Accordion("Unified Runtime Log", open=False):
                    logs_box = gr.Textbox(
                        label="Logs",
                        lines=12,
                        interactive=False,
                        autoscroll=True,
                    )

                draft_text = gr.Textbox(
                    label="Message",
                    lines=5,
                    placeholder="Type your message. Click insert buttons to add media tokens at cursor position.",
                    elem_id="draft_textarea",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Image", type="filepath")
                        insert_image_btn = gr.Button("Insert Image Token", variant="secondary")
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="Audio", type="filepath")
                        insert_audio_btn = gr.Button("Insert Audio Token", variant="secondary")
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Video")
                        insert_video_btn = gr.Button("Insert Video Token", variant="secondary")

                draft_preview = gr.HTML(_render_draft_preview("", {}))

                with gr.Row():
                    clear_draft_btn = gr.Button("Clear Draft", variant="secondary")
                    send_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=3, elem_classes=["panel-card"]):
                with gr.Accordion("Advanced Settings", open=False):
                    planner_model = gr.Textbox(label="Understanding Model", value="gpt-5")
                    writer_model = gr.Textbox(label="Generation Model", value="gpt-5-mini")
                    sora_model = gr.Textbox(
                        label="Video Model (Videos API)",
                        value=os.getenv("SORA_MODEL", "sora-2"),
                    )
                    video_seconds = gr.Dropdown(
                        label="Default Video Length (seconds)",
                        choices=["4", "8", "12"],
                        value="8",
                    )
                    video_size = gr.Dropdown(
                        label="Default Video Resolution",
                        choices=["720x1280", "1280x720", "1024x1792", "1792x1024"],
                        value="1280x720",
                    )
                    output_root = gr.Textbox(label="Output Root Directory", value="./runs")

        insert_image_btn.click(
            insert_image_token,
            inputs=[draft_text, media_state, counters_state, image_input],
            outputs=[draft_text, media_state, counters_state, draft_preview, status_md, image_input],
            js=JS_INSERT_CURSOR,
        )

        insert_audio_btn.click(
            insert_audio_token,
            inputs=[draft_text, media_state, counters_state, audio_input],
            outputs=[draft_text, media_state, counters_state, draft_preview, status_md, audio_input],
            js=JS_INSERT_CURSOR,
        )

        insert_video_btn.click(
            insert_video_token,
            inputs=[draft_text, media_state, counters_state, video_input],
            outputs=[draft_text, media_state, counters_state, draft_preview, status_md, video_input],
            js=JS_INSERT_CURSOR,
        )

        clear_draft_btn.click(
            clear_draft,
            outputs=[
                draft_text,
                media_state,
                counters_state,
                draft_preview,
                status_md,
                image_input,
                audio_input,
                video_input,
            ],
        )

        new_chat_btn.click(
            new_chat,
            outputs=[
                chat_state,
                chat_html,
                logs_box,
                draft_text,
                media_state,
                counters_state,
                draft_preview,
                status_md,
                image_input,
                audio_input,
                video_input,
            ],
        )

        send_btn.click(
            send_message,
            inputs=[
                draft_text,
                media_state,
                counters_state,
                chat_state,
                planner_model,
                writer_model,
                sora_model,
                video_seconds,
                video_size,
                output_root,
            ],
            outputs=[
                draft_text,
                media_state,
                counters_state,
                draft_preview,
                status_md,
                chat_state,
                chat_html,
                logs_box,
                image_input,
                audio_input,
                video_input,
            ],
        )

    return demo


if __name__ == "__main__":
    if os.getenv("QWEN_OMNI_WARMUP", "0").lower() in {"1", "true", "yes"}:
        print("[startup] loading local Qwen2.5-Omni model...")
        warmup_local_av_model()
        print("[startup] local Qwen2.5-Omni model loaded.")
    app = build_demo()
    project_root = Path(__file__).resolve().parent
    scratch_root = Path("/scratch/e1539128")
    allowed = [str(project_root), "/tmp"]
    if scratch_root.exists():
        allowed.append(str(scratch_root))
    app.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
        allowed_paths=allowed,
    )
