from __future__ import annotations

import asyncio
import base64
import copy
import concurrent.futures
import mimetypes
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from agents import Agent, Runner, function_tool
from openai import OpenAI
from qwen_omni import analyze_with_qwen_omni, warmup_qwen_omni

try:
    from openai.types.responses import ResponseTextDeltaEvent
except Exception:  # pragma: no cover - compatibility across OpenAI SDK versions
    ResponseTextDeltaEvent = None

InputModality = Literal["text", "image", "audio", "video"]
OutputModality = Literal["image", "audio", "video"]

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
OUTPUT_TAG_RE = re.compile(r"(<(?:image|audio|video)\d+>)")
OUTPUT_TAG_ENTITY_RE = re.compile(r"<(image|audio|video)\d+>")
VIDEO_ALLOWED_SECONDS = (4, 8, 12)
VIDEO_ALLOWED_SIZES = ("720x1280", "1280x720", "1024x1792", "1792x1024")
REQUIRED_OUTPUTS_RE = re.compile(r"^\s*-?\s*RequiredOutputs\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
REQ_COUNT_RE = re.compile(r"\b(image|audio|video)\s*=\s*(\d+)\b", re.IGNORECASE)
INPUT_MARKER_RE = re.compile(r"\[IN_(?:IMAGE|AUDIO|VIDEO)_\d+\]")
BLOCK_PREFIX_RE = re.compile(r"^\s*Block\s+\d+\s*:\s*", re.IGNORECASE)
INTERNAL_NOTE_PREFIX_RE = re.compile(
    r"^\s*(?:reference video note|audio note|how to use these panels|key elements to depict)\s*:",
    re.IGNORECASE,
)
INTERNAL_MARKER_RE = re.compile(r"\[(?:BLOCK\s+\d+\|[^\]]+|QWEN2\.5_OMNI_AV_REPORT)\]", re.IGNORECASE)
EMOJI_TAG_RE = re.compile(r"[🖼🎵🎬]\s*(<(?:image|audio|video)\d+>)")
TAG_FILENAME_SUFFIX_RE = re.compile(
    r"(<(?:image|audio|video)\d+>)\s*[·•-]\s*[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)


def _progress_bar(pct: float, length: int = 28) -> str:
    pct = max(0.0, min(100.0, float(pct or 0.0)))
    filled = int(round((pct / 100.0) * length))
    return "=" * filled + "-" * (length - filled)


def _preview_text(text: str, max_len: int = 1200) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[:max_len] + "\n...(truncated)..."


@dataclass
class InputBlock:
    modality: InputModality
    value: str
    tag: Optional[str] = None


@dataclass
class Artifact:
    tag: str
    modality: OutputModality
    path: str
    prompt: str


@dataclass
class AgentRunResult:
    final_text: str
    understanding_brief: str
    input_blocks: List[InputBlock]
    output_artifacts: List[Artifact]
    output_sequence: List[Dict[str, str]]


@dataclass
class PipelineCallbacks:
    on_log: Optional[Callable[[str], None]] = None
    on_qwen_delta: Optional[Callable[[str], None]] = None
    on_understanding_delta: Optional[Callable[[str], None]] = None
    on_generation_delta: Optional[Callable[[str], None]] = None


@dataclass
class ArtifactRegistry:
    output_dir: Path
    _counters: Dict[OutputModality, int] = field(
        default_factory=lambda: {"image": 0, "audio": 0, "video": 0}
    )
    artifacts: List[Artifact] = field(default_factory=list)

    def allocate(self, modality: OutputModality, extension: str) -> Tuple[str, Path]:
        self._counters[modality] += 1
        idx = self._counters[modality]
        tag = f"<{modality}{idx}>"
        ext = extension if extension.startswith(".") else f".{extension}"
        path = self.output_dir / f"{modality}_{idx:02d}{ext.lower()}"
        return tag, path

    def record(self, tag: str, modality: OutputModality, path: Path, prompt: str) -> None:
        self.artifacts.append(
            Artifact(tag=tag, modality=modality, path=str(path.resolve()), prompt=prompt.strip())
        )

    @property
    def by_tag(self) -> Dict[str, Artifact]:
        return {item.tag: item for item in self.artifacts}


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def _guess_mime(path: Path) -> str:
    guessed = mimetypes.guess_type(str(path))[0]
    return guessed or "application/octet-stream"


def _file_to_data_url(path: Path) -> str:
    mime = _guess_mime(path)
    blob = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{blob}"


def _extract_b64_image(response: object) -> Optional[str]:
    data = getattr(response, "data", None)
    if not data and isinstance(response, dict):
        data = response.get("data")
    if not data or not isinstance(data, list):
        return None

    first = data[0]
    if isinstance(first, dict):
        raw = first.get("b64_json") or first.get("base64")
        return raw if isinstance(raw, str) else None

    raw = getattr(first, "b64_json", None) or getattr(first, "base64", None)
    return raw if isinstance(raw, str) else None


def _ensure_existing_file(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _normalize_video_seconds(seconds: int) -> str:
    target = int(seconds)
    nearest = min(VIDEO_ALLOWED_SECONDS, key=lambda s: abs(s - target))
    return str(nearest)


def _normalize_video_size(size: str, default_size: str) -> str:
    val = (size or "").strip()
    if val in VIDEO_ALLOWED_SIZES:
        return val
    if default_size in VIDEO_ALLOWED_SIZES:
        return default_size
    return "1280x720"


def normalize_raw_blocks(raw_blocks: Sequence[Dict[str, str]]) -> List[InputBlock]:
    out: List[InputBlock] = []
    for block in raw_blocks:
        if not isinstance(block, dict):
            continue
        modality = str(block.get("modality", "")).strip().lower()
        value = block.get("value")
        if modality not in {"text", "image", "audio", "video"}:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        out.append(InputBlock(modality=modality, value=value.strip()))
    return out


class InterleaveAgentPipeline:
    def __init__(
        self,
        *,
        output_dir: str,
        planner_model: str = "gpt-5",
        writer_model: str = "gpt-5-mini",
        image_model: str = "gpt-image-1.5",
        tts_model: str = "gpt-4o-mini-tts",
        sora_model: str = "sora-2",
        default_video_seconds: int = 8,
        default_video_size: str = "1280x720",
        video_timeout_seconds: int = 900,
        video_poll_seconds: float = 2.0,
        client: Optional[OpenAI] = None,
        callbacks: Optional[PipelineCallbacks] = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.planner_model = planner_model
        self.writer_model = writer_model
        self.image_model = image_model
        self.tts_model = tts_model
        self.sora_model = sora_model
        self.default_video_seconds = int(default_video_seconds)
        self.default_video_size = default_video_size
        self.video_timeout_seconds = int(video_timeout_seconds)
        self.video_poll_seconds = float(video_poll_seconds)

        self.client = client or OpenAI()
        self.callbacks = callbacks or PipelineCallbacks()
        self.understanding_prompt = _read_prompt(PROMPTS_DIR / "understanding_agent.txt")
        self.generation_prompt = _read_prompt(PROMPTS_DIR / "generation_agent.txt")
        self.qwen_av_prompt = _read_prompt(PROMPTS_DIR / "qwen_omni_av.txt")
        self._tool_success_counts: Dict[str, int] = {"image": 0, "audio": 0, "video": 0}

    @staticmethod
    def _safe_callback(callback: Optional[Callable[[str], None]], payload: str) -> None:
        if not callback:
            return
        try:
            callback(payload)
        except Exception:
            pass

    def _emit_log(self, message: str) -> None:
        print(message, flush=True)
        self._safe_callback(self.callbacks.on_log, message)

    def _forward_external_log(self, message: str) -> None:
        # Nested modules may already print; only forward to UI callbacks to avoid duplicates.
        self._safe_callback(self.callbacks.on_log, message)

    def _emit_qwen_delta(self, text: str) -> None:
        self._safe_callback(self.callbacks.on_qwen_delta, text)

    def _emit_understanding_delta(self, text: str) -> None:
        self._safe_callback(self.callbacks.on_understanding_delta, text)

    def _emit_generation_delta(self, text: str) -> None:
        self._safe_callback(self.callbacks.on_generation_delta, text)

    @staticmethod
    def _extract_text_delta(raw_event_data: Any) -> str:
        if raw_event_data is None:
            return ""

        if ResponseTextDeltaEvent is not None and isinstance(raw_event_data, ResponseTextDeltaEvent):
            delta = getattr(raw_event_data, "delta", "")
            return delta if isinstance(delta, str) else ""

        if getattr(raw_event_data, "type", "") == "response.output_text.delta":
            delta = getattr(raw_event_data, "delta", "")
            return delta if isinstance(delta, str) else ""

        return ""

    @staticmethod
    def _run_async(coro_factory: Callable[[], Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())

        # asyncio.run cannot start another loop in the current thread.
        result: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def run_in_thread() -> None:
            try:
                result.set_result(asyncio.run(coro_factory()))
            except BaseException as exc:
                result.set_exception(exc)

        threading.Thread(target=run_in_thread, daemon=True).start()
        return result.result()

    @staticmethod
    def _extract_required_outputs(understanding_brief: str) -> Dict[str, int]:
        required = {"image": 0, "audio": 0, "video": 0}
        match = REQUIRED_OUTPUTS_RE.search(understanding_brief or "")
        if match:
            for key, count in REQ_COUNT_RE.findall(match.group(1)):
                required[key.lower()] = int(count)
        return required

    @staticmethod
    def _count_artifacts(artifacts: Sequence[Artifact]) -> Dict[str, int]:
        return {
            "image": sum(1 for a in artifacts if a.modality == "image"),
            "audio": sum(1 for a in artifacts if a.modality == "audio"),
            "video": sum(1 for a in artifacts if a.modality == "video"),
        }

    @staticmethod
    def _compute_missing(required: Dict[str, int], got: Dict[str, int]) -> Dict[str, int]:
        return {k: max(0, int(required.get(k, 0)) - int(got.get(k, 0))) for k in ("image", "audio", "video")}

    @staticmethod
    def _format_modality_counts(counts: Dict[str, int]) -> str:
        return f"image={counts.get('image', 0)}, audio={counts.get('audio', 0)}, video={counts.get('video', 0)}"

    def _stream_agent_output(
        self,
        *,
        agent: Agent,
        input_payload: Any,
        stage_name: str,
        model_name: str,
        on_delta: Optional[Callable[[str], None]],
    ) -> str:
        self._emit_log(f"[pipeline] stage: {stage_name} ({model_name})")
        started = time.time()
        stream_activity = False

        async def _consume() -> str:
            nonlocal stream_activity
            result = Runner.run_streamed(agent, input=input_payload)
            async for event in result.stream_events():
                event_type = getattr(event, "type", "")
                if event_type == "raw_response_event":
                    stream_activity = True
                    delta = self._extract_text_delta(getattr(event, "data", None))
                    if delta:
                        self._safe_callback(on_delta, delta)
                    continue

                if event_type == "run_item_stream_event":
                    stream_activity = True
                    name = str(getattr(event, "name", "") or "unknown")
                    item = getattr(event, "item", None)
                    item_type = str(getattr(item, "type", "unknown"))
                    self._emit_log(f"[{stage_name}] event={name} item={item_type}")
                    continue

                if event_type == "agent_updated_stream_event":
                    new_agent = getattr(event, "new_agent", None)
                    if new_agent is not None:
                        new_name = getattr(new_agent, "name", "")
                        if new_name:
                            self._emit_log(f"[{stage_name}] active-agent={new_name}")
                    continue

            return str(result.final_output or "").strip()

        try:
            final_output = str(self._run_async(_consume) or "").strip()
        except Exception as exc:
            self._emit_log(
                f"[pipeline] stage error: {stage_name} streaming failed: {type(exc).__name__}: {exc!r}"
            )
            cause = exc.__cause__ or exc.__context__
            if cause is not None:
                self._emit_log(
                    f"[pipeline] stage error cause: {type(cause).__name__}: {cause!r}"
                )

            tb_lines = traceback.format_exc().rstrip().splitlines()
            if tb_lines:
                self._emit_log("[pipeline] traceback (last 12 lines):")
                for line in tb_lines[-12:]:
                    self._emit_log(f"[trace] {line}")

            if stream_activity:
                # A retry could generate media twice and charge for both runs.
                raise RuntimeError(f"{stage_name} failed after streaming began; refusing to repeat tool calls") from exc

            self._emit_log(f"[pipeline] fallback: retry {stage_name} with non-streaming run_sync")
            try:
                sync_result = Runner.run_sync(agent, input=input_payload)
                final_output = str(sync_result.final_output or "").strip()
                if final_output:
                    self._safe_callback(on_delta, final_output)
                self._emit_log(f"[pipeline] fallback done: {stage_name} (run_sync)")
            except Exception as sync_exc:
                self._emit_log(
                    f"[pipeline] fallback failed: {stage_name}: {type(sync_exc).__name__}: {sync_exc!r}"
                )
                raise

        self._emit_log(f"[pipeline] stage done: {stage_name} ({time.time() - started:.1f}s)")
        return final_output

    def run(self, blocks: Sequence[InputBlock]) -> AgentRunResult:
        if not blocks:
            raise ValueError("At least one input block is required.")
        self._tool_success_counts = {"image": 0, "audio": 0, "video": 0}

        self._emit_log("[pipeline] stage: build interleave input")
        stage_start = time.time()
        normalized_blocks, input_content = self._build_interleave_input(blocks)
        self._emit_log(
            f"[pipeline] stage done: build interleave input ({time.time() - stage_start:.1f}s, blocks={len(normalized_blocks)})",
        )
        understanding_agent = Agent(
            name="understanding-agent",
            instructions=self.understanding_prompt,
            model=self.planner_model,
        )

        understanding_brief = self._stream_agent_output(
            agent=understanding_agent,
            input_payload=[{"role": "user", "content": input_content}],
            stage_name="understanding agent",
            model_name=self.planner_model,
            on_delta=self._emit_understanding_delta,
        )
        if not understanding_brief:
            understanding_brief = "GENERATION_BRIEF\n- Goal: answer the user request clearly."
        self._emit_log("[pipeline] understanding brief begin")
        self._emit_log(_preview_text(understanding_brief))
        self._emit_log("[pipeline] understanding brief end")

        registry = ArtifactRegistry(output_dir=self.output_dir)
        generation_agent = Agent(
            name="generation-agent",
            instructions=self.generation_prompt,
            model=self.writer_model,
            tools=self._build_generation_tools(registry),
        )

        generation_input = [
            {"role": "user", "content": copy.deepcopy(input_content)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Follow this concise execution brief from the understanding agent:\n"
                        + understanding_brief,
                    }
                ],
            },
        ]

        final_text = self._stream_agent_output(
            agent=generation_agent,
            input_payload=generation_input,
            stage_name="generation agent",
            model_name=self.writer_model,
            on_delta=self._emit_generation_delta,
        )

        required_outputs = self._extract_required_outputs(understanding_brief)
        got_outputs = self._count_artifacts(registry.artifacts)
        missing_outputs = self._compute_missing(required_outputs, got_outputs)
        self._emit_log(f"[pipeline] required outputs: {self._format_modality_counts(required_outputs)}")
        self._emit_log(f"[pipeline] tool success counts: {self._format_modality_counts(self._tool_success_counts)}")
        self._emit_log(f"[pipeline] produced artifacts: {self._format_modality_counts(got_outputs)}")

        if any(v > 0 for v in missing_outputs.values()):
            self._emit_log(f"[pipeline] missing required outputs: {self._format_modality_counts(missing_outputs)}")
            recovery_prompt = (
                "Your previous draft did not satisfy required modality counts.\n"
                f"Missing outputs now: image={missing_outputs['image']}, "
                f"audio={missing_outputs['audio']}, video={missing_outputs['video']}.\n"
                "You MUST call tools to generate the missing modalities and then output a corrected final answer.\n"
                "Already generated tags: "
                + (", ".join(item.tag for item in registry.artifacts) or "none")
                + ". Reuse these tags for existing media, and use new tool-returned tags for missing media.\n"
                "Do not regenerate existing media. Use each tag once.\n\n"
                "Previous draft:\n"
                + (final_text or "(empty)")
            )
            recovery_input = generation_input + [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": recovery_prompt}],
                }
            ]
            final_text = self._stream_agent_output(
                agent=generation_agent,
                input_payload=recovery_input,
                stage_name="generation agent recovery",
                model_name=self.writer_model,
                on_delta=None,
            )
            got_outputs = self._count_artifacts(registry.artifacts)
            missing_outputs = self._compute_missing(required_outputs, got_outputs)
            self._emit_log(
                f"[pipeline] after recovery artifacts: {self._format_modality_counts(got_outputs)}"
            )
            if any(v > 0 for v in missing_outputs.values()):
                self._emit_log(
                    f"[pipeline] warning: still missing outputs after recovery: {self._format_modality_counts(missing_outputs)}"
                )

        self._emit_log("[pipeline] final output begin")
        self._emit_log(_preview_text(final_text))
        self._emit_log("[pipeline] final output end")

        cleaned_text = self._cleanup_final_text(final_text or "")
        normalized_text, normalized_artifacts, dropped_tags = self._normalize_output_tags(
            cleaned_text,
            registry.artifacts,
        )
        if cleaned_text != (final_text or ""):
            self._emit_log("[pipeline] postprocess: removed internal markers/analysis boilerplate")
        if normalized_text != cleaned_text:
            self._emit_log("[pipeline] postprocess: reconciled output tags with generated artifacts")
        if dropped_tags:
            self._emit_log(
                "[pipeline] warning: output contained tags without matching artifacts: "
                + ", ".join(dropped_tags)
            )

        artifact_map = {item.tag: item for item in normalized_artifacts}
        output_sequence = self._parse_output_sequence(normalized_text, artifact_map)
        self._emit_log(
            f"[pipeline] artifacts: image={sum(1 for a in normalized_artifacts if a.modality == 'image')}, "
            f"audio={sum(1 for a in normalized_artifacts if a.modality == 'audio')}, "
            f"video={sum(1 for a in normalized_artifacts if a.modality == 'video')}",
        )
        return AgentRunResult(
            final_text=normalized_text,
            understanding_brief=understanding_brief,
            input_blocks=normalized_blocks,
            output_artifacts=normalized_artifacts,
            output_sequence=output_sequence,
        )

    def _build_interleave_input(
        self, blocks: Sequence[InputBlock]
    ) -> Tuple[List[InputBlock], List[Dict[str, str]]]:
        content: List[Dict[str, str]] = []
        normalized: List[InputBlock] = []
        counters = {"image": 0, "audio": 0, "video": 0}
        text_fragments: List[str] = []
        input_sequence_lines: List[str] = []
        ordered_av_inputs: List[Tuple[str, str, str]] = []

        for idx, block in enumerate(blocks, start=1):
            modality = block.modality
            value = block.value.strip()
            if not value:
                continue

            if modality == "text":
                normalized.append(InputBlock(modality="text", value=value, tag=None))
                text_fragments.append(value)
                input_sequence_lines.append(f"#{idx}: text -> {value}")
                content.append(
                    {
                        "type": "input_text",
                        "text": f"[BLOCK {idx}|text]\n{value}",
                    }
                )
                continue

            counters[modality] += 1
            tag = f"[IN_{modality.upper()}_{counters[modality]}]"

            if modality == "image":
                image_path = _ensure_existing_file(value)
                normalized.append(InputBlock(modality="image", value=str(image_path), tag=tag))
                input_sequence_lines.append(f"#{idx}: image({tag}) -> {image_path.name}")
                content.append(
                    {"type": "input_text", "text": f"[BLOCK {idx}|image] {tag} (input image)"}
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": _file_to_data_url(image_path),
                    }
                )
                continue

            if modality == "audio":
                audio_path = _ensure_existing_file(value)
                normalized.append(InputBlock(modality="audio", value=str(audio_path), tag=tag))
                ordered_av_inputs.append(("audio", str(audio_path), tag))
                input_sequence_lines.append(f"#{idx}: audio({tag}) -> {audio_path.name}")
                content.append(
                    {
                        "type": "input_text",
                        "text": f"[BLOCK {idx}|audio] {tag} (input audio)",
                    }
                )
                continue

            if modality == "video":
                video_path = _ensure_existing_file(value)
                normalized.append(InputBlock(modality="video", value=str(video_path), tag=tag))
                ordered_av_inputs.append(("video", str(video_path), tag))
                input_sequence_lines.append(f"#{idx}: video({tag}) -> {video_path.name}")
                content.append(
                    {
                        "type": "input_text",
                        "text": f"[BLOCK {idx}|video] {tag} (input video)",
                    }
                )
                continue

        if ordered_av_inputs:
            user_request = "\n".join(text_fragments).strip() or "Please analyze the provided media."
            input_sequence = "\n".join(input_sequence_lines).strip() or "(empty)"
            av_instruction = (
                self.qwen_av_prompt.replace("<<USER_REQUEST>>", user_request)
                .replace("<<INPUT_SEQUENCE>>", input_sequence)
                .strip()
            )
            self._emit_log("[pipeline] local Qwen2.5-Omni AV analysis started")
            av_start = time.time()
            av_report = analyze_with_qwen_omni(
                instruction=av_instruction,
                ordered_media=ordered_av_inputs,
                on_log=self._forward_external_log,
                on_delta=self._emit_qwen_delta,
            )
            self._emit_log(
                f"[pipeline] local Qwen2.5-Omni AV analysis finished in {time.time() - av_start:.1f}s",
            )
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "[QWEN2.5_OMNI_AV_REPORT]\n"
                        + (av_report.strip() or "(empty report)")
                    ),
                }
            )

        content.append(
            {
                "type": "input_text",
                "text": "The block order above is authoritative. Keep this order when producing interleaved output.",
            }
        )
        return normalized, content

    def _build_generation_tools(self, registry: ArtifactRegistry):
        client = self.client

        @function_tool
        def generate_image(prompt: str, size: str = "1024x1024") -> str:
            """Generate one image and return its placeholder tag, e.g. <image1>."""
            prompt = prompt.strip()
            if not prompt:
                raise ValueError("Image prompt must not be empty.")

            self._emit_log(f"[tool:image] start size={size} prompt={_preview_text(prompt, 180)}")
            response = client.images.generate(
                model=self.image_model,
                prompt=prompt,
                size=size,
            )
            b64_payload = _extract_b64_image(response)
            if not b64_payload:
                raise RuntimeError("Image generation returned no b64 payload.")

            tag, path = registry.allocate("image", ".png")
            path.write_bytes(base64.b64decode(b64_payload))
            registry.record(tag, "image", path, prompt)
            self._tool_success_counts["image"] += 1
            self._emit_log(f"[tool:image] done tag={tag} path={path}")
            return tag

        @function_tool
        def generate_audio(text: str, voice: str = "alloy") -> str:
            """Generate one speech audio file and return its placeholder tag, e.g. <audio1>."""
            text = text.strip()
            if not text:
                raise ValueError("Audio text must not be empty.")

            self._emit_log(f"[tool:audio] start voice={voice} text={_preview_text(text, 180)}")
            stream_ctx = client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=voice,
                input=text,
                response_format="mp3",
            )

            audio_bytes = bytearray()
            with stream_ctx as stream:
                if hasattr(stream, "iter_bytes"):
                    for chunk in stream.iter_bytes(chunk_size=32 * 1024):
                        audio_bytes.extend(chunk)
                elif hasattr(stream, "read"):
                    audio_bytes.extend(stream.read())
                else:
                    raise RuntimeError("Unexpected TTS streaming response object.")

            tag, path = registry.allocate("audio", ".mp3")
            path.write_bytes(bytes(audio_bytes))
            registry.record(tag, "audio", path, text)
            self._tool_success_counts["audio"] += 1
            self._emit_log(f"[tool:audio] done tag={tag} path={path}")
            return tag

        @function_tool
        def generate_video(
            prompt: str,
            seconds: Optional[int] = None,
            size: Optional[str] = None,
            model: Optional[str] = None,
        ) -> str:
            """Generate one video with the OpenAI Videos API and return its placeholder tag."""
            prompt = prompt.strip()
            if not prompt:
                raise ValueError("Video prompt must not be empty.")

            use_model = (model or self.sora_model).strip()
            raw_seconds = int(seconds) if seconds is not None else self.default_video_seconds
            use_seconds = _normalize_video_seconds(raw_seconds)
            use_size = _normalize_video_size(size or self.default_video_size, self.default_video_size)

            self._emit_log(
                f"[tool:video] start model={use_model} seconds={use_seconds} size={use_size} prompt={_preview_text(prompt, 180)}",
            )
            video = client.videos.create(
                model=use_model,
                prompt=prompt,
                seconds=use_seconds,
                size=use_size,
            )
            video_id = getattr(video, "id", None)
            if not video_id:
                raise RuntimeError("Video API did not return an id.")

            status = str(getattr(video, "status", "queued")).lower()
            progress = float(getattr(video, "progress", 0.0) or 0.0)
            self._emit_log(f"[tool:video] created id={video_id} status={status} progress={progress:.1f}%")
            deadline = time.time() + self.video_timeout_seconds
            while status in {"queued", "in_progress"}:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Video generation timed out after {self.video_timeout_seconds}s (id={video_id})."
                    )
                time.sleep(self.video_poll_seconds)
                video = client.videos.retrieve(video_id)
                status = str(getattr(video, "status", "")).lower()
                progress = float(getattr(video, "progress", 0.0) or 0.0)
                self._emit_log(
                    f"[tool:video] id={video_id} {status:<11} [{_progress_bar(progress)}] {progress:5.1f}%",
                )

            if status != "completed":
                err_obj = getattr(video, "error", None)
                err_msg = getattr(err_obj, "message", None) or f"status={status}"
                raise RuntimeError(f"Video generation failed: {err_msg}")

            tag, path = registry.allocate("video", ".mp4")
            self._emit_log(f"[tool:video] download start id={video_id}")
            content = client.videos.download_content(video_id, variant="video")
            content.write_to_file(str(path))
            registry.record(tag, "video", path, prompt)
            self._tool_success_counts["video"] += 1
            self._emit_log(f"[tool:video] done tag={tag} path={path}")
            return tag

        return [generate_image, generate_audio, generate_video]

    @staticmethod
    def _cleanup_final_text(text: str) -> str:
        if not text:
            return ""

        cleaned_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            line = INTERNAL_MARKER_RE.sub("", line)
            line = INPUT_MARKER_RE.sub("", line)
            line = EMOJI_TAG_RE.sub(r"\1", line)
            line = TAG_FILENAME_SUFFIX_RE.sub(r"\1", line)
            line = BLOCK_PREFIX_RE.sub("", line).strip()
            if not line:
                cleaned_lines.append("")
                continue
            if INTERNAL_NOTE_PREFIX_RE.match(line):
                continue
            cleaned_lines.append(line)

        merged = "\n".join(cleaned_lines)
        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
        return merged

    @staticmethod
    def _normalize_output_tags(
        text: str,
        artifacts: Sequence[Artifact],
    ) -> Tuple[str, List[Artifact], List[str]]:
        artifacts_by_tag = {a.tag: a for a in artifacts}
        used_tags: set[str] = set()
        dropped_tags: List[str] = []

        def replace_tag(match: re.Match[str]) -> str:
            tag = match.group(0)
            if tag not in artifacts_by_tag or tag in used_tags:
                dropped_tags.append(tag)
                return ""
            used_tags.add(tag)
            return tag

        normalized_text = OUTPUT_TAG_ENTITY_RE.sub(replace_tag, text)
        # Keep every successful tool result visible even when the writer omits its tag.
        omitted = [item.tag for item in artifacts if item.tag not in used_tags]
        if omitted:
            normalized_text = "\n\n".join(part for part in (normalized_text.strip(), " ".join(omitted)) if part)
        normalized_text = re.sub(r"[ \t]+\n", "\n", normalized_text)
        normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()
        return normalized_text, list(artifacts), dropped_tags

    def _parse_output_sequence(
        self,
        final_text: str,
        artifacts_by_tag: Dict[str, Artifact],
    ) -> List[Dict[str, str]]:
        if not final_text:
            return []

        sequence: List[Dict[str, str]] = []
        for part in OUTPUT_TAG_RE.split(final_text):
            if not part:
                continue

            if OUTPUT_TAG_RE.fullmatch(part):
                artifact = artifacts_by_tag.get(part)
                if artifact:
                    sequence.append(
                        {
                            "modality": artifact.modality,
                            "tag": artifact.tag,
                            "value": artifact.path,
                        }
                    )
                else:
                    sequence.append({"modality": "text", "tag": "", "value": part})
                continue

            if part.strip():
                sequence.append({"modality": "text", "tag": "", "value": part.strip()})

        return sequence


def run_interleave_session(
    raw_blocks: Sequence[Dict[str, str]],
    *,
    output_dir: str,
    planner_model: str = "gpt-5",
    writer_model: str = "gpt-5-mini",
    sora_model: str = "sora-2",
    default_video_seconds: int = 8,
    default_video_size: str = "1280x720",
    callbacks: Optional[PipelineCallbacks] = None,
) -> AgentRunResult:
    blocks = normalize_raw_blocks(raw_blocks)
    pipeline = InterleaveAgentPipeline(
        output_dir=output_dir,
        planner_model=planner_model,
        writer_model=writer_model,
        sora_model=sora_model,
        default_video_seconds=default_video_seconds,
        default_video_size=default_video_size,
        callbacks=callbacks,
    )
    return pipeline.run(blocks)


def warmup_local_av_model() -> None:
    warmup_qwen_omni()
