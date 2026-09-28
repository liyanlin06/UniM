from __future__ import annotations

import os
import shutil
import subprocess
import threading
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("PyTorch is required for local Qwen2.5-Omni inference.") from exc


QWEN_OMNI_MODEL_ID = os.getenv("QWEN_OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
QWEN_OMNI_CACHE_DIR = Path(
    os.getenv("QWEN_OMNI_CACHE_DIR")
    or os.getenv("HF_HOME")
    or str(Path.home() / ".cache" / "huggingface")
).expanduser().resolve()
QWEN_OMNI_MAX_NEW_TOKENS = int(os.getenv("QWEN_OMNI_MAX_NEW_TOKENS", "512"))
QWEN_OMNI_VIDEO_MAX_SECONDS = int(os.getenv("QWEN_OMNI_VIDEO_MAX_SECONDS", "12"))
QWEN_OMNI_VIDEO_FPS = int(os.getenv("QWEN_OMNI_VIDEO_FPS", "12"))
QWEN_OMNI_USE_AUDIO_IN_VIDEO = os.getenv("QWEN_OMNI_USE_AUDIO_IN_VIDEO", "0").strip() not in {
    "0",
    "false",
    "False",
}
QWEN_OMNI_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_HF_INSTALL_GUIDE = (
    "Missing Qwen2.5-Omni runtime dependencies.\n"
)


class LocalQwenOmniAnalyzer:
    def __init__(
        self,
        *,
        model_id: str = QWEN_OMNI_MODEL_ID,
        cache_dir: Path = QWEN_OMNI_CACHE_DIR,
        max_new_tokens: int = QWEN_OMNI_MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.max_new_tokens = int(max_new_tokens)
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._process_mm_info = None
        self._is_thinker = False

    def load(self) -> None:
        if self.is_ready:
            return

        with self._lock:
            if self.is_ready:
                return

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir))

            try:
                from transformers import Qwen2_5OmniProcessor
                try:
                    from transformers import Qwen2_5OmniThinkerForConditionalGeneration as ModelClass
                    is_thinker = True
                except Exception:
                    from transformers import Qwen2_5OmniForConditionalGeneration as ModelClass
                    is_thinker = False
                from qwen_omni_utils import process_mm_info
            except Exception as exc:  # pragma: no cover - runtime dependency guard
                raise RuntimeError(_HF_INSTALL_GUIDE) from exc

            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
                "cache_dir": str(self.cache_dir),
            }
            try:
                model = ModelClass.from_pretrained(
                    self.model_id,
                    **model_kwargs,
                )
            except TypeError:
                # Some preview versions use dtype instead of torch_dtype.
                model_kwargs.pop("torch_dtype", None)
                model_kwargs["dtype"] = "auto"
                model = ModelClass.from_pretrained(
                    self.model_id,
                    **model_kwargs,
                )
            if not is_thinker and hasattr(model, "disable_talker"):
                model.disable_talker()

            processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
            )

            self._model = model
            self._processor = processor
            self._process_mm_info = process_mm_info
            self._is_thinker = is_thinker

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._processor is not None and self._process_mm_info is not None

    def analyze(
        self,
        *,
        instruction: str,
        audio_paths: Sequence[str] = (),
        video_paths: Sequence[str] = (),
        ordered_media: Optional[Sequence[Tuple[str, str, str]]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> str:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")

        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._process_mm_info is not None

        def emit_log(msg: str) -> None:
            print(msg, flush=True)
            if on_log:
                try:
                    on_log(msg)
                except Exception:
                    pass

        def emit_delta(text: str) -> None:
            if not text:
                return
            if on_delta:
                try:
                    on_delta(text)
                except Exception:
                    pass

        started = time.time()
        emit_log("[qwen-omni] start AV analysis")

        content: List[dict] = []
        temp_dir = Path(tempfile.mkdtemp(prefix="qwen_omni_video_"))
        try:
            media = list(ordered_media) if ordered_media is not None else (
                [("video", path, f"[IN_VIDEO_{i}]") for i, path in enumerate(video_paths, 1)]
                + [("audio", path, f"[IN_AUDIO_{i}]") for i, path in enumerate(audio_paths, 1)]
            )
            video_count = 0
            audio_count = 0
            for modality, path, tag in media:
                if modality not in {"audio", "video"}:
                    raise ValueError(f"Unsupported AV modality: {modality}")
                p = Path(path).expanduser().resolve()
                if not p.is_file():
                    raise FileNotFoundError(f"AV input not found: {p}")
                content.append({"type": "text", "text": f"Input {tag}:"})
                if modality == "video":
                    video_count += 1
                    prepared = self._prepare_video_for_omni(p, temp_dir, video_count)
                    content.append({"type": "video", "video": str(prepared)})
                else:
                    audio_count += 1
                    content.append({"type": "audio", "audio": str(p)})

            if not content:
                return "QWEN_OMNI_AV_REPORT\n- MediaFindings:\n  - none\n- DenseCaptions:\n  - none\n- CandidateAnswer:\n  none\n- Uncertainty:\n  no audio/video media provided"

            content.append({"type": "text", "text": instruction})
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": QWEN_OMNI_DEFAULT_SYSTEM_PROMPT}],
                },
                {"role": "user", "content": content},
            ]

            use_audio_in_video = QWEN_OMNI_USE_AUDIO_IN_VIDEO and video_count > 0
            emit_log(
                f"[qwen-omni] inputs ready: videos={video_count}, audios={audio_count}, use_audio_in_video={use_audio_in_video}",
            )

            emit_log("[qwen-omni] building chat template ...")
            text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            emit_log("[qwen-omni] process_mm_info ...")
            audios, images, videos = self._process_mm_info(
                messages,
                use_audio_in_video=use_audio_in_video,
            )
            emit_log("[qwen-omni] building model inputs ...")
            inputs = self._processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )

            device = self._resolve_model_device()
            if device is not None:
                try:
                    inputs = inputs.to(device)
                except Exception:
                    pass

            emit_log("[qwen-omni] generating ...")
            result_text = ""
            streamed = False
            streamed_parts: List[str] = []
            generate_kwargs = {
                "use_audio_in_video": use_audio_in_video,
                "max_new_tokens": self.max_new_tokens,
            }
            # For AV understanding we only need text output, matching official usage.
            if not self._is_thinker:
                generate_kwargs["return_audio"] = False

            try:
                from transformers import TextIteratorStreamer

                tokenizer = getattr(self._processor, "tokenizer", None)
                if tokenizer is not None:
                    streamer = TextIteratorStreamer(
                        tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )
                    err: dict[str, Exception] = {}
                    kwargs = dict(
                        **inputs,
                        **generate_kwargs,
                        streamer=streamer,
                    )

                    def _generate_worker() -> None:
                        try:
                            with torch.inference_mode():
                                self._model.generate(**kwargs)
                        except Exception as e:
                            err["exc"] = e
                        finally:
                            # Unblock the iterator even when model.generate fails.
                            streamer.end()

                    t = threading.Thread(target=_generate_worker, daemon=True)
                    t.start()

                    emit_log("[qwen-omni] output(begin, streaming)")
                    for piece in streamer:
                        if not piece:
                            continue
                        streamed_parts.append(piece)
                        emit_delta(piece)
                    t.join()
                    if "exc" in err:
                        raise err["exc"]
                    result_text = "".join(streamed_parts).strip()
                    emit_log("[qwen-omni] output(end, streaming)")
                    streamed = True
            except Exception:
                if streamed_parts:
                    raise
                streamed = False

            if not streamed:
                with torch.inference_mode():
                    generated = self._model.generate(
                        **inputs,
                        **generate_kwargs,
                    )

                if hasattr(generated, "sequences"):
                    generated_ids = generated.sequences
                else:
                    generated_ids = generated

                try:
                    prompt_len = int(inputs["input_ids"].shape[1])
                    if generated_ids.ndim == 2 and generated_ids.shape[1] > prompt_len:
                        generated_ids = generated_ids[:, prompt_len:]
                except Exception:
                    pass

                outputs = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                result_text = str(outputs[0]).strip() if outputs else ""
                if result_text:
                    emit_delta(result_text)
                emit_log("[qwen-omni] output(fallback non-streaming)")

            emit_log(f"[qwen-omni] done in {time.time() - started:.1f}s")
            return result_text
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _prepare_video_for_omni(self, src: Path, temp_dir: Path, index: int) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return src

        out = temp_dir / f"video_{index:03d}_qwen.mp4"
        vf = "scale='if(gt(iw,1280),1280,iw)':'if(gt(ih,720),720,ih)':force_original_aspect_ratio=decrease"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-t",
            str(max(1, QWEN_OMNI_VIDEO_MAX_SECONDS)),
            "-vf",
            vf,
            "-r",
            str(max(1, QWEN_OMNI_VIDEO_FPS)),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(out),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            if out.is_file():
                return out
        except Exception:
            pass
        return src

    def _resolve_model_device(self) -> Optional[torch.device]:
        thinker = getattr(self._model, "thinker", None)
        if thinker is not None:
            d = getattr(thinker, "device", None)
            if d is not None and str(d) != "meta":
                return d
        d = getattr(self._model, "device", None)
        if d is not None and str(d) != "meta":
            return d
        return None


_GLOBAL_ANALYZER: Optional[LocalQwenOmniAnalyzer] = None
_GLOBAL_LOCK = threading.Lock()


def get_analyzer() -> LocalQwenOmniAnalyzer:
    global _GLOBAL_ANALYZER
    if _GLOBAL_ANALYZER is not None:
        return _GLOBAL_ANALYZER
    with _GLOBAL_LOCK:
        if _GLOBAL_ANALYZER is None:
            _GLOBAL_ANALYZER = LocalQwenOmniAnalyzer()
        return _GLOBAL_ANALYZER


def warmup_qwen_omni() -> None:
    get_analyzer().load()


def analyze_with_qwen_omni(
    *,
    instruction: str,
    audio_paths: Sequence[str] = (),
    video_paths: Sequence[str] = (),
    ordered_media: Optional[Sequence[Tuple[str, str, str]]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
    return get_analyzer().analyze(
        instruction=instruction,
        audio_paths=audio_paths,
        video_paths=video_paths,
        ordered_media=ordered_media,
        on_log=on_log,
        on_delta=on_delta,
    )
