import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


agents = types.ModuleType("agents")
agents.Agent = object
agents.Runner = types.SimpleNamespace()
agents.function_tool = lambda func: func
openai = types.ModuleType("openai")
openai.OpenAI = object
qwen = types.ModuleType("qwen_omni")
qwen.analyze_with_qwen_omni = lambda **kwargs: "report"
qwen.warmup_qwen_omni = lambda: None
gradio = types.ModuleType("gradio")
gradio.Blocks = object
with patch.dict(sys.modules, {"agents": agents, "openai": openai, "qwen_omni": qwen, "gradio": gradio}):
    import unima
    import app


class PipelineTests(unittest.TestCase):
    def test_required_counts_only_from_brief_field(self):
        brief = "- Goal: explain image=9\n- RequiredOutputs: text=required; image=2; audio=0; video=1"
        self.assertEqual(
            unima.InterleaveAgentPipeline._extract_required_outputs(brief),
            {"image": 2, "audio": 0, "video": 1},
        )

    def test_output_tags_never_reassign_hallucinated_media(self):
        artifacts = [
            unima.Artifact("<image1>", "image", "/tmp/one.png", "one"),
            unima.Artifact("<image2>", "image", "/tmp/two.png", "two"),
        ]
        text, kept, dropped = unima.InterleaveAgentPipeline._normalize_output_tags(
            "First <image2>, invented <image8>, repeated <image2>.", artifacts
        )
        self.assertEqual([a.tag for a in kept], ["<image1>", "<image2>"])
        self.assertEqual(dropped, ["<image8>", "<image2>"])
        self.assertEqual(text.count("<image2>"), 1)
        self.assertEqual(text.count("<image1>"), 1)
        self.assertNotIn("<image8>", text)

    def test_async_runner_works_inside_existing_event_loop(self):
        async def scenario():
            return unima.InterleaveAgentPipeline._run_async(lambda: asyncio.sleep(0, result=7))

        self.assertEqual(asyncio.run(scenario()), 7)

    def test_stream_failure_after_tool_event_does_not_retry(self):
        class Stream:
            async def stream_events(self):
                yield types.SimpleNamespace(type="run_item_stream_event", name="tool_called", item=None)
                raise RuntimeError("stream disconnected")

        pipeline = object.__new__(unima.InterleaveAgentPipeline)
        pipeline.callbacks = unima.PipelineCallbacks(on_log=lambda _: None)
        with patch.object(unima.Runner, "run_streamed", return_value=Stream(), create=True), patch.object(
            unima.Runner, "run_sync", create=True
        ) as sync:
            with self.assertRaisesRegex(RuntimeError, "refusing to repeat tool calls"):
                pipeline._stream_agent_output(
                    agent=object(), input_payload=[], stage_name="test", model_name="fake", on_delta=None
                )
            sync.assert_not_called()

    def test_av_inputs_preserve_interleaved_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "a.wav"
            video = root / "v.mp4"
            audio.write_bytes(b"audio")
            video.write_bytes(b"video")
            pipeline = unima.InterleaveAgentPipeline(output_dir=temp, client=object())
            with patch.object(unima, "analyze_with_qwen_omni", return_value="report") as analyze:
                pipeline._build_interleave_input([
                    unima.InputBlock("audio", str(audio)),
                    unima.InputBlock("text", "Summarize"),
                    unima.InputBlock("video", str(video)),
                ])
            self.assertEqual(
                analyze.call_args.kwargs["ordered_media"],
                [("audio", str(audio), "[IN_AUDIO_1]"), ("video", str(video), "[IN_VIDEO_1]")],
            )

    def test_missing_media_token_is_rejected_before_run(self):
        with self.assertRaisesRegex(ValueError, "has no attached file"):
            app._build_input_blocks("Explain [IMG1]", {})


if __name__ == "__main__":
    unittest.main()
