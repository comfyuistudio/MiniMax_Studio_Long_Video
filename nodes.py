from typing import Any
from comfy_api.latest import io
from .pipeline import MAX_SECONDS, MAX_CHUNK_SECONDS, generate, plan_segments


class MiniMaxH3SimpleVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SimpleContinuousVideo",
            display_name="MiniMax Studio Long Video v0.2.2",
            category="MiniMax Studio/Video",
            description="One prompt, selectable chunk size, rolling video/audio context, one video.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                default="A continuous tracking shot through a sunlit forest, with birdsong and rustling leaves. No cuts."),
                io.Int.Input("seconds", default=10, min=1, max=MAX_SECONDS,
                             tooltip="Total duration. The last chunk is shortened to fit."),
                io.Int.Input("width", default=1344, min=256, max=2048, step=32),
                io.Int.Input("height", default=768, min=256, max=2048, step=32),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=100, advanced=True),
                io.Float.Input("cfg", default=1.0, min=1.0, max=10.0, step=0.1, advanced=True),
                io.Image.Input("first_frame", optional=True),
                io.Sampler.Input("sampler", optional=True, advanced=True),
                io.Sigmas.Input("sigmas", optional=True, advanced=True),
                io.Int.Input("chunk_seconds", default=5, min=1, max=MAX_CHUNK_SECONDS,
                             optional=True, step=1,
                             tooltip="Output seconds per chunk (1–14). Default 5. Larger chunks use more memory. The final chunk is shortened automatically."),
                io.Image.Input("reference_image", optional=True,
                               tooltip="Persistent appearance reference. Use an H3 ref2va model and <Picture 1> in the prompt."),
                io.Audio.Input("audio_input", optional=True,
                               tooltip="Timed speech guidance. Original audio is retained, trimmed or silence-padded to seconds."),
            ],
            outputs=[io.Video.Output("video"), io.Image.Output("images"),
                     io.Audio.Output("audio"), io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, model: Any, clip: Any, vae: Any, audio_vae: Any,
                prompt: str, seconds: int, width: int, height: int, seed: int,
                steps: int = 20, cfg: float = 1.0, first_frame: Any = None,
                sampler: Any = None, sigmas: Any = None,
                chunk_seconds: int = 5, reference_image: Any = None,
                audio_input: Any = None) -> io.NodeOutput:
        plan_segments(seconds, seed, chunk_seconds)
        if not prompt.strip():
            raise ValueError("Enter a scene description in prompt.")
        if any(not isinstance(size, int) or size < 256 or size > 2048 or size % 32 for size in (width, height)):
            raise ValueError("Width and height must be multiples of 32 between 256 and 2048.")
        if not 1 <= steps <= 100 or not 1.0 <= cfg <= 10.0:
            raise ValueError("Steps must be 1–100 and CFG must be 1–10.")
        from .runtime import H3Runtime
        backend = H3Runtime(model, clip, vae, audio_vae, prompt, width, height,
                            steps, cfg, first_frame, sampler, sigmas,
                            reference_image=reference_image, audio_input=audio_input)
        return io.NodeOutput(*generate(backend, seconds, seed, chunk_seconds))
