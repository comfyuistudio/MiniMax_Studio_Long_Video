"""Use ComfyUI's native H3 guide, sampler, decoder and video implementations."""
from typing import Any
from collections.abc import Mapping
import logging

from .pipeline import CONTEXT_FRAMES, FPS, Segment

LOGGER = logging.getLogger(__name__)


class H3Runtime:
    def __init__(self, model: Any, clip: Any, vae: Any, audio_vae: Any,
                 prompt: str, width: int, height: int, steps: int, cfg: float,
                 first_frame: Any, sampler: Any, sigmas: Any,
                 reference_image: Any = None, audio_input: Any = None):
        try:
            import torch
            import comfy.samplers
            import comfy.model_management
            import comfy.utils
            import nodes
            from comfy_api.latest import InputImpl, Types
            from comfy_extras.nodes_minimax_h3 import (
                EmptyMiniMaxH3LatentAV, MiniMaxH3ImageToVideo, MiniMaxH3AddGuide,
                MiniMaxH3ReferenceToVideo,
            )
            from comfy_extras.nodes_custom_sampler import (
                BasicScheduler, Noise_RandomNoise, SamplerCustomAdvanced,
            )
            from comfy_extras.nodes_audio import VAEDecodeAudio
            from comfy_extras.nodes_video import CreateVideo
            if not hasattr(InputImpl, "VideoFromList"):
                raise ImportError("VideoFromList is unavailable")
        except ImportError as error:
            raise RuntimeError(
                "Update ComfyUI: this node requires native MiniMax H3, "
                "MiniMaxH3AddGuide and VideoFromList support. Easy Media is not required."
            ) from error
        self.torch, self.comfy, self.nodes = torch, comfy, nodes
        self.InputImpl, self.Types = InputImpl, Types
        self.Empty, self.AddGuide = EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide
        self.ImageToVideo, self.ReferenceToVideo = MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
        self.Noise, self.Sample = Noise_RandomNoise, SamplerCustomAdvanced
        self.DecodeAudio, self.CreateVideo = VAEDecodeAudio, CreateVideo
        self.model, self.vae, self.audio_vae = model, vae, audio_vae
        self.width, self.height, self.cfg = width, height, cfg
        self.audio_input = self._validate_audio(audio_input)
        self.emitted_frames = 0
        self.frame_chunks, self.audio_chunks = [], []
        self.final_latent = None
        self.bar = comfy.utils.ProgressBar(1)
        self.check_cancelled()
        # Encode the user's single prompt once. Keep the optional initial image
        # in its text embedding, but remove its time-zero anchor on later clips.
        self.positive = self._encode_conditioning(clip, prompt, first_frame, reference_image)
        self.negative = clip.encode_from_tokens_scheduled(clip.tokenize("", images=[])) if cfg != 1.0 else self.positive
        self.sampler = sampler if sampler is not None else comfy.samplers.sampler_object("euler")
        self.sigmas = sigmas if sigmas is not None else BasicScheduler.execute(model, "simple", steps, 1.0)[0]
        if len(self.sigmas) < 2 or not torch.isfinite(self.sigmas).all() or (self.sigmas < 0).any():
            raise ValueError("Sigmas must contain at least two finite, nonnegative values.")
        if float(self.sigmas[-1]) != 0.0 or float(self.sigmas[0]) <= 0 or (self.sigmas[1:] > self.sigmas[:-1]).any():
            raise ValueError("Use a descending full-denoise sigma schedule ending at zero.")

    def _validate_audio(self, audio: Any) -> Any:
        if audio is None:
            return None
        if not isinstance(audio, Mapping):
            raise ValueError(f"audio_input must be a ComfyUI AUDIO mapping; received {type(audio).__name__}.")
        # VHS_LoadVideo returns LazyAudioMap, a Mapping rather than a dict.
        # Access only the required keys to materialize its waveform once;
        # do not iterate, test truthiness, or coerce the entire lazy mapping.
        try:
            wave, rate = audio["waveform"], audio["sample_rate"]
        except KeyError as error:
            raise ValueError(f"audio_input is missing the required field {error.args[0]!r}.") from error
        if (not isinstance(wave, self.torch.Tensor) or wave.ndim != 3
                or wave.shape[0] != 1 or wave.shape[1] not in (1, 2) or wave.shape[-1] < 1):
            raise ValueError("audio_input requires one nonempty mono/stereo waveform [1, channels, samples].")
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise ValueError("audio_input sample_rate must be a positive integer.")
        if not self.torch.isfinite(wave).all():
            raise ValueError("audio_input contains non-finite samples.")
        return {"waveform": wave.detach().to(device="cpu", dtype=self.torch.float32), "sample_rate": rate}

    def _audio_window(self, start_frame: int, frame_count: int) -> dict[str, Any]:
        rate = self.audio_input["sample_rate"]
        start = round(start_frame * rate / FPS)
        end = round((start_frame + frame_count) * rate / FPS)
        if start < 0 or end <= start:
            raise ValueError("Invalid audio timeline window.")
        part = self.audio_input["waveform"][..., start:end]
        missing = end - start - part.shape[-1]
        if missing:
            part = self.torch.nn.functional.pad(part, (0, missing))
        return {"waveform": part.clone().contiguous(), "sample_rate": rate}

    def _encode_conditioning(self, clip: Any, prompt: str, first_frame: Any, reference_image: Any) -> Any:
        common = dict(clip=clip, vae=self.vae, prompt=prompt,
                      width=self.width, height=self.height, length=124)
        if reference_image is None:
            return self.ImageToVideo.execute(**common, first_frame=first_frame)[0]
        if getattr(reference_image, "ndim", 0) != 4 or reference_image.shape[0] != 1:
            raise ValueError("reference_image requires exactly one IMAGE, not a frame batch.")
        result = self.ReferenceToVideo.execute(
            **common, audio_vae=self.audio_vae, ref_image_size="match",
            ref_images={"ref_image_0": reference_image},
        )
        positive, latent = result[0], result[1]
        if first_frame is not None:
            positive = self.AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0,
                vae=self.vae, image=first_frame[:1],
            )[0]
        return positive

    def check_cancelled(self) -> None:
        self.comfy.model_management.throw_exception_if_processing_interrupted()

    def progress(self, current: int, total: int) -> None:
        self.bar.update_absolute(current, total)
        LOGGER.info("MiniMax H3: finished section %d/%d", current, total)

    def render(self, segment: Segment, context: Any, keep_context: bool) -> tuple[Any, Any]:
        torch = self.torch
        with torch.inference_mode():
            return self._render(segment, context, keep_context)

    def _render(self, segment: Segment, context: Any, keep_context: bool) -> tuple[Any, Any]:
        latent = self.Empty.execute(self.width, self.height, segment.generation_frames)[0]
        positive = [[embedding, metadata.copy()] for embedding, metadata in self.positive]
        if segment.prefix_frames:
            if context is None:
                raise RuntimeError("Previous section's motion context is missing.")
            for _, metadata in positive:
                metadata.pop("minimax_keyframes", None)
            # Context ends at the LAST DELIVERED frame, never the discarded
            # padding frame. Both streams start at frame zero of the overlap.
            positive = self.AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0,
                vae=self.vae, audio_vae=self.audio_vae,
                image=context[0], audio=context[1] if self.audio_input is None else None,
            )[0]
        if self.audio_input is not None:
            guide_audio = self._audio_window(
                self.emitted_frames - segment.prefix_frames, segment.generation_frames,
            )
            positive = self.AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0,
                audio_vae=self.audio_vae, audio=guide_audio,
            )[0]
        guider = self.comfy.samplers.CFGGuider(self.model)
        guider.set_conds(positive, self.negative)
        guider.set_cfg(self.cfg)
        sampled = self.Sample.execute(
            self.Noise(segment.seed), guider, self.sampler, self.sigmas, latent,
        )[1]
        self.check_cancelled()
        images = self.nodes.VAEDecode().decode(self.vae, sampled)[0]
        start = segment.prefix_frames
        end = start + segment.output_frames
        if images.ndim != 4 or len(images) < end:
            raise RuntimeError(f"H3 decoded too few video frames: need {end}, got {len(images)}.")
        frames = images[start:end].detach().to(device="cpu", copy=True).contiguous()
        audio = (self._audio_window(self.emitted_frames, segment.output_frames)
                 if self.audio_input is not None else self.DecodeAudio.execute(self.audio_vae, sampled)[0])
        rate = int(audio["sample_rate"])
        if rate <= 0:
            raise RuntimeError("H3 returned an invalid audio sample rate.")
        offset = round(start * rate / FPS) if self.audio_input is None else 0
        count = round(segment.output_frames * rate / FPS)
        wave = audio["waveform"][..., offset:offset + count].detach().to(device="cpu", copy=True)
        if wave.shape[-1] < count:
            raise RuntimeError("H3 decoded too few audio samples for the requested section.")
        trimmed_audio = {"waveform": wave.contiguous(), "sample_rate": rate}
        next_context = None
        if keep_context:
            tail_count = round(CONTEXT_FRAMES * rate / FPS)
            next_context = (
                frames[-CONTEXT_FRAMES:].clone(),
                {"waveform": wave[..., -tail_count:].clone(), "sample_rate": rate},
            )
        self.frame_chunks.append(frames)
        self.audio_chunks.append(trimmed_audio)
        self.emitted_frames += segment.output_frames
        if not keep_context:
            self.final_latent = {
                "samples": sampled["samples"].to("cpu"),
                "h3_trim_frames": start, "h3_output_frames": segment.output_frames, "h3_fps": FPS,
            }
        video = self.CreateVideo.execute(images=frames, fps=float(FPS), audio=trimmed_audio, codec="h264")[0]
        return video, next_context

    def join(self, videos: list[Any]) -> Any:
        if not self.frame_chunks or self.final_latent is None:
            raise RuntimeError("No complete H3 generation is available.")
        rate = self.audio_chunks[0]["sample_rate"]
        if any(a["sample_rate"] != rate for a in self.audio_chunks):
            raise RuntimeError("H3 audio sample rate changed between chunks.")
        images = self.torch.cat(self.frame_chunks, dim=0)
        audio = {"waveform": self.torch.cat([a["waveform"] for a in self.audio_chunks], dim=-1),
                 "sample_rate": rate}
        self.frame_chunks.clear()
        self.audio_chunks.clear()
        video = self.InputImpl.VideoFromList(videos, complete_audio=audio, codec=self.Types.VideoCodec("h264"))
        return video, images, audio, self.final_latent
