"""Exercise the real trimming/guide orchestration with small tensor test doubles."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("h3_test_package")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
from h3_test_package.runtime import H3Runtime
from h3_test_package.pipeline import plan_segments


class Tensor:
    def __init__(self, values, audio=False):
        self.values = list(values)
        self.audio = audio
        self.ndim = 3 if audio else 4
        self.shape = (1, 1, len(self.values)) if audio else (len(self.values), 1, 1, 3)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[-1]
        return Tensor(self.values[item], self.audio)

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self.clone()

    def contiguous(self):
        return self

    def clone(self):
        return Tensor(self.values, self.audio)


class Guider:
    def __init__(self, model):
        pass

    def set_conds(self, positive, negative):
        pass

    def set_cfg(self, cfg):
        pass


def backend(frame_count=158, audio_count=316):
    b = H3Runtime.__new__(H3Runtime)
    b.width, b.height, b.cfg = 256, 256, 1.0
    b.frame_chunks, b.audio_chunks, b.final_latent = [], [], None
    b.audio_input, b.emitted_frames = None, 0
    b.positive = [["shared embedding", {"minimax_keyframes": ["initial image"], "other": 7}]]
    b.negative = []
    b.model, b.vae, b.audio_vae = "model", "vae", "audio_vae"
    b.sampler, b.sigmas = "sampler", [1, 0]
    b.Empty = types.SimpleNamespace(execute=lambda *args: ({"samples": "empty"},))
    b.comfy = types.SimpleNamespace(samplers=types.SimpleNamespace(CFGGuider=Guider))
    b.check_cancelled = lambda: None
    b.Noise = lambda seed: seed
    b.Sample = types.SimpleNamespace(execute=lambda *args: ("unused", {"samples": Tensor([123])}))
    b.nodes = types.SimpleNamespace(VAEDecode=lambda: types.SimpleNamespace(
        decode=lambda *args: (Tensor(range(frame_count)),)))
    b.DecodeAudio = types.SimpleNamespace(execute=lambda *args: (
        {"waveform": Tensor(range(audio_count), audio=True), "sample_rate": 48},))
    b.guides = []
    def add_guide(**kwargs):
        b.guides.append(kwargs)
        return (kwargs["positive"],)
    b.AddGuide = types.SimpleNamespace(execute=add_guide)
    b.CreateVideo = types.SimpleNamespace(execute=lambda **kwargs: (kwargs,))
    return b


class RuntimeTests(unittest.TestCase):
    def test_audio_guidance_continues_recording_and_preserves_output(self):
        b = backend()
        b.audio_input = {"waveform": Tensor(range(1000), True), "sample_rate": 48}
        b.DecodeAudio = types.SimpleNamespace(execute=lambda *args: self.fail("Unexpected generated audio decode"))
        first, context = b._render(plan_segments(7, 0)[0], None, True)
        second, _ = b._render(plan_segments(7, 0)[1], context, False)
        self.assertEqual(first["audio"]["waveform"].values, list(range(240)))
        self.assertEqual(second["audio"]["waveform"].values, list(range(240, 336)))
        self.assertEqual(b.guides[2]["audio"]["waveform"].values, list(range(196, 444)))
        self.assertIsNone(b.guides[1]["audio"])

    def test_reference_retained_with_rolling_context(self):
        b = backend()
        ref = Tensor([99])
        calls = []
        def encode(**kwargs):
            calls.append(kwargs)
            return [["embedding", {"minimax_refs": ["ref"]}]], {"samples": "empty"}
        b.ReferenceToVideo = types.SimpleNamespace(execute=encode)
        b.positive = b._encode_conditioning("clip", "<Picture 1>", None, ref)
        self.assertIs(calls[0]["ref_images"]["ref_image_0"], ref)
        b._render(plan_segments(10, 0)[1], (Tensor(range(22)), {}), False)
        self.assertEqual(b.guides[0]["positive"][0][1]["minimax_refs"], ["ref"])

    def test_full_media_and_final_latent_outputs(self):
        b = backend()
        b.torch = types.SimpleNamespace(cat=lambda ts, dim: Tensor([v for t in ts for v in t.values], dim == -1))
        b.InputImpl = types.SimpleNamespace(VideoFromList=lambda videos, **kwargs: (videos, kwargs))
        b.Types = types.SimpleNamespace(VideoCodec=lambda value: value)
        first, context = b._render(plan_segments(7, 0)[0], None, True)
        last, _ = b._render(plan_segments(7, 0)[1], context, False)
        video, images, audio, latent = b.join([first, last])
        self.assertEqual(len(images), 168)
        self.assertEqual(len(audio["waveform"]), 336)
        self.assertIs(video[1]["complete_audio"], audio)
        self.assertEqual(latent["h3_trim_frames"], 22)
        self.assertEqual(latent["h3_output_frames"], 48)

    def test_first_section_context_ends_at_delivered_frame_not_padding(self):
        b = backend(124, 248)
        video, context = b._render(plan_segments(10, 0)[0], None, True)
        self.assertEqual(video["images"].values, list(range(120)))
        self.assertEqual(context[0].values, list(range(98, 120)))
        self.assertEqual(context[1]["waveform"].values, list(range(196, 240)))
        self.assertEqual(b.guides, [])

    def test_continuation_removes_overlap_and_uses_both_context_streams(self):
        b = backend()
        previous = (Tensor(range(22)), {"waveform": Tensor(range(44), True), "sample_rate": 48})
        video, context = b._render(plan_segments(15, 0)[1], previous, True)
        self.assertEqual(video["images"].values, list(range(22, 142)))
        self.assertEqual(video["audio"]["waveform"].values, list(range(44, 284)))
        self.assertEqual(context[0].values, list(range(120, 142)))
        self.assertEqual(context[1]["waveform"].values, list(range(240, 284)))
        self.assertIs(b.guides[0]["image"], previous[0])
        self.assertIs(b.guides[0]["audio"], previous[1])
        self.assertNotIn("minimax_keyframes", b.guides[0]["positive"][0][1])
        self.assertEqual(b.positive[0][1]["minimax_keyframes"], ["initial image"])
        self.assertEqual(video["codec"], "h264")

    def test_final_partial_section_has_no_retained_context(self):
        b = backend()
        video, context = b._render(plan_segments(12, 0)[2], (Tensor(range(22)), {}), False)
        self.assertEqual(video["images"].values, list(range(22, 70)))
        self.assertEqual(len(video["audio"]["waveform"].values), 96)
        self.assertIsNone(context)

    def test_missing_context_fails(self):
        with self.assertRaisesRegex(RuntimeError, "context is missing"):
            backend()._render(plan_segments(10, 0)[1], None, False)

    def test_short_decode_fails_instead_of_silently_shortening_video(self):
        for frames, samples, message in ((119, 240, "video frames"), (124, 239, "audio samples")):
            with self.subTest(frames=frames), self.assertRaisesRegex(RuntimeError, message):
                backend(frames, samples)._render(plan_segments(5, 0)[0], None, False)


if __name__ == "__main__":
    unittest.main()
