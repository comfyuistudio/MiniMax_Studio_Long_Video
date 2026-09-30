from collections.abc import Mapping
import types
import unittest

from test_runtime import Tensor, backend


class LazyAudioMap(Mapping):
    """VHS-style lazy keyed access; prohibit accidental full materialization."""
    def __init__(self, values):
        self.values = values
        self.accesses = []

    def __getitem__(self, key):
        self.accesses.append(key)
        return self.values[key]

    def __iter__(self):
        raise AssertionError("Do not enumerate lazy audio")

    def __len__(self):
        raise AssertionError("Do not check lazy audio length")


class AudioMappingTests(unittest.TestCase):
    def setUp(self):
        self.b = backend()
        self.b.torch = types.SimpleNamespace(Tensor=Tensor, float32="float32",
            isfinite=lambda wave: types.SimpleNamespace(all=lambda: True))

    def test_vhs_mapping_is_accepted_without_iteration(self):
        lazy = LazyAudioMap({"waveform": Tensor([0, 1, 2], True), "sample_rate": 48000})
        result = self.b._validate_audio(lazy)
        self.assertEqual(result["waveform"].values, [0, 1, 2])
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(lazy.accesses, ["waveform", "sample_rate"])
        self.assertIsInstance(result, dict)

    def test_plain_audio_and_disconnected_input_still_supported(self):
        self.assertIsNone(self.b._validate_audio(None))
        result = self.b._validate_audio({"waveform": Tensor([1], True), "sample_rate": 44100})
        self.assertEqual(result["sample_rate"], 44100)

    def test_missing_field_reports_required_key(self):
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            self.b._validate_audio(LazyAudioMap({"waveform": Tensor([1], True)}))

    def test_extraction_failure_is_preserved(self):
        class BrokenAudio(LazyAudioMap):
            def __getitem__(self, key):
                raise RuntimeError("VHS failed to extract audio")
        with self.assertRaisesRegex(RuntimeError, "VHS failed"):
            self.b._validate_audio(BrokenAudio({}))

    def test_invalid_waveform_or_rate_is_rejected(self):
        for wave, rate in [(Tensor([], True), 48000), (Tensor([1]), 48000),
                           (Tensor([1], True), 0), (Tensor([1], True), True)]:
            with self.subTest(rate=rate, shape=wave.shape), self.assertRaises(ValueError):
                self.b._validate_audio(LazyAudioMap({"waveform": wave, "sample_rate": rate}))

    def test_other_input_types_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "received str"):
            self.b._validate_audio("audio.wav")


if __name__ == "__main__":
    unittest.main()
