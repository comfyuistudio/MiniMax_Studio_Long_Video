import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_pipeline", ROOT / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


class Backend:
    def __init__(self):
        self.calls = []
        self.updates = []
        self.checks = 0
        self.cancel_after = None

    def check_cancelled(self):
        self.checks += 1
        if self.cancel_after is not None and len(self.calls) >= self.cancel_after:
            raise InterruptedError("cancelled")

    def render(self, segment, context, keep_context):
        self.calls.append((segment, context, keep_context))
        return f"video{segment.index}", f"tail{segment.index}" if keep_context else None

    def progress(self, current, total):
        self.updates.append((current, total))

    def join(self, videos):
        return videos


class PlanningTests(unittest.TestCase):
    def test_all_chunk_sizes_preserve_duration_and_model_window(self):
        for chunk in range(1, 15):
            for seconds in range(1, 301):
                plan = pipeline.plan_segments(seconds, 0, chunk)
                self.assertEqual(sum(s.output_frames for s in plan), seconds * 24)
                self.assertEqual(len(plan), (seconds + chunk - 1) // chunk)
                for s in plan:
                    self.assertTrue(1 <= s.output_frames <= chunk * 24)
                    self.assertTrue(124 <= s.generation_frames <= 362)
                    self.assertEqual(s.generation_frames % 17, 5)
                    self.assertGreaterEqual(s.generation_frames, s.prefix_frames + s.output_frames)
                    self.assertEqual(s.prefix_frames, 22 if s.index else 0)

    def test_selected_chunk_size_reaches_generation(self):
        b = Backend()
        pipeline.generate(b, 20, 0, chunk_seconds=8)
        self.assertEqual([c[0].output_frames for c in b.calls], [192, 192, 96])
        self.assertEqual([c[1] for c in b.calls], [None, "tail0", "tail1"])

    def test_invalid_chunk_sizes(self):
        for value in (0, -1, 15, 1.5, True, None, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.plan_segments(20, 0, value)

    def test_every_supported_duration_is_exact_and_bounded(self):
        for seconds in range(1, 301):
            plan = pipeline.plan_segments(seconds, 1)
            self.assertEqual(sum(s.output_frames for s in plan), seconds * 24)
            self.assertEqual(len(plan), (seconds + 4) // 5)
            for s in plan:
                self.assertLessEqual(s.output_frames, 120)
                self.assertGreaterEqual(s.generation_frames, 124)
                self.assertEqual(s.generation_frames % 17, 5)
                self.assertGreaterEqual(s.generation_frames, s.prefix_frames + s.output_frames)
                self.assertEqual(s.prefix_frames, 22 if s.index else 0)

    def test_five_seconds_has_one_section_and_no_overlap(self):
        self.assertEqual(pipeline.plan_segments(5, 0), [pipeline.Segment(0, 120, 0, 124, 0)])

    def test_partial_final_section(self):
        self.assertEqual([s.output_frames for s in pipeline.plan_segments(12, 10)], [120, 120, 48])

    def test_seeds_are_reproducible_and_wrap(self):
        a = pipeline.plan_segments(15, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(a, pipeline.plan_segments(15, 0xFFFFFFFFFFFFFFFF))
        self.assertEqual([s.seed for s in a], [0xFFFFFFFFFFFFFFFF, 0, 1])

    def test_invalid_durations(self):
        for value in (0, -1, 301, 1.5, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.plan_segments(value, 0)

    def test_invalid_seeds(self):
        for value in (-1, 2**64, 1.2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.plan_segments(5, value)

    def test_only_previous_context_is_passed_and_order_preserved(self):
        b = Backend()
        self.assertEqual(pipeline.generate(b, 12, 8), ["video0", "video1", "video2"])
        self.assertEqual([c[1] for c in b.calls], [None, "tail0", "tail1"])
        self.assertEqual([c[2] for c in b.calls], [True, True, False])
        self.assertEqual(b.updates, [(1, 3), (2, 3), (3, 3)])

    def test_cancel_before_next_section(self):
        b = Backend()
        b.cancel_after = 1
        with self.assertRaises(InterruptedError):
            pipeline.generate(b, 15, 0)
        self.assertEqual(len(b.calls), 1)

    def test_failure_is_not_returned_as_a_partial_video(self):
        b = Backend()
        def fail(*args):
            raise RuntimeError("sampling failed")
        b.render = fail
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            pipeline.generate(b, 10, 0)
        self.assertEqual(b.updates, [])


if __name__ == "__main__":
    unittest.main()
