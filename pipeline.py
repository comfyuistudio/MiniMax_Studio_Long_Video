"""Frame-exact planning and sequential execution, independent of ComfyUI."""
from dataclasses import dataclass
from typing import Any

FPS = 24
DEFAULT_CHUNK_SECONDS = 5
MAX_CHUNK_SECONDS = 14  # 14s + 22 context frames fits H3's 362-frame window.
CONTEXT_FRAMES = 22
MAX_SECONDS = 300


@dataclass(frozen=True)
class Segment:
    index: int
    output_frames: int
    prefix_frames: int
    generation_frames: int
    seed: int


def plan_segments(seconds: int, seed: int, chunk_seconds: int = DEFAULT_CHUNK_SECONDS) -> list[Segment]:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= MAX_SECONDS:
        raise ValueError(f"Duration must be a whole number from 1 to {MAX_SECONDS} seconds.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Seed must be an unsigned 64-bit integer.")
    if isinstance(chunk_seconds, bool) or not isinstance(chunk_seconds, int) or not 1 <= chunk_seconds <= MAX_CHUNK_SECONDS:
        raise ValueError(f"Chunk size must be a whole number from 1 to {MAX_CHUNK_SECONDS} seconds.")
    remaining = seconds * FPS
    result = []
    while remaining:
        index = len(result)
        wanted = min(chunk_seconds * FPS, remaining)
        prefix = CONTEXT_FRAMES if index else 0
        # H3's trained minimum is about five seconds. Overlap and temporal
        # padding are generated internally and never extend the delivered clip.
        needed = max(124, wanted + prefix)
        generated = needed + (5 - needed) % 17
        result.append(Segment(index, wanted, prefix, generated, (seed + index) & 0xFFFFFFFFFFFFFFFF))
        remaining -= wanted
    return result


def generate(backend: Any, seconds: int, seed: int, chunk_seconds: int = DEFAULT_CHUNK_SECONDS) -> Any:
    segments = plan_segments(seconds, seed, chunk_seconds)
    context = None
    videos = []
    for segment in segments:
        backend.check_cancelled()
        video, context = backend.render(segment, context, segment.index < len(segments) - 1)
        videos.append(video)
        backend.progress(segment.index + 1, len(segments))
    backend.check_cancelled()
    return backend.join(videos)
