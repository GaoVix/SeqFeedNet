"""Background references for standalone video inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tempfile

import cv2
import numpy as np


def read_background(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot read background image: {path}')
    if image.shape[:2] != shape:
        raise ValueError(f'Background {path} has size {image.shape[:2]}; expected video size {shape}.')
    return image


def lower_median(frames: list[np.ndarray]) -> np.ndarray:
    """Match torch.median: for an even count, choose the lower middle value."""
    if not frames:
        raise ValueError('No background frames were provided.')
    stack = np.stack(frames)
    middle = (len(frames) - 1) // 2
    stack.partition(middle, axis=0)
    return stack[middle].copy()


def median_from_video_samples(samples: np.ndarray) -> np.ndarray:
    """Compute the native-resolution lower median in small row blocks."""
    count, height, width, channels = samples.shape
    result = np.empty((height, width, channels), dtype=np.uint8)
    middle = (count - 1) // 2
    # Keep each working block around 32 MiB even for large videos/windows.
    rows = max(1, min(32, (32 * 1024 * 1024) // (count * width * channels)))
    for start in range(0, height, rows):
        end = min(start + rows, height)
        block = np.array(samples[:, start:end], copy=True)
        block.partition(middle, axis=0)
        result[start:end] = block[middle]
    return result


@dataclass
class BackgroundReferences:
    initial_empty: np.ndarray
    recent: np.ndarray
    frame_empty: np.ndarray
    empty_sequence: list[Path] | None = None

    def empty_at(self, frame_index: int) -> np.ndarray:
        if self.empty_sequence is None:
            return self.frame_empty
        if frame_index >= len(self.empty_sequence):
            raise ValueError(f'PTZ emptyBg sequence has no reference for video frame {frame_index + 1}.')
        return read_background(self.empty_sequence[frame_index], self.initial_empty.shape[:2])


def load_repo_backgrounds(empty_dir: Path, recent_path: Path, shape, seed=0, ptz=False):
    """Use the reference selection rules from utils/data_process.py."""
    if not empty_dir.is_dir():
        raise ValueError(f'emptyBg directory does not exist: {empty_dir}')
    paths = sorted(
        path for path in empty_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    )
    if not paths:
        raise ValueError(f'No background images found in {empty_dir}')
    recent = read_background(recent_path, shape)
    if ptz:
        empty = read_background(paths[0], shape)
        print('Background mode: repo PTZ (one emptyBg image per video frame).', flush=True)
        return BackgroundReferences(empty, recent, empty, paths)

    rng = random.Random(seed)

    def sample_empty():
        # Sampling is with replacement, including when fewer than 30 images exist.
        return lower_median([read_background(path, shape) for path in rng.choices(paths, k=30)])

    # The dataset independently samples the initial features and the inference input.
    initial_empty = sample_empty()
    frame_empty = sample_empty()
    print(f'Background mode: repo (30 emptyBg samples per reference; seed={seed}).', flush=True)
    return BackgroundReferences(initial_empty, recent, frame_empty)


def estimate_video_backgrounds(video: Path, empty_frames: int, recent_frames: int, background=None):
    """Offline approximation: a long window and its recent tail, then rewind."""
    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f'Cannot read any frames from {video}')
        shape = frame.shape
        print(f'Estimating references at source resolution from up to {empty_frames} frames.', flush=True)
        # Use temporary disk storage instead of retaining hundreds of HD frames in RAM.
        with tempfile.TemporaryFile() as temporary:
            samples = np.memmap(temporary, dtype=np.uint8, mode='w+', shape=(empty_frames, *shape))
            try:
                while ok and count < empty_frames:
                    if frame.shape != shape:
                        raise ValueError('Video frame dimensions changed during background estimation.')
                    samples[count] = frame
                    count += 1
                    if count < empty_frames:
                        ok, frame = capture.read()
                empty = median_from_video_samples(samples[:count])
                recent = median_from_video_samples(samples[max(0, count - recent_frames):count])
            finally:
                del samples
    finally:
        capture.release()
    if background is not None:
        empty = background
    print(
        f'Background mode: estimate (two-window initialization). Warm-up: {count} frames; '
        f'recent window: last {min(recent_frames, count)} warm-up frames. '
        'Inference starts at the first video frame.',
        flush=True,
    )
    return BackgroundReferences(empty, recent, empty)
