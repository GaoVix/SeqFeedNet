"""Reconstruct a background using video frames and existing foreground masks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


def read_video_frame(video: Path, frame_index: int) -> np.ndarray:
    """Decode from the beginning to match predict_video.py's zero-based numbering."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        for index in range(frame_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {frame_index}; video ended at frame {index}.')
        return frame
    finally:
        capture.release()


def prepare_mask(mask: np.ndarray, shape: tuple[int, int], dilate: int, allow_full=False) -> np.ndarray:
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f'Mask size {mask.shape} does not match video frame size {shape}.')
    binary = (mask != 0).astype(np.uint8) * 255
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        binary = cv2.dilate(binary, kernel)
    if not allow_full and np.all(binary):
        raise ValueError('The mask covers the entire frame; no background pixels are available.')
    return binary


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, radius: float, method: str) -> np.ndarray:
    """Fill nonzero mask pixels; keep all pixels outside the mask exactly intact."""
    if not np.any(mask):
        return frame.copy()
    flag = {'telea': cv2.INPAINT_TELEA, 'ns': cv2.INPAINT_NS}[method]
    background = cv2.inpaint(frame, mask, radius, flag)
    background[mask == 0] = frame[mask == 0]
    return background


def comparison_panel(frame, mask, background, method):
    removed = frame.copy()
    removed[mask != 0] = 0
    panels = []
    for label, image in [('Original', frame), ('Foreground removed (black = hole)', removed),
                         (f'OpenCV {method} inpainting', background)]:
        title = np.full((44, frame.shape[1], 3), 245, np.uint8)
        cv2.putText(title, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
        panels.append(np.vstack((title, image)))
    return np.hstack(panels)


def image_panel(images):
    panels = []
    for label, image in images:
        title = np.full((44, image.shape[1], 3), 245, np.uint8)
        cv2.putText(title, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
        panels.append(np.vstack((title, image)))
    return np.hstack(panels)


def check_outputs(output_dir, names, inputs, overwrite):
    for name in names:
        path = output_dir / name
        if path.resolve() in inputs:
            raise ValueError(f'Output would overwrite an input file: {path}')
        if path.exists() and not overwrite:
            raise ValueError(f'Output exists: {path}. Use --overwrite or another output directory.')


def write_outputs(output_dir, images, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        if not cv2.imwrite(str(output_dir / name), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f'Cannot write image: {output_dir / name}')
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def masked_temporal_median(samples, valid, observations):
    """Aggregate only visible background, with bounded working memory per row block."""
    count, height, width, channels = samples.shape
    median = np.zeros((height, width, channels), np.uint8)
    rows = max(1, min(32, (32 * 1024 * 1024) // (count * width * channels * 2)))
    for start in range(0, height, rows):
        end = min(start + rows, height)
        block = np.array(samples[:, start:end], dtype=np.uint16)
        visible = np.asarray(valid[:, start:end], dtype=bool)
        # 256 sorts after all valid uint8 colors; masked samples cannot enter the median.
        block[~visible] = 256
        block.sort(axis=0)
        n = observations[start:end].astype(np.int64)
        lower = np.maximum(0, (n - 1) // 2)[None, :, :, None]
        upper = (n // 2)[None, :, :, None]
        low_values = np.take_along_axis(block, lower, axis=0)[0]
        high_values = np.take_along_axis(block, upper, axis=0)[0]
        values = (low_values + high_values).astype(np.float32) / 2
        values[n == 0] = 0
        median[start:end] = np.rint(values).astype(np.uint8)
        if end == height or (start // rows) % 40 == 0:
            print(f'Temporal median: rows {end}/{height}.', flush=True)
    return median


def run_temporal(args):
    if not args.video.is_file():
        raise ValueError(f'Video does not exist: {args.video}')
    if not args.masks_dir.is_dir():
        raise ValueError(f'Mask directory does not exist: {args.masks_dir}')
    output_dir = args.output_dir or Path('out') / args.video.stem / 'background_all_frames'
    names = ['original.png', 'temporal_background.png', 'unobserved_mask.png', 'observations.npy',
             'background.png', 'comparison.png', 'metadata.json']
    inputs = {args.video.resolve(), *(path.resolve() for path in args.masks_dir.glob('*.png'))}
    check_outputs(output_dir, names, inputs, args.overwrite)
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(args.video))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {args.video}')
        count = 0
        original = None
        observations = None
        # Disk-backed samples avoid retaining every HD frame as a float array in RAM.
        with tempfile.TemporaryFile() as frames_file, tempfile.TemporaryFile() as valid_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if observations is None:
                    shape = frame.shape
                    observations = np.zeros(shape[:2], np.uint32)
                elif frame.shape != shape:
                    raise ValueError('Video frame dimensions changed during decoding.')
                mask_path = args.masks_dir / f'{count:06d}.png'
                raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if raw_mask is None:
                    raise ValueError(f'Missing or unreadable mask for frame {count}: {mask_path}')
                mask = prepare_mask(raw_mask, shape[:2], args.dilate, allow_full=True)
                visible = (mask == 0).astype(np.uint8)
                observations += visible
                frames_file.write(frame.tobytes())
                valid_file.write(visible.tobytes())
                if count == args.frame_index:
                    original = frame.copy()
                count += 1
                if count % 100 == 0:
                    print(f'Collected {count} frames and masks.', flush=True)
            if not count:
                raise ValueError('Video has no decodable frames.')
            if original is None:
                raise ValueError(f'Comparison frame {args.frame_index} is outside the video ({count} frames).')
            holes = (observations < args.min_observations).astype(np.uint8) * 255
            if np.all(holes):
                raise ValueError('No pixels have enough visible background observations.')
            frames_file.flush()
            valid_file.flush()
            samples = np.memmap(frames_file, dtype=np.uint8, mode='r', shape=(count, *shape))
            valid = np.memmap(valid_file, dtype=np.uint8, mode='r', shape=(count, *shape[:2]))
            try:
                temporal = masked_temporal_median(samples, valid, observations)
            finally:
                del samples, valid
        temporal[holes != 0] = 0
        aggregate_s = time.perf_counter() - started
        started = time.perf_counter()
        background = inpaint_frame(temporal, holes, args.radius, args.method)
        inpaint_s = time.perf_counter() - started
        images = {
            'original.png': original,
            'temporal_background.png': temporal,
            'unobserved_mask.png': holes,
            'background.png': background,
            'comparison.png': image_panel([
                ('Reference frame', original), ('All-frame background (black = unobserved)', temporal),
                (f'Remaining holes filled ({args.method})', background),
            ]),
        }
        metadata = {
            'mode': 'temporal', 'video': str(args.video.resolve()), 'masks_dir': str(args.masks_dir.resolve()),
            'camera_assumption': 'fixed camera, same pixel coordinates correspond across all frames',
            'frames_used': count, 'frame_range_inclusive': [0, count - 1],
            'comparison_frame_index': args.frame_index, 'resolution': [shape[1], shape[0]],
            'aggregation': 'per-channel median of only unmasked pixels across ALL decoded frames',
            'mask_dilation_radius_pixels': args.dilate, 'min_observations': args.min_observations,
            'background_observed_fraction': float(np.count_nonzero(holes == 0) / holes.size),
            'spatial_inpaint_fraction': float(np.count_nonzero(holes) / holes.size),
            'always_masked_fraction': float(np.count_nonzero(observations == 0) / observations.size),
            'method_for_remaining_holes': f'cv2.INPAINT_{args.method.upper()}',
            'radius_pixels': args.radius, 'opencv_version': cv2.__version__,
            'decode_mask_collection_and_median_seconds': aggregate_s,
            'remaining_holes_inpaint_seconds': inpaint_s,
            'unchanged_outside_remaining_holes': bool(np.array_equal(background[holes == 0], temporal[holes == 0])),
            'note': 'Foreground predictions are excluded before aggregation. Segmentation errors, '
                    'background motion or camera motion can cause artifacts. Hidden regions without '
                    'enough valid observations are approximated by spatial inpainting.',
        }
        write_outputs(output_dir, images, metadata)
        np.save(output_dir / 'observations.npy', observations, allow_pickle=False)
        print(f'Used all {count} frames. Background observed: {metadata["background_observed_fraction"]:.1%}; '
              f'remaining holes: {metadata["spatial_inpaint_fraction"]:.1%}.\n'
              f'Background: {output_dir / "background.png"}\n'
              f'Comparison: {output_dir / "comparison.png"}', flush=True)
    finally:
        capture.release()


def run_single(args):
    mask_path = args.mask or args.masks_dir / f'{args.frame_index:06d}.png'
    output_dir = args.output_dir or Path('out') / args.video.stem / f'inpaint_{args.frame_index:06d}_{args.method}'
    names = ['original.png', 'mask.png', 'inpaint_mask.png', 'foreground_removed.png',
             'background.png', 'comparison.png', 'metadata.json']
    input_paths = {args.video.resolve(), mask_path.resolve()}
    check_outputs(output_dir, names, input_paths, args.overwrite)
    if not args.video.is_file():
        raise ValueError(f'Video does not exist: {args.video}')
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise ValueError(f'Cannot read mask: {mask_path}')
    frame = read_video_frame(args.video, args.frame_index)
    mask = prepare_mask(raw_mask, frame.shape[:2], args.dilate)
    started = time.perf_counter()
    background = inpaint_frame(frame, mask, args.radius, args.method)
    inpaint_s = time.perf_counter() - started
    removed = frame.copy()
    removed[mask != 0] = 0
    images = {
        'original.png': frame,
        'mask.png': raw_mask,
        'inpaint_mask.png': mask,
        'foreground_removed.png': removed,
        'background.png': background,
        'comparison.png': comparison_panel(frame, mask, background, args.method),
    }
    metadata = {
        'mode': 'single',
        'video': str(args.video.resolve()), 'mask': str(mask_path.resolve()),
        'frame_index': args.frame_index, 'frame_numbering': 'zero-based, sequential decoding',
        'resolution': [frame.shape[1], frame.shape[0]], 'method': f'cv2.INPAINT_{args.method.upper()}',
        'opencv_version': cv2.__version__, 'radius_pixels': args.radius,
        'mask_dilation_radius_pixels': args.dilate,
        'raw_foreground_fraction': float(np.count_nonzero(raw_mask) / raw_mask.size),
        'inpaint_fraction': float(np.count_nonzero(mask) / mask.size),
        'inpaint_seconds': inpaint_s,
        'unchanged_outside_inpaint_mask': bool(np.array_equal(background[mask == 0], frame[mask == 0])),
        'note': 'Single-frame spatial inpainting with existing mask; no model rerun. '
                'Hidden content is approximated from nearby pixels, not recovered ground truth. '
                'Large foreground regions may produce blurred fills. Timing covers inpainting only.',
    }
    write_outputs(output_dir, images, metadata)
    print(f'Frame: {args.frame_index}; foreground: {metadata["raw_foreground_fraction"]:.1%}; '
          f'inpaint: {inpaint_s:.3f}s ({args.method}).\n'
          f'Background: {output_dir / "background.png"}\n'
          f'Comparison: {output_dir / "comparison.png"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--mode', choices=['single', 'temporal'], default='single',
                        help='Single-frame inpainting or fixed-camera aggregation of every frame')
    parser.add_argument('--frame-index', type=int, default=180,
                        help='Zero-based source frame; temporal mode uses it only in the comparison')
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument('--masks-dir', type=Path, help='Directory of masks named 000000.png, 000001.png, ...')
    masks.add_argument('--mask', type=Path, help='Mask PNG aligned with the selected video frame')
    parser.add_argument('--method', choices=['telea', 'ns'], default='telea', help='OpenCV inpainting algorithm')
    parser.add_argument('--radius', type=float, default=5.0, help='Inpainting neighborhood radius in pixels')
    parser.add_argument('--dilate', type=int, default=3, help='Expand mask by this many pixels; 0 uses the raw mask')
    parser.add_argument('--min-observations', type=int, default=5,
                        help='Temporal mode: minimum unmasked observations for each background pixel')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true', help='Replace only this script\'s named outputs')
    args = parser.parse_args()
    if args.frame_index < 0:
        parser.error('--frame-index must be >= 0')
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error('--radius must be finite and > 0')
    if args.dilate < 0:
        parser.error('--dilate must be >= 0')
    if args.min_observations < 1:
        parser.error('--min-observations must be >= 1')
    if args.mode == 'temporal' and args.masks_dir is None:
        parser.error('--mode temporal requires --masks-dir with a mask for every video frame')
    try:
        if args.mode == 'temporal':
            run_temporal(args)
        else:
            run_single(args)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
