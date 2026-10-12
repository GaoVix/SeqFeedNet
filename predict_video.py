"""Run SeqFeedNet on a video without the CDnet dataset or training pipeline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize

from models.SeqFeedNet import SeqFeedNet
from models.unet import UNetVgg16
from utils.video_background import BackgroundReferences, estimate_video_backgrounds, load_repo_backgrounds, read_background


def load_model(weights_path: Path, device: torch.device) -> SeqFeedNet:
    try:
        weights = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    except RuntimeError as error:
        if 'PytorchStreamReader' not in str(error):
            raise
        raise ValueError(
            f'Cannot read checkpoint archive: {weights_path.resolve()} '
            f'({weights_path.stat().st_size} bytes). The file may be incomplete or corrupted. '
            f'Re-copy or re-download the complete .pt file before retrying.\nPyTorch: {error}'
        ) from error
    if isinstance(weights, dict):
        weights = weights.get('state_dict', weights.get('model_state_dict', weights))
    if not isinstance(weights, dict):
        raise ValueError('Expected a model state_dict checkpoint.')

    # Published cv*.pt files use the older names from Processor.mapping_new_weight_key.
    state_dict = {}
    for key, value in weights.items():
        key = key.removeprefix('module.')
        prefix, separator, suffix = key.partition('.')
        prefix = {'erd_model': 'si_encoder', 'me_model': 'fp_model'}.get(prefix, prefix)
        state_dict[prefix + separator + suffix] = value

    model = SeqFeedNet(UNetVgg16(12, 9), UNetVgg16(15, 1))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def inference_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    scale = min(1.0, max_size / max(width, height)) if max_size else 1.0
    # The U-Nets require both dimensions to be multiples of 16.
    return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)


class VideoPredictor:
    def __init__(self, model, empty_bgr, recent_bgr, size, device, frame_empty_bgr=None):
        self.model = model
        self.size = size
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.empty = self.rgb_tensor(empty_bgr)
        self.recent = self.rgb_tensor(recent_bgr)
        if self.empty.shape != self.recent.shape:
            raise ValueError('The initial empty and recent backgrounds must have matching dimensions.')
        empty_input = self.empty if frame_empty_bgr is None else self.rgb_tensor(frame_empty_bgr)
        self.empty_normalized = self.normalize(self.resize_input(empty_input).unsqueeze(0))
        self.features = None
        self.bg_only = self.normalize(self.resize_input(self.empty).unsqueeze(0))

    @staticmethod
    def rgb_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0

    def resize_input(self, value):
        # Match the repository's CPU bilinear, antialiased tensor resize.
        return resize(value, [self.size[1], self.size[0]], antialias=True).to(self.device)

    def normalize(self, value):
        return (value - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray, frame_empty_bgr=None) -> np.ndarray:
        frame_raw = self.rgb_tensor(bgr)
        if self.features is None:
            reference_frame = frame_raw
            if reference_frame.shape != self.recent.shape:
                raise ValueError('Background references must match the original video dimensions.')
            # Compute the difference at source resolution BEFORE resize, as in the repo.
            initial = torch.stack([self.empty, self.recent, torch.abs(self.recent - reference_frame)])
            self.features = self.model.si_encoder(self.normalize(self.resize_input(initial).unsqueeze(0)))
        if frame_empty_bgr is not None:
            self.empty_normalized = self.normalize(self.resize_input(self.rgb_tensor(frame_empty_bgr)).unsqueeze(0))

        pred, normalized_frame, self.features = self.model(
            self.normalize(self.resize_input(frame_raw).unsqueeze(0)), self.empty_normalized, self.features, self.bg_only
        )
        # Preserve the soft probability feedback used by Processor.get_bgOnly_and_mask.
        self.bg_only = normalized_frame * (1 - pred)
        return pred[0, 0].cpu().numpy()


def pad_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=0)


def make_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Tint foreground green while preserving the original background pixels."""
    overlay = frame.copy()
    selected = mask != 0
    tint = np.array([0, 255, 0], dtype=np.float32)
    overlay[selected] = np.rint(frame[selected] * (1 - alpha) + tint * alpha).astype(np.uint8)
    return overlay


def run(args):
    if not args.video.is_file():
        raise ValueError(f'Video does not exist: {args.video}')
    if not args.weights.is_file():
        raise ValueError(f'Checkpoint does not exist: {args.weights}')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif device_name.isdigit():
        device_name = f'cuda:{device_name}'
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu or enable GPU access in Docker.')

    output_dir = args.output_dir or Path('out') / args.video.stem
    overlay_path = output_dir / 'overlay.mp4'
    masks_dir = output_dir / 'masks'
    input_paths = {
        path.resolve()
        for path in [args.video, args.weights, args.background, args.shared_background, args.recent_bg, args.empty_bg_dir]
        if path is not None
    }
    for path in [overlay_path, masks_dir]:
        resolved = path.resolve()
        if any(resolved == source or resolved in source.parents for source in input_paths):
            raise ValueError(f'Output would overwrite an input file: {path}')
        if path.exists() and not args.overwrite:
            raise ValueError(f'Output exists: {path}. Use --overwrite or a different --output-dir.')

    capture = cv2.VideoCapture(str(args.video))
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f'Cannot decode video: {args.video}')
        height, width = frame.shape[:2]
        repo_mode = args.empty_bg_dir is not None
        max_size = args.max_size if args.max_size is not None else 0
        size = inference_size(width, height, max_size)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
            print('Video FPS is unavailable; using 30 FPS.', flush=True)

        background = None
        background_path = args.shared_background or args.background
        if background_path is not None:
            background = read_background(background_path, (height, width))
        print(f'Loading checkpoint: {args.weights.resolve()}', flush=True)
        model = load_model(args.weights, device)
        if args.shared_background is not None:
            references = BackgroundReferences(background, background, background)
            print('Background mode: shared (provided image for empty and recent references; '
                  'no video background estimation).', flush=True)
        elif repo_mode:
            references = load_repo_backgrounds(args.empty_bg_dir, args.recent_bg, (height, width), args.seed, args.ptz)
        else:
            references = estimate_video_backgrounds(args.video, args.empty_frames, args.recent_frames, background)
        predictor = VideoPredictor(
            model, references.initial_empty, references.recent, size, device, references.frame_empty
        )
        print(f'Device: {device}; inference: {size[0]}x{size[1]}; output FPS: {fps:g}', flush=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        if args.overwrite:
            # Remove only generated frame names, so a shorter rerun leaves no stale masks.
            for path in masks_dir.iterdir():
                if (path.is_file() and path.suffix == '.png' and len(path.stem) >= 6
                        and path.stem.isascii() and path.stem.isdigit()):
                    path.unlink()
        output_size = (width + width % 2, height + height % 2)
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create MP4 video: {overlay_path}')

        count = 0
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError('Video frame dimensions changed during decoding.')
            empty_frame = references.empty_at(count) if references.empty_sequence is not None else None
            probability = predictor.predict(frame, empty_frame)
            probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = (probability > args.threshold).astype(np.uint8) * 255
            mask_path = masks_dir / f'{count:06d}.png'
            if not cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f'Cannot write mask PNG: {mask_path}')
            overlay = make_overlay(frame, mask, args.overlay_alpha)
            writer.write(pad_even(overlay))
            count += 1
            if count % 100 == 0:
                print(f'Processed {count} frames.', flush=True)
            if args.max_frames and count >= args.max_frames:
                break
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f'Done: {count} frames.\nOverlay: {overlay_path}\nMasks (lossless PNG): {masks_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', type=Path, required=True, help='Input video file')
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parent / 'checkpoints/cv3.pt', help='SeqFeedNet checkpoint')
    parser.add_argument('--output-dir', type=Path, help='Output directory (default: out/<video name>)')
    parser.add_argument('--empty-bg-dir', type=Path, help='Repo mode: emptyBg images for this video')
    parser.add_argument('--recent-bg', type=Path, help='Repo mode: recentBg image aligned with the first input frame')
    parser.add_argument('--ptz', action='store_true', help='PTZ mode: per-frame emptyBg references, or 100/30-frame estimates')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the repo mode random background selection')
    background_options = parser.add_mutually_exclusive_group()
    background_options.add_argument('--background', type=Path,
                                    help='Estimate mode: clean empty background; recent is still estimated from video')
    background_options.add_argument('--shared-background', type=Path,
                                    help='Use one image for empty and recent references; skip video background estimation')
    parser.add_argument('--empty-frames', '--background-frames', type=int, help='Estimate mode: warm-up frames (default: 200, or 100 for PTZ)')
    parser.add_argument('--recent-frames', type=int, help='Estimate mode: recent tail of warm-up frames (default: up to 100, or 30 for PTZ)')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, or a GPU index such as 0')
    parser.add_argument('--max-size', type=int, default=0, help='Maximum inference dimension; 0 keeps native size (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Foreground probability threshold')
    parser.add_argument('--overlay-alpha', type=float, default=0.4, help='Green foreground overlay opacity, from 0 to 1')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit output frames; 0 processes the entire video')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the overlay and replace numbered mask PNGs')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('--threshold must be between 0 and 1, exclusively')
    if not 0 <= args.overlay_alpha <= 1:
        parser.error('--overlay-alpha must be between 0 and 1')
    if args.max_size is not None and args.max_size != 0 and args.max_size < 16:
        parser.error('--max-size must be 0 or >= 16')
    if args.max_frames < 0:
        parser.error('--max-frames must be >= 0')
    if (args.empty_bg_dir is None) != (args.recent_bg is None):
        parser.error('Repo mode requires both --empty-bg-dir and --recent-bg')
    if args.empty_bg_dir is not None and any(value is not None for value in [args.background, args.shared_background, args.empty_frames, args.recent_frames]):
        parser.error('--background, --shared-background, --empty-frames and --recent-frames cannot be used in repo mode')
    if args.shared_background is not None and any(value is not None for value in [args.empty_frames, args.recent_frames]):
        parser.error('--shared-background skips estimation; do not supply --empty-frames or --recent-frames')
    if args.empty_frames is None:
        args.empty_frames = 100 if args.ptz else 200
    if args.recent_frames is None:
        args.recent_frames = min(args.empty_frames, 30 if args.ptz else 100)
    if args.empty_frames < 1 or not 1 <= args.recent_frames <= args.empty_frames:
        parser.error('Frame windows must satisfy 1 <= --recent-frames <= --empty-frames')
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
