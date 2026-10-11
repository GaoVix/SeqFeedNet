import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from predict_video import VideoPredictor, inference_size, main, make_overlay
from utils.video_background import estimate_video_backgrounds, load_repo_backgrounds, lower_median


class BackgroundTests(unittest.TestCase):
    def test_even_median_matches_repository_torch_median(self):
        frames = [np.full((16, 16, 3), value, np.uint8) for value in [10, 20, 200, 250]]
        expected = torch.median(torch.from_numpy(np.stack(frames)), dim=0).values.numpy()
        np.testing.assert_array_equal(lower_median(frames), expected)
        self.assertEqual(int(expected[0, 0, 0]), 20)

    def test_repo_samples_with_replacement_and_separate_initial_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_dir = root / 'empty'
            empty_dir.mkdir()
            for name, value in [('a', 10), ('b', 200)]:
                self.assertTrue(cv2.imwrite(str(empty_dir / f'{name}.png'), np.full((16, 16, 3), value, np.uint8)))
            recent_path = root / 'recent.png'
            self.assertTrue(cv2.imwrite(str(recent_path), np.full((16, 16, 3), 70, np.uint8)))
            refs = load_repo_backgrounds(empty_dir, recent_path, (16, 16), seed=1)
            # Two available files can supply 30 samples; seed 1 gives different medians.
            np.testing.assert_array_equal(refs.initial_empty, np.full((16, 16, 3), 10, np.uint8))
            np.testing.assert_array_equal(refs.frame_empty, np.full((16, 16, 3), 200, np.uint8))
            np.testing.assert_array_equal(refs.recent, np.full((16, 16, 3), 70, np.uint8))
            with self.assertRaisesRegex(ValueError, 'expected video size'):
                load_repo_backgrounds(empty_dir, recent_path, (32, 32))

            ptz = load_repo_backgrounds(empty_dir, recent_path, (16, 16), ptz=True)
            self.assertEqual(int(ptz.empty_at(0)[0, 0, 0]), 10)
            self.assertEqual(int(ptz.empty_at(1)[0, 0, 0]), 200)
            with self.assertRaisesRegex(ValueError, 'no reference'):
                ptz.empty_at(2)

    def test_video_estimates_distinct_windows_and_handles_short_input(self):
        class Capture:
            def __init__(self):
                self.frames = iter([np.full((16, 16, 3), value, np.uint8) for value in [0, 10, 20, 30, 40, 50]])
                self.released = False

            def read(self):
                frame = next(self.frames, None)
                return frame is not None, frame

            def release(self):
                self.released = True

        capture = Capture()
        with patch('utils.video_background.cv2.VideoCapture', return_value=capture):
            refs = estimate_video_backgrounds(Path('input.mp4'), 200, 2)
        self.assertTrue(capture.released)
        self.assertEqual(int(refs.initial_empty[0, 0, 0]), 20)
        self.assertEqual(int(refs.recent[0, 0, 0]), 40)

        with patch('utils.video_background.cv2.VideoCapture', return_value=Capture()):
            refs = estimate_video_backgrounds(Path('input.mp4'), 4, 2)
        self.assertEqual(int(refs.initial_empty[0, 0, 0]), 10)
        self.assertEqual(int(refs.recent[0, 0, 0]), 20)


class PredictorTests(unittest.TestCase):
    def test_difference_before_resize_and_sequential_feedback(self):
        class Model:
            def __init__(self):
                self.calls = []

            def si_encoder(self, initial):
                self.initial = initial.clone()
                return initial

            def __call__(self, frame, empty, features, background):
                self.calls.append(tuple(value.clone() for value in [frame, empty, features, background]))
                return torch.full((1, 1, 16, 16), 0.25), frame, torch.ones(1, 9, 16, 16)

        model = Model()
        checker = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
        recent = np.repeat(checker[:, :, None], 3, axis=2)
        frame = 255 - recent
        initial_empty = np.full((32, 32, 3), 10, np.uint8)
        frame_empty = np.full((32, 32, 3), 200, np.uint8)
        predictor = VideoPredictor(model, initial_empty, recent, (16, 16), torch.device('cpu'), frame_empty)
        predictor.predict(frame)
        # Opposite checkerboards differ by 1 before resize, but nearly 0 after resize.
        expected_difference = ((1 - predictor.mean) / predictor.std).expand(1, 3, 16, 16)
        torch.testing.assert_close(model.initial[:, 2], expected_difference)
        torch.testing.assert_close(model.calls[0][3], ((10 / 255 - predictor.mean) / predictor.std).expand(1, 3, 16, 16))
        torch.testing.assert_close(model.calls[0][1], ((200 / 255 - predictor.mean) / predictor.std).expand(1, 3, 16, 16))

        predictor.predict(frame, initial_empty)
        torch.testing.assert_close(model.calls[1][3], model.calls[0][0] * 0.75)
        torch.testing.assert_close(model.calls[1][2], torch.ones(1, 9, 16, 16))
        torch.testing.assert_close(model.calls[1][1], ((10 / 255 - predictor.mean) / predictor.std).expand(1, 3, 16, 16))
        self.assertEqual(inference_size(1920, 1080, 0), (1920, 1072))

    def test_overlay_preserves_background_and_tints_only_foreground(self):
        frame = np.full((4, 4, 3), (30, 70, 110), np.uint8)
        original = frame.copy()
        mask = np.zeros((4, 4), np.uint8)
        mask[1, 2] = 255
        overlay = make_overlay(frame, mask, 0.4)
        np.testing.assert_array_equal(overlay[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(overlay[1, 2], [18, 144, 66])
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(make_overlay(frame, mask, 0), frame)

    def test_invalid_modes_and_opacity_fail_before_inference(self):
        for arguments in [
            ['--empty-bg-dir', '/empty'],
            ['--empty-bg-dir', '/empty', '--recent-bg', '/recent', '--empty-frames', '200'],
            ['--empty-frames', '30', '--recent-frames', '100'],
            ['--overlay-alpha', '1.5'],
        ]:
            with self.subTest(arguments=arguments):
                with patch('sys.argv', ['predict_video.py', '--video', 'input.mp4', *arguments]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        main()
                self.assertEqual(raised.exception.code, 2)


class MaskOutputTests(unittest.TestCase):
    def run_cli(self, root, probabilities, *extra):
        video = root / 'input.mp4'
        weights = root / 'weights.pt'
        video.touch()
        weights.touch()
        frames = [np.zeros((*probability.shape, 3), np.uint8) for probability in probabilities]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        capture.get.return_value = 30.0
        references = SimpleNamespace(
            initial_empty=frames[0], recent=frames[0], frame_empty=frames[0], empty_sequence=None
        )
        predictor = Mock()
        predictor.predict.side_effect = probabilities
        arguments = [
            'predict_video.py', '--video', str(video), '--weights', str(weights),
            '--device', 'cpu', '--output-dir', str(root / 'out'), *extra,
        ]
        with (patch('sys.argv', arguments),
              patch('predict_video.cv2.VideoCapture', return_value=capture),
              patch('predict_video.load_model'),
              patch('predict_video.estimate_video_backgrounds', return_value=references),
              patch('predict_video.VideoPredictor', return_value=predictor)):
            main()
        capture.release.assert_called_once()

    def test_lossless_binary_masks_keep_frame_order_and_odd_source_size(self):
        probability = np.tile(np.array([0.49, 0.5, 0.51], np.float32), (17, 7))[:, :19]
        probabilities = [probability, 1 - probability]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_cli(root, probabilities)
            masks_dir = root / 'out' / 'masks'
            self.assertEqual(sorted(path.name for path in masks_dir.iterdir()), ['000000.png', '000001.png'])
            for index, selected_column in enumerate([2, 0]):
                saved = cv2.imread(str(masks_dir / f'{index:06d}.png'), cv2.IMREAD_UNCHANGED)
                expected = np.zeros((17, 19), np.uint8)
                expected[:, selected_column::3] = 255
                np.testing.assert_array_equal(saved, expected)
                self.assertEqual(saved.dtype, np.uint8)
                self.assertEqual(set(np.unique(saved)), {0, 255})
            self.assertTrue((root / 'out' / 'overlay.mp4').is_file())
            self.assertFalse((root / 'out' / 'mask.mp4').exists())

    def test_shorter_overwrite_removes_stale_masks_and_preserves_other_files(self):
        probabilities = [np.ones((17, 19), np.float32)] * 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_cli(root, probabilities)
            masks_dir = root / 'out' / 'masks'
            original = (masks_dir / '000000.png').read_bytes()
            note = masks_dir / 'reference.png'
            note.write_bytes(b'keep me')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.run_cli(root, probabilities)
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual((masks_dir / '000000.png').read_bytes(), original)
            self.run_cli(root, [np.zeros((17, 19), np.float32)] * 2, '--overwrite', '--max-frames', '1')
            self.assertFalse((masks_dir / '000001.png').exists())
            self.assertEqual(note.read_bytes(), b'keep me')
            saved = cv2.imread(str(masks_dir / '000000.png'), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, np.zeros((17, 19), np.uint8))

    def test_failed_png_write_exits_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            error_output = io.StringIO()
            with (patch('predict_video.cv2.imwrite', return_value=False),
                  contextlib.redirect_stderr(error_output),
                  self.assertRaises(SystemExit) as raised):
                self.run_cli(Path(directory), [np.ones((17, 19), np.float32)])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn('Cannot write mask PNG:', error_output.getvalue())
            self.assertIn('000000.png', error_output.getvalue())

    def test_overwrite_does_not_delete_an_input_background_in_masks_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks_dir = root / 'out' / 'masks'
            masks_dir.mkdir(parents=True)
            background = masks_dir / '000000.png'
            background.write_bytes(b'input background')
            error_output = io.StringIO()
            with contextlib.redirect_stderr(error_output), self.assertRaises(SystemExit) as raised:
                self.run_cli(root, [np.ones((17, 19), np.float32)], '--overwrite', '--background', str(background))
            self.assertEqual(raised.exception.code, 1)
            self.assertIn('Output would overwrite an input', error_output.getvalue())
            self.assertEqual(background.read_bytes(), b'input background')


if __name__ == '__main__':
    unittest.main()
