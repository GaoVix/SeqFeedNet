# SeqFeedNet: Sequential Feature Feedback Network for Background Subtraction

---

## Abstract

Background subtraction (BGS) is a fundamental task in computer vision with applications in video surveillance, object tracking, and people recognition. Despite recent advancements, many deep learning-based BGS algorithms rely on large models to extract high-level representations, demanding significant computational resources and leading to inefficiencies in processing video streams. To address these limitations, we introduce the Sequential Feature Feedback Network (SeqFeedNet), a novel supervised algorithm for BGS in unseen videos that operates without additional pre-processing models. SeqFeedNet innovatively incorporates time-scale diverse sequential features and employs a feedback mechanism for each iteration. Moreover, we propose the Sequential Fit Training (SeqFiT) technique, enhancing model convergence during training. Evaluated on the CDNet 2014 dataset, SeqFeedNet not only achieves **$\sim5$** times increase in processing speed but also having better F-Measure scores than the leading supervised algorithms, making it highly suitable for real-world applications. Our experiment demonstrates that SeqFeedNet surpasses state-of-the-art network without pre-trained segmentation mdoel by **3.83\%** F-Measure on the CDnet 2014 dataset. Leading the way to establish a new benchmark for efficient and effective BGS on unseen videos.

| Previous                                                   | SeqFeedNet(Porposed)                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| ![previous-concept](./figures/system-concept-a.drawio.svg) | ![seqfeednet-concept](./figures/system-concept-b.drawio.svg) |

- The framework of SeqFeedNet.
  ![inference](./figures/inference.drawio.svg)

## Results

- Comparison of methods according to the per-category F-Measure for unseen videos from CDnet 2014.
  ![cdnet2014-compare](./figures/cdnet2014-compare.png)
  Without the pre-trained segmentation model, SeqFeedNet achieves the best BGS on unseen videos.

- Qualitative practical demand comparison for BGS.
  ![qualitative-practical](./figures/qualitative-practical.png)

  SeqFeedNet aims to achieve all the key factors for the practical application and deployment.

- FPS is calculated using PyTorch 2.1 implementation on a single Nvidia GeForce RTX 3090 GPU.
  ![perfromance](./figures/perfromance.png)
  Note: ZBS was not included in the speed tests due to its higher setup costs, although it reportedly uses C++ language and reaches about 20 FPS on an A100 GPU according to its publication.

  SeqFeedNet is **~5** times faster than the best supervised BGS algorithm.

Overall, SeqFeedNet become a **leading BGS algorithm for real-world application scenarios**.

## Performance

Follow the link to download the evaluation models mention on paper:
<https://drive.google.com/drive/folders/1GljFqnQp7vxh_96-moHWdUFEdTriTvh2?usp=sharing>

- Cross-validation of overall and per-category results of SeqFeedNet on the CDnet 2014 dataset.

| Model   | baseline | cameraJitter | badWeather | dynamicBackground | intermittentObjectMotion | lowFramerate | nightVideos | PTZ    | shadow | thermal | turbulence |   | Overall |
|---------|----------|--------------|------------|-------------------|--------------------------|--------------|-------------|--------|--------|---------|------------|---|---------|
| [cv1](https://drive.google.com/file/d/141c946KH4wnXTbFTpAIt05qAGWyqZAEB/view?usp=drive_link)     | 0.9719   | 0.8808       | 0.8846     | 0.9353            | 0.9111                   | 0.1843       | 0.6239      | 0.1541 | 0.9493 | 0.7675  | 0.8938     |   | 0.7415  |
| [cv2](https://drive.google.com/file/d/1eJH9inr6UjtTJA_-wWZiezB05o9LT2hs/view?usp=drive_link)     | 0.9685   | 0.8874       | 0.9432     | 0.3776            | 0.8228                   | 0.9826       | 0.7419      | 0.9151 | 0.9133 | 0.9423  | 0.7864     |   | 0.8437  |
| [cv3](https://drive.google.com/file/d/1U5xHNw9m00uepnQ-OQsPZAWHIaElZVBJ/view?usp=drive_link)     | 0.9881   | 0.9129       | 0.7645     | 0.9674            | 0.9004                   | 0.9589       | 0.6953      | 0.8387 | 0.9457 | 0.9384  | 0.9816     |   | 0.8993  |
| [cv4](https://drive.google.com/file/d/1AkrdZhSsOoHb21OfjMBT0sZ60TlHGfzZ/view?usp=drive_link)     | 0.9701   | 0.9509       | 0.7823     | 0.8166            | 0.7478                   | 0.9201       | 0.9336      | 0.8876 | 0.9438 | 0.9426  | 0.8330     |   | 0.8844  |
|         |          |              |            |                   |                          |              |             |        |        |         |            |   |         |
| Overall | 0.9747   | 0.9080       | 0.8437     | 0.7743            | 0.8455                   | 0.7615       | 0.7487      | 0.6989 | 0.9380 | 0.8977  | 0.8737     |   | 0.8422  |

- Ablation
  ![seq-loop](./figures/seq_loop-compare.png)

  The findings reveal that increasing the number of sequential loops by including additional frame groups significantly boosts the performance of SeqFeedNet.

## Installation

Test on `python 3.10`, `cuda 12.1`, and `torch 2.1.1`

```shell
pip3 install -r requirements.txt
pip3 install torch torchvision
```

### Docker (GPU server)

Requires a Linux x86_64 server with Docker, an NVIDIA driver compatible with CUDA 12.1,
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image uses PyTorch 2.1.1 / CUDA 12.1 and installs dependencies only; mount your code at `/workspace`.

Run from the repository root on the server:

```shell
git submodule update --init --recursive
docker build -t seqfeednet .
docker run --rm -it --gpus all --shm-size=8g -v "$PWD:/workspace" seqfeednet
```

Inside the container, verify the environment and run training/testing as shown below:

```shell
python -c "import torch, Processor; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'"
python training.py --help
```

Place the prepared dataset in `Data/` under the repository, or add
`-v /absolute/path/to/Data:/workspace/Data` to `docker run` before `seqfeednet`.
If `Data/` contains symlinks, their targets must also be accessible inside the container.
Outputs in `/workspace/out` persist in the host repository's `out/` directory.
Code changes take effect immediately; rebuild the image when dependencies change.

## Dataset

1. Download the CDnet 2014 Dataset:
   <http://jacarini.dinf.usherbrooke.ca/static/dataset/dataset2014.zip>

2. Download the model trained using both an empty background and a recent background as provided by M. Ozan Tezcan et al. at the following link:
   <https://drive.google.com/drive/folders/1fskxV1paCsoZvqTVLjnlAdPOCHk1_XmF>

3. Prepare the folder structure as follow:

   ```shell
   Data
    ├── currentFr -> cdnet2014/dataset/
    ├── emptyBg -> BSUV-Net-2.0_Training_Data/emptyBg
    └── recentBg -> BSUV-Net-2.0_Training_Data/recentBg
   ```

## Training

```shell
cross_validation_set=<number_of_set>
python3 training.py --device 0 -epochs 200 --batch_size 9 -workers 9 -cv $cross_validation_set -imghw 224-224 -use-t2val -opt Adam -out $cross_validation_set
```

The model's weight will be saved to the `out/` directory.

## Testing

```shell
cross_validation_set=<number_of_set>

# generate measurement csv
python3 testing.py -cv $cross_validation_set --device 0 -weight <model_weight>
# generate measurement csv & video result
python3 testing.py -cv $cross_validation_set --device 0 -weight <model_weight> -save
```

## Background subtraction on a video

Run from the repository root (or `/workspace` inside Docker). This script uses the
SeqFeedNet model directly and does not require the CDnet dataset:

```shell
python predict_video.py --video /path/to/input.mp4 --weights checkpoints/cv3.pt
```

On a GPU server, you can also run it directly in Docker without entering a shell:

```shell
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD:/workspace" -v /path/to/videos:/videos:ro \
  seqfeednet python predict_video.py --video /videos/input.mp4 --output-dir out/demo
```

Processing is headless: no display server or GUI is needed. Mount the repository
(including `checkpoints/`) at `/workspace`; the output directory above is saved in
the host repository. Code is mounted, so adding this script does not require an image rebuild.

The default checkpoint is `checkpoints/cv3.pt`. CUDA is selected automatically when
available; use `--device cpu` or `--device 0` to select a device explicitly.
Results are saved in `out/<video name>/`:

- `overlay.mp4`: the original video with predicted foreground tinted translucent green;
  background areas retain the original image. `--overlay-alpha` controls opacity (default `0.4`).
- `masks/000000.png`, `masks/000001.png`, ...: lossless, single-channel `uint8`
  binary masks at the exact source resolution. Foreground is `255` and background
  is `0`. Frame numbering starts at zero and follows the decoded input frames;
  there is no padding or frame sampling. `mask.mp4` is no longer generated.

The command above needs no additional flags to save PNG masks. `--overwrite`
replaces the overlay and removes previous numbered mask PNGs before writing the
new sequence, so rerunning with fewer frames does not leave stale masks. Other
files in the output directory are preserved. For batch resume checks, use
`masks/000000.png` instead of `mask.mp4`, together with your completion marker.

No background images are needed by default. The script estimates two references:
the pixelwise median of the first 200 frames for the long-term reference, and the
median of the last 100 frames within that initial window for the recent reference.
It then processes the video from its first frame with the model's sequential feedback.
For a shorter video, it uses the available frames; the references can coincide if
fewer than 100 frames are available. Customize the windows with:

```shell
python predict_video.py --video input.mp4 --empty-frames 200 --recent-frames 100 --output-dir out/demo
```

This is offline initialization using frames from the beginning of the video,
including frames later than the first output frame. It approximates the two kinds
of reference used by the official repository; it does not reproduce its precomputed
background data. Medians are computed at the original video resolution, with the
same lower-middle convention as `torch.median`. The initial difference is computed
before the repository's antialiased bilinear resize and normalization. Temporary
disk storage and row blocks avoid keeping all decoded samples in RAM; initialization
needs scratch space of approximately `frames * width * height * 3` bytes under
`TMPDIR` (normally `/tmp`), released after the references have been computed.
For moving cameras, `--ptz` defaults to 100/30-frame initialization windows; video-only
mode still uses a fixed empty reference after initialization. Objects that remain
in the same area during initialization can become part of the estimated background.

If a clean background image from the same camera view is available, use:

```shell
python predict_video.py --video input.mp4 --background empty_background.jpg --output-dir out/demo
```

The legacy `--background-frames` option is an alias for `--empty-frames`.
`--threshold` changes the mask threshold (default `0.5`). `--max-size` controls
inference resolution (default `640` for estimated references, rounded down to
multiples of 16); `--max-size 0` keeps native resolution rounded down to multiples
of 16. Use `--max-frames 100` for a short trial and `--overwrite` to replace existing
outputs. The overlay retains source FPS and resolution (odd dimensions are padded
by one pixel for MP4 encoding); audio is not copied. PNG masks retain the original
width and height even when either dimension is odd.

To use already prepared backgrounds with the repository's selection rules:

```shell
python predict_video.py --video input.mp4 \
  --empty-bg-dir /path/to/emptyBg/this_video \
  --recent-bg /path/to/recentBg/this_video/first_frame.jpg
```

Both references must match this video and its original resolution. In this mode,
30 empty-background images are sampled **with replacement**, and their lower median
is used, matching `torch.median` in `utils/data_process.py`. Separate draws provide
the initial feature reference and the foreground network's reference, as in the
dataset loader. `--seed` controls sampling (default `0`); a seed does not reproduce
the global random state of a separate dataset evaluation. The difference reference
is computed before the repository's antialiased bilinear resize and normalization.
This mode defaults to native inference resolution rounded down to multiples of 16.

For prepared PTZ references, add `--ptz`: files in `--empty-bg-dir` are sorted by name
and read one per input frame. The first file must align with the first video frame;
`--recent-bg` supplies the recent reference for initialization. A missing per-frame
reference raises an error. Prepared-reference options cannot be mixed with the
automatic-estimation options.

## Demo

The order of the first row, from left to right, is as follows: `input`, `label`, `prediction mask(threshold=0.5)`, and `prediction probability`.
The second row displays the sequential features' visual results.

- PTZ/intermittenPan
  link: <https://drive.google.com/file/d/12-dvboDZkgFxo4dM-YxM1wBMwJJWCbB1>

<div style="position: relative; padding-bottom: 20%; height: 0; overflow: hidden;">
  <iframe src="https://drive.google.com/file/d/12-dvboDZkgFxo4dM-YxM1wBMwJJWCbB1/preview" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen></iframe>
</div>
<!-- <iframe src="https://drive.google.com/file/d/12-dvboDZkgFxo4dM-YxM1wBMwJJWCbB1/preview"></iframe> -->

| No. Frame | Sequential Features                                   |
| --------- | ----------------------------------------------------- |
| 1         | ![intermittenPan-f1](./figures/intermittenPan-f1.png) |
| 2         | ![intermittenPan-f2](./figures/intermittenPan-f2.png) |
| 3         | ![intermittenPan-f3](./figures/intermittenPan-f3.png) |

From Frame 1 (initialize) to Frame 3, it is obvious that the sequential features can be adapted quickly.

- shadow/peopleInShade
  link: <https://drive.google.com/file/d/1WbR_CjdeGmQn2NpPwPVL57v_eL2RmEK1/view?usp=drive_link>

<div style="position: relative; padding-bottom: 20%; height: 0; overflow: hidden;">
  <iframe src="https://drive.google.com/file/d/1WbR_CjdeGmQn2NpPwPVL57v_eL2RmEK1/preview" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen></iframe>
</div>

<!-- <iframe src="https://drive.google.com/file/d/1WbR_CjdeGmQn2NpPwPVL57v_eL2RmEK1/preview"></iframe> -->

| Judgment of shadow | Results                                                 |
| ------------------ | ------------------------------------------------------- |
| As FG              | ![peopleInShade-miss](./figures/peopleInShade-miss.png) |
| As BG              | ![peopleInShade-hit](./figures/peopleInShade-hit.png)   |

The result shows that the sequential features can divide the shadow shape to prevent misjudgment of the shadow as foreground.

## Citation
```bibtex
@InProceedings{SeqFeedNet,
    author    = {Huang, Yu-Shun and Guo, Jing-Ming and Yang, Yi-Xiang},
    title     = {SeqFeedNet: Sequential Feature Feedback Network for Background Subtraction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {8199-8208}
}
```
