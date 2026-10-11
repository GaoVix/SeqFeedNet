#!/usr/bin/env bash
set -u

DATA_ROOT=/data1/droid_raw/1.0.1
OUTPUT_ROOT=/data2/droid_sfn

[[ -d "$DATA_ROOT" && -f checkpoints/cv3.pt ]] || {
  echo "请检查数据目录、权重文件及当前工作目录"
  exit 1
}

find "$DATA_ROOT" -type f \
  -path '*/recordings/MP4/*.mp4' \
  ! -name '*-stereo.mp4' -print0 |
while IFS= read -r -d '' video; do
  relative="${video#"$DATA_ROOT"/}"
  output="$OUTPUT_ROOT/${relative%.mp4}"

  # 成功完成的跳过；中断或失败的下次重跑
  if [[ -f "$output/.complete" &&
        -s "$output/masks/000000.png" &&
        -s "$output/overlay.mp4" ]]; then
    continue
  fi

  mkdir -p "$output" || exit 1
  rm -f "$output/.complete"
  echo "正在处理：$relative"

  if python predict_video.py \
      --video "$video" \
      --weights checkpoints/cv3.pt \
      --device 2 \
      --overwrite \
      --output-dir "$output" \
      >"$output/run.log" 2>&1; then
    touch "$output/.complete"
  else
    echo "处理失败，查看：$output/run.log" >&2
  fi
done
