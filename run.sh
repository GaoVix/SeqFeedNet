# docker build -t seqfeednet .

docker run --rm -it --gpus all \
  -v /home/weixiang:/wx \
  -v /home/Datasets0:/data0 \
  -v /home/Datasets1:/data1 \
  -v /home/Datasets2:/data2 \
  -v "$PWD:/workspace" seqfeednet

# cd /workspace &&
# python -m pip install -U gdown &&
# mkdir -p checkpoints &&
# gdown 1U5xHNw9m00uepnQ-OQsPZAWHIaElZVBJ -O checkpoints/cv3.pt.download &&
# python -c "import torch; torch.load('checkpoints/cv3.pt.download', map_location='cpu', weights_only=True); print('权重校验通过')" &&
# mv checkpoints/cv3.pt.download checkpoints/cv3.pt

# python predict_video.py \
#   --video /data1/droid_raw/1.0.1/ILIAD/success/2023-05-07/Sun_May__7_16:54:37_2023/recordings/MP4/23897859.mp4 \
#   --weights checkpoints/cv3.pt \
#   --device 0 \
#   --overwrite \
#   --output-dir /wx/Align_Camera/output/bgs