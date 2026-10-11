FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# The original requirements.txt omits matplotlib, which the project imports.
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt matplotlib==3.8.2 \
    && python -c "import cv2, matplotlib, numpy, pandas, torch, torchvision; from torch.utils.tensorboard import SummaryWriter"

WORKDIR /workspace
CMD ["bash"]
