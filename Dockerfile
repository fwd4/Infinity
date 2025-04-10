FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="$CUDA_HOME/bin:$PATH"

RUN sed -i 's/http:\/\/archive.ubuntu.com/https:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean #&& rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

COPY requirements.txt /workspace/requirements.txt

RUN apt update && apt install -y python3-pip \
    && apt install -y python3-tk \
    && apt install -y libgl1

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && pip install --upgrade pip \
    && pip install ninja nvitop packaging \
    && pip install torch==2.5.1

RUN pip install -r requirements.txt

RUN pip install image-reward pytorch_lightning \
    && pip install timm==0.6.13  \
    && pip install openai==1.34.0  \
    && pip install httpx==0.20.0  \
    && pip install diffusers==0.16.0 \
    && pip install hpsv2 \
    && pip install -U openmim \
    && mim install mmengine mmcv-full==1.7.2 \
    && pip install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0 \
    && pip install openai \
    && pip install httpx==0.20.0 \
    && pip install pytorch_fid 

RUN MAX_JOBS=8 pip install flash-attn --no-build-isolation \
    && pip install opencv-fixer==0.2.5


RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"


CMD ["/bin/bash"]
