FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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

RUN apt update && apt install -y python3-pip

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && pip install --upgrade pip \
    && pip install ninja nvitop packaging \
    && pip install torch==2.5.1

RUN pip install -r requirements.txt

RUN MAX_JOBS=8 pip install flash-attn --no-build-isolation \
    && pip install opencv-fixer==0.2.5

RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"

RUN pip install datasets

CMD ["/bin/bash"]
