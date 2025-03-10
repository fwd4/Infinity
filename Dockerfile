ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
ARG CUDNN=cudnn

FROM nvcr.io/nvidia/cuda:{CUDA_VERSION}-{CUDNN}-devel-ubuntu{UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="$CUDA_HOME/bin:/opt/conda/bin:$PATH"

RUN sed -i 's/http:\/\/archive.ubuntu.com/https:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-tk \
    libgl1 \
    && apt-get clean #&& rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && pip install --upgrade pip \
    && pip install ninja nvitop packaging \
    && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    && pip install tensorflow==2.15 \
                   pytorch_fid \
                   easydict \
                   typed-argument-parser \
                   seaborn \
                   kornia \
                   gputil \
                   colorama \
                   omegaconf \
                   pandas \
                   timm==0.9.6 \
                   decord \
                   transformers \
                   pytz \
                   pandas \
                   wandb \
                   colorama \
                   imageio \
                   einops \
                   openai==1.34.0 \
                   httpx==0.20.0 \
                   opencv-python \
                   opencv-fixer==0.2.5 \
                   datasets \
                   openmim \
                   mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0 \
                   diffusers \
                   image-reward \
                   hpsv2 \
    && MAX_JOBS=8 pip install flash-attn --no-build-isolation
    && mim install mmengine mmcv-full==1.7.2


    # PYSITE="/usr/local/lib/python3.10/dist-packages"
    # mv bpe_simple_vocab_16e6.txt.gz $PYSITE/hpsv2/src/open_clip


RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

CMD ["/bin/bash"]
