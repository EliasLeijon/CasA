FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       zlib1g-dev \
                       vim \
                       python3 \
                       python3-pip \
                       ninja-build

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install Open3D system dependencies https://www.open3d.org/docs/release/docker.html
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# We need this for train.py
# apt-get install libglib2.0-0
# Answer 8 49

RUN mkdir /workspace
WORKDIR '/workspace'

RUN mkdir /dataset
# TODO: Maybe remove git install instead?
RUN git config --global --add safe.directory /workspace


ENTRYPOINT ["/bin/bash"]