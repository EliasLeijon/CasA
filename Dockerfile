FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       zlib1g-dev \
                       vim \
                       python3 \
                       python3-pip \
                       ninja-build

RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN mkdir /workspace

WORKDIR '/workspace'
RUN mkdir /dataset
COPY . .
# git config --global --add safe.directory /workspace
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install zod
RUN pip install prefetch_generator

ENTRYPOINT ["/bin/bash"]