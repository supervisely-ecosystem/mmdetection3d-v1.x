FROM supervisely/base-py-sdk:6.72.199

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA="1"

# Install required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# MMDetection3D and dependencies
RUN pip install -U openmim
RUN mim install mmengine 'mmcv>=2.0.0rc4' 'mmdet>=3.0.0' 'mmdet3d>=1.1.0'

# To convert .pcd to .bin
RUN pip install git+https://github.com/DanielPollithy/pypcd.git

# Boost GPU perfomance (optional)
RUN pip install spconv-cu118 cumm-cu118

# Fix issue with open3d
RUN pip install Werkzeug==2.2.3

RUN pip install numpy==1.22.0

# Update Supervisely
RUN pip install -U supervisely==6.73.292
