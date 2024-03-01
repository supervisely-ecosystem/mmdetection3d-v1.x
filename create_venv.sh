#!/bin/bash

# Checking if .venv dir already exists.
if [ -d ".venv" ]; then
    echo "VENV dir (.venv) already exists, it will be removed."
    rm -rf .venv
fi

echo "VENV will be created"

# Checking if python3.8 is available in PATH.
if command -v python3.8 &>/dev/null; then
    python_executable="python3.8" && \
    echo "Python 3.8 found, it will be used for creating VENV dir."
else
    python_executable="python3" && \
    echo "Python 3.8 not found, default python3 will be used for creating VENV dir."
fi

# Creating VENV dir with selected python executable.
$python_executable -m venv .venv && \
source .venv/bin/activate && \

# Installing requirements from requirements.txt.
echo "Install requirements..." && \
pip3 install -r dev_requirements.txt && \
mim install mmengine 'mmcv>=2.0.0rc4' 'mmdet>=3.0.0' 'mmdet3d>=1.1.0' && \
echo "Installing supervisely." && \
pip3 install supervisely==6.73.39 && \
echo "Requirements have been successfully installed, VENV ready." && \
deactivate
