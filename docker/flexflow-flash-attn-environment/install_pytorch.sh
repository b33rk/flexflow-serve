#!/bin/bash

# Check if CUDA and Python versions are supplied
if [ -z "$1" ]; then
  echo "Please provide the CUDA version as XX.Y (e.g., 12.1)"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Please provide the Python version (e.g., 3.8, 3.9, 3.10, 3.11, 3.12, or latest)"
  exit 1
fi

CUDA_VERSION=$1
PYTHON_VERSION=$2
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d '.' -f 1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d '.' -f 2)

# Activate conda base
source /opt/conda/bin/activate base

# Ensure we are using the correct Python version
if [ "$PYTHON_VERSION" != "latest" ]; then
  conda install -y python=${PYTHON_VERSION}
fi

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA ${CUDA_MAJOR}.${CUDA_MINOR} for Python ${PYTHON_VERSION}..."
pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}

# Verify installation
python3 -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

if [ $? -ne 0 ]; then
  echo "PyTorch installation failed."
  exit 1
else
  echo "PyTorch successfully installed with CUDA ${CUDA_MAJOR}.${CUDA_MINOR} for Python ${PYTHON_VERSION}."
fi