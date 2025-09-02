#! /usr/bin/env bash
set -euo pipefail

# Usage: ./pull.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version, hip_version

set_cuda_version_version() {
  # If the user provided a cuda_version, use that.
  if [[ -n "${cuda_version:-}" ]]; then
    return 0
  fi

  # Otherwise, check that nvidia-smi is available.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found and cuda_version is not set." >&2
    return 1
  fi

  # Extract the CUDA version from nvidia-smi output.
  local nvidia_output cuda_ver
  nvidia_output=$(nvidia-smi)
  cuda_ver=$(echo "$nvidia_output" | grep "CUDA Version" | sed -n 's/.*CUDA Version:\s*\([0-9]\+\.[0-9]\+\).*/\1/p')
  
  if [[ -z "$cuda_ver" ]]; then
    echo "Error: Unable to detect CUDA version from nvidia-smi." >&2
    return 1
  fi

  export cuda_version="$cuda_ver"
  return 0
}


# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
hip_version=${hip_version:-"empty"}

# Check docker image name
if [[ "${image}" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, docker image name '${image}' is invalid. Choose between 'flexflow-environment' and 'flexflow'."
  exit 1
fi

# Check GPU backend
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Downloading $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Downloading $image docker image with default GPU backend: cuda"
fi

# gpu backend version suffix for the docker image.
gpu_backend_version=""

if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  set_cuda_version_version || { echo "Failed to set gpu_backend_version." >&2; exit 1; }
  # Check that CUDA version is supported
  if [[ "$cuda_version" != @(11.1|11.6|11.7|11.8|12.0|12.1|12.2|12.3|12.4|12.5|12.6|12.7|12.8) ]]; then
    echo "cuda_version is not available for download, please choose among {11.1|11.6|11.7|11.8|12.0|12.1|12.2|12.3|12.4|12.5|12.6|12.7|12.8}"
    exit 1
  fi
  # Set cuda version suffix to docker image name
  echo "Downloading $image docker image with CUDA $cuda_version"
  gpu_backend_version="-${cuda_version}"
fi

if [[ "${FF_GPU_BACKEND}" == "hip_rocm" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  # Autodetect HIP version if not specified
  if [[ $hip_version == "empty" ]]; then
    # shellcheck disable=SC2015
    hip_version=$(command -v hipcc >/dev/null 2>&1 && hipcc --version | grep "HIP version:" | awk '{print $NF}' || true)
    # Change hip_version eg. 5.6.31061-8c743ae5d to 5.6
    hip_version=${hip_version:0:3}
    if [[ -z "$hip_version" ]]; then
      echo "Could not detect HIP version. Please specify one manually by setting the 'hip_version' env."
      exit 1
    fi
  fi
  # Check that HIP version is supported
  if [[ "$hip_version" != @(5.3|5.4|5.5|5.6) ]]; then
    echo "hip_version is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"
    exit 1
  fi
  echo "Downloading $image docker image with HIP $hip_version"
  if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    gpu_backend_version="-${hip_version}"
  fi
fi

# Download image
docker pull ghcr.io/flexflow/"$image-${FF_GPU_BACKEND}${gpu_backend_version}"

# Tag downloaded image
docker tag ghcr.io/flexflow/"$image-${FF_GPU_BACKEND}${gpu_backend_version}":latest "$image-${FF_GPU_BACKEND}${gpu_backend_version}":latest 

# Check that image exists
docker image inspect "${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest > /dev/null
