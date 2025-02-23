#! /usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version, hip_version, ATTACH_GPUS, SHM_SIZE

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

# Parameter controlling whether to attach GPUs to the Docker container
ATTACH_GPUS=${ATTACH_GPUS:-true}
gpu_arg=""
if $ATTACH_GPUS ; then gpu_arg="--gpus all" ; fi
FORWARD_STREAMLIT_PORT=${FORWARD_STREAMLIT_PORT:-false}
port_forward_arg=""
if $FORWARD_STREAMLIT_PORT ; then
  port_forward_arg+="-p 8501:8501"
fi


# Amount of shared memory to give the Docker container access to
# If you get a Bus Error, increase this value. If you don't have enough memory
# on your machine, decrease this value.
SHM_SIZE=${SHM_SIZE:-8192m}

# Check docker image name
if [[ "$image" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment', 'flexflow'."
  exit 1
fi

# Check GPU backend
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Running $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Running $image docker image with default GPU backend: cuda"
fi

# gpu backend version suffix for the docker image.
gpu_backend_version=""

if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  set_cuda_version_version || { echo "Failed to set gpu_backend_version." >&2; exit 1; }
  echo "Running $image docker image with CUDA: $cuda_version"
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
  echo "Running $image docker image with HIP $hip_version"
  if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    gpu_backend_version="-${hip_version}"
  fi
fi

# Check that image exists, if fails, print the default error message.
if [[ "$(docker images -q "${image}-${FF_GPU_BACKEND}${gpu_backend_version}":latest 2> /dev/null)" == "" ]]; then
  echo "Error, ${image}-${FF_GPU_BACKEND}${gpu_backend_version}:latest does not exist!"
  if [[ "${FF_GPU_BACKEND}" == "cuda" ]]; then
    echo ""
    echo "To download the docker image, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} cuda_version=${cuda_version} $(pwd)/pull.sh $image"
    echo "To build the docker image from source, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} cuda_version=${cuda_version} $(pwd)/build.sh $image"
    echo ""
  elif [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    echo ""
    echo "To download the docker image, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} hip_version=${hip_version} $(pwd)/pull.sh $image"
    echo "To build the docker image from source, run:"
    echo "    FF_GPU_BACKEND=${FF_GPU_BACKEND} hip_version=${hip_version} $(pwd)/build.sh $image"
    echo ""
  fi
  exit 1
fi

hf_token_volume=""
hf_token_path="$HOME/.cache/huggingface/token"
if [ -f "$hf_token_path" ]; then
  # If the token exists, add the volume mount to the Docker command
  hf_token_volume+="-v $hf_token_path:/root/.cache/huggingface/token"
fi

ssh_key_volume=""
ssh_key_path="$HOME/.ssh/id_rsa"
if [ -f "$ssh_key_path" ] && [ -f "$ssh_key_path.pub" ]; then
  ssh_key_volume="-v $ssh_key_path:/root/.ssh/id_rsa -v $ssh_key_path.pub:/root/.ssh/id_rsa.pub"
fi
eval docker run -it "$gpu_arg" "--shm-size=${SHM_SIZE}" "--cap-add=SYS_PTRACE" "${ssh_key_volume}" "${hf_token_volume}" "${port_forward_arg}" "${image}-${FF_GPU_BACKEND}${gpu_backend_version}:latest"
