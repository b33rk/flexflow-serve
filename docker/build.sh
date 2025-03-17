#! /usr/bin/env bash
set -euo pipefail

# Usage: ./build.sh <docker_image_name>
# Optional environment variables: FF_GPU_BACKEND, cuda_version, hip_version

get_cuda_docker_image() {
  local docker_user="nvidia"
  local docker_image="cuda"
  local page=1
  local per_page=100

  # Determine Ubuntu version: use lsb_release if available, else default to 22.04.
  local ubuntu_version
  if command -v lsb_release >/dev/null 2>&1; then
      ubuntu_version=$(lsb_release -rs)
  else
      ubuntu_version="22.04"
  fi

  # Determine CUDA version.
  # If the environment variable 'cuda_version' is set (in "<major>.<minor>" format), use that.
  # Otherwise, use nvidia-smi to extract the CUDA version.
  local cuda_full_version
  local installed_major_minor
  if [[ -n "${cuda_version:-}" ]]; then
      cuda_full_version="$cuda_version"
      installed_major_minor="$cuda_version"
  else
      if ! command -v nvidia-smi >/dev/null 2>&1; then
          echo "Error: nvidia-smi not found and cuda_version is not set." >&2
          return 1
      fi
      local nvidia_smi_output
      nvidia_smi_output=$(nvidia-smi)
      local cuda_version_line
      cuda_version_line=$(echo "$nvidia_smi_output" | grep "CUDA Version")
      cuda_full_version=$(echo "$cuda_version_line" | sed -n 's/.*CUDA Version:\s*\([0-9]\+\.[0-9]\+\).*/\1/p')
      if [[ -z "$cuda_full_version" ]]; then
          echo "Error: Unable to determine CUDA version from nvidia-smi." >&2
          return 1
      fi
      installed_major_minor="$cuda_full_version"
  fi

  # Query Docker Hub for matching tags.
  local -a tags_list=()
  while true; do
      local response new_tags
      response=$(curl -s "https://hub.docker.com/v2/repositories/${docker_user}/${docker_image}/tags?page=${page}&page_size=${per_page}")
      new_tags=$(echo "$response" | jq -r --arg v "$ubuntu_version" '.results[].name | select(contains("cudnn") and contains("devel-ubuntu") and test("ubuntu"+$v+"$"))')
      if [[ -z "$new_tags" ]]; then
          break
      fi
      while read -r tag; do
          tags_list+=("$tag")
      done <<< "$new_tags"
      ((page++))
  done

  if [ ${#tags_list[@]} -eq 0 ]; then
      echo "Error: No docker images found matching criteria." >&2
      return 1
  fi

  # Sort the tags in descending order based on the CUDA version.
  local sorted_tags
  sorted_tags=$(printf "%s\n" "${tags_list[@]}" | sort -rV -t '-' -k1,1)

  # Find the most appropriate tag.
  local selected_tag=""
  while read -r tag; do
      local version tag_major_minor
      version=$(echo "$tag" | cut -d '-' -f1)
      tag_major_minor=$(echo "$version" | awk -F. '{print $1"."$2}')
      if [[ "$tag_major_minor" == "$installed_major_minor" ]]; then
          selected_tag="$tag"
          break
      fi
  done <<< "$sorted_tags"

  # If no exact match, choose the highest version lower than the installed version.
  if [[ -z "$selected_tag" ]]; then
      while read -r tag; do
          local version
          version=$(echo "$tag" | cut -d '-' -f1)
          if [[ $(printf '%s\n' "$version" "$cuda_full_version" | sort -V | head -n1) == "$version" && "$version" != "$cuda_full_version" ]]; then
              selected_tag="$tag"
              break
          fi
      done <<< "$sorted_tags"
  fi

  if [[ -n "$selected_tag" ]]; then
      echo "${docker_user}/${docker_image}:${selected_tag}"
      return 0
  else
      echo "Error: No suitable docker image found." >&2
      return 1
  fi
}

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

# Cd into flexflow-serve. Assumes this script is in flexflow-serve/docker
cd "${BASH_SOURCE[0]%/*}/.."

# Parse input params
image=${1:-flexflow}
FF_GPU_BACKEND=${FF_GPU_BACKEND:-cuda}
hip_version=${hip_version:-"empty"}
python_version=${python_version:-latest}

# Check docker image name
if [[ "$image" != @(flexflow-environment|flexflow) ]]; then
  echo "Error, image name ${image} is invalid. Choose between 'flexflow-environment' and 'flexflow'."
  exit 1
fi

# Check GPU backend
if [[ "${FF_GPU_BACKEND}" != @(cuda|hip_cuda|hip_rocm|intel) ]]; then
  echo "Error, value of FF_GPU_BACKEND (${FF_GPU_BACKEND}) is invalid. Pick between 'cuda', 'hip_cuda', 'hip_rocm' or 'intel'."
  exit 1
elif [[ "${FF_GPU_BACKEND}" != "cuda" ]]; then
  echo "Building $image docker image with gpu backend: ${FF_GPU_BACKEND}"
else
  echo "Building $image docker image with default GPU backend: cuda"
fi

# base image to use when building the flexflow environment docker image.
ff_environment_base_image="ubuntu:20.04"
# gpu backend version suffix for the docker image.
gpu_backend_version=""

if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  ff_environment_base_image=$(get_cuda_docker_image) || { echo "Failed to get docker image." >&2; exit 1; }
  echo "Using base docker image: $ff_environment_base_image"
  set_cuda_version_version || { echo "Failed to set gpu_backend_version." >&2; exit 1; }
  echo "GPU Backend Version is set to: $cuda_version"
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
  echo "Building $image docker image with HIP $hip_version"
  if [[ "${FF_GPU_BACKEND}" == "hip_rocm" ]]; then
    gpu_backend_version="-${hip_version}"
  fi
  cuda_version="empty"
fi

# Get number of cores available on the machine. Build with all cores but one, to prevent RAM choking
cores_available=$(nproc --all)
n_build_cores=$(( cores_available -1 ))

# check python_version
if [[ "$python_version" != @(3.8|3.9|3.10|3.11|3.12|latest) ]]; then
  echo "python_version not supported!"
  exit 0
fi

docker build \
  --build-arg "ff_environment_base_image=${ff_environment_base_image}" \
  --build-arg "N_BUILD_CORES=${n_build_cores}" \
  --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" \
  --build-arg "cuda_version=${cuda_version}" \
  --build-arg "hip_version=${hip_version}" \
  --build-arg "python_version=${python_version}" \
  -t "flexflow-environment-${FF_GPU_BACKEND}${gpu_backend_version}" \
  -f docker/flexflow-environment/Dockerfile \
  .

# If the user only wants to build the environment image, we are done
if [[ "$image" == "flexflow-environment" ]]; then
  exit 0
fi

# Done with flexflow-environment image

###########################################################################################

# Build flexflow image if requested 
if [[ "${FF_GPU_BACKEND}" == "cuda" || "${FF_GPU_BACKEND}" == "hip_cuda" ]]; then
  # If FF_CUDA_ARCH is set to autodetect, we need to perform the autodetection here because the Docker
  # image will not have access to GPUs during the build phase (due to a Docker restriction). In all other
  # cases, we pass the value of FF_CUDA_ARCH directly to Cmake.
  if [[ "${FF_CUDA_ARCH:-autodetect}" == "autodetect" ]]; then
    # Get CUDA architecture(s), if GPUs are available
    cat << EOF > ./get_gpu_arch.cu
#include <stdio.h>
int main() {
  int count = 0;
  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
  if (count == 0) return -1;
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop;
    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
      printf("%d ", prop.major*10+prop.minor);
  }
  return 0;
}
EOF
    gpu_arch_codes=""
    if command -v nvcc &> /dev/null
    then
      nvcc ./get_gpu_arch.cu -o ./get_gpu_arch
      gpu_arch_codes="$(./get_gpu_arch)"
    fi
    gpu_arch_codes="$(echo "$gpu_arch_codes" | xargs -n1 | sort -u | xargs)"
    gpu_arch_codes="${gpu_arch_codes// /,}"
    rm -f ./get_gpu_arch.cu ./get_gpu_arch

    if [[ -n "$gpu_arch_codes" ]]; then
    echo "Host machine has GPUs with architecture codes: $gpu_arch_codes"
    echo "Configuring FlexFlow to build for the $gpu_arch_codes code(s)."
    FF_CUDA_ARCH="${gpu_arch_codes}"
    export FF_CUDA_ARCH
    else
      echo "FF_CUDA_ARCH is set to 'autodetect', but the host machine does not have any compatible GPUs."
      exit 1
    fi
  fi
fi

# Build FlexFlow Docker image
# shellcheck source=/dev/null
. config/config.linux get-docker-configs
# Set value of BUILD_CONFIGS
get_build_configs

docker build \
  --build-arg "N_BUILD_CORES=${n_build_cores}" \
  --build-arg "FF_GPU_BACKEND=${FF_GPU_BACKEND}" \
  --build-arg "BUILD_CONFIGS=${BUILD_CONFIGS}" \
  --build-arg "gpu_backend_version=${gpu_backend_version}" \
  -t "flexflow-${FF_GPU_BACKEND}${gpu_backend_version}" \
  -f docker/flexflow/Dockerfile \
  .
