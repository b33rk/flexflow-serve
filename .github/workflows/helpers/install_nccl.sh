#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

ubuntu_version=$(lsb_release -rs)
ubuntu_version=${ubuntu_version//./}
wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_version}/x86_64/cuda-keyring_1.0-1_all.deb"
DEBIAN_FRONTEND=noninteractive dpkg -i cuda-keyring_1.0-1_all.deb
DEBIAN_FRONTEND=noninteractive apt-get update -y --allow-change-held-packages
rm -f cuda-keyring_1.0-1_all.deb
DEBIAN_FRONTEND=noninteractive apt install -y --allow-change-held-packages libnccl2 libnccl-dev
