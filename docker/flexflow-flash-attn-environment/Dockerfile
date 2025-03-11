ARG ff_environment_base_image=flexflow-environment-cuda-12.1
FROM ${ff_environment_base_image}

LABEL org.opencontainers.image.source=https://github.com/flexflow/flexflow-serve
LABEL org.opencontainers.image.description="flexflow-serve environment container"

SHELL ["/bin/bash", "-c"]

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget sudo binutils git zlib1g-dev lsb-release nano gdb libhdf5-dev jq openssh-client \
    software-properties-common build-essential apt-utils ca-certificates libssl-dev curl unzip htop ninja-build && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    apt-get upgrade -y libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python3 with Miniconda
ARG python_version="latest"
RUN rm -rf /opt/conda && \
    MINICONDA_SCRIPT_NAME="Miniconda3-latest-Linux-x86_64.sh"; \
    if [ "${python_version}" = "3.8" ]; then \
        MINICONDA_SCRIPT_NAME="Miniconda3-py38_23.5.2-0-Linux-x86_64.sh"; \
    elif [ "${python_version}" = "3.9" ]; then \
        MINICONDA_SCRIPT_NAME="Miniconda3-py39_23.5.2-0-Linux-x86_64.sh"; \
    elif [ "${python_version}" = "3.10" ]; then \
        MINICONDA_SCRIPT_NAME="Miniconda3-py310_23.5.2-0-Linux-x86_64.sh"; \
    elif [ "${python_version}" = "3.11" ]; then \
        MINICONDA_SCRIPT_NAME="Miniconda3-py311_23.5.2-0-Linux-x86_64.sh"; \
    elif [ "${python_version}" = "3.12" ]; then \
        MINICONDA_SCRIPT_NAME="Miniconda3-py312_25.1.1-2-Linux-x86_64.sh"; \
    fi; \
    wget -c -q https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT_NAME} && \
    bash ${MINICONDA_SCRIPT_NAME} -b -p /opt/conda && \
    rm ${MINICONDA_SCRIPT_NAME} && \
    /opt/conda/bin/conda config --set solver classic && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

# Set environment variables
ENV PATH="/opt/conda/bin:${PATH}"
ENV CUDNN_DIR="/usr/local/cuda"
ENV CUDA_DIR="/usr/local/cuda"
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set up parallel build flags
ARG N_BUILD_CORES
ENV MAKEFLAGS="-j${N_BUILD_CORES}"

# GPU-specific dependencies
ARG FF_GPU_BACKEND="cuda"
ARG cuda_version="12.1"
ARG hip_version="5.6"

# Install hip dependencies if FF_GPU_BACKEND is hip_cuda or hip_rocm
# Note that amd's docs say to also install the `hip-runtime-nvidia` package. This
# package attempts to re-install cuda even though cuda is already installed
# in the container. It also attempts to install packages for a graphical install.
# For our container, we don't need `hip-runtime-nvidia`
RUN  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
        # Check that hip_version is one of 5.3,5.4,5.5,5.6
        if [ "$hip_version" != "5.3" ] && [ "$hip_version" != "5.4" ] && [ "$hip_version" != "5.5" ] && [ "$hip_version" != "5.6" ]; then \
            echo "hip_version '${hip_version}' is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"; \
            exit 1; \
        fi; \
        # Compute script name and url given the version
        AMD_GPU_SCRIPT_NAME=amdgpu-install_5.6.50600-1_all.deb; \
        if [ "$hip_version" = "5.3" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.3.50300-1_all.deb; \
        elif [ "$hip_version" = "5.4" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.4.50400-1_all.deb; \
        elif [ "$hip_version" = "5.5" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.5.50500-1_all.deb; \
        fi; \
        ubuntu_codename=$(lsb_release -cs); \
        AMD_GPU_SCRIPT_URL="https://repo.radeon.com/amdgpu-install/${hip_version}/ubuntu/${ubuntu_codename}/${AMD_GPU_SCRIPT_NAME}"; \
        # Download and install AMD GPU software with ROCM and HIP support
        wget $AMD_GPU_SCRIPT_URL; \
        apt-get install -y ./${AMD_GPU_SCRIPT_NAME}; \
        rm ./${AMD_GPU_SCRIPT_NAME}; \
        amdgpu-install -y --usecase=hip,rocm --no-dkms; \
        apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk rocm-device-libs; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
    fi
RUN rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN conda install -c conda-forge cmake make cmake-build-extension numpy pandas

# Install PyTorch (CUDA Version)
ARG cuda_version="12.1"
COPY install_pytorch.sh /usr/local/bin/install_pytorch.sh
# RUN chmod +x /usr/local/bin/install_pytorch.sh && \
#     /usr/local/bin/install_pytorch.sh ${cuda_version} ${python_version} && \
#     rm /usr/local/bin/install_pytorch.sh

RUN if [ "$FF_GPU_BACKEND" == "cuda" ] ; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing PyTorch with CUDA"; \
        chmod +x /usr/local/bin/install_pytorch.sh && \
        /usr/local/bin/install_pytorch.sh ${cuda_version} ${python_version}; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing CPU-only PyTorch"; \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi
RUN rm /usr/local/bin/install_pytorch.sh

# Install other dependencies
RUN pip3 install transformers>=4.47.1 sentencepiece einops
RUN pip3 install tensorflow notebook
RUN pip3 install scipy bitsandbytes datasets accelerate loralib triton peft pytest streamlit

# Install Rust
RUN rm -rf /root/.cargo /root/.rustup && \
    mkdir -p /root/.cargo /root/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable && \
    chmod -R a+w /root/.rustup /root/.cargo

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_HOME="/root/.rustup"
ENV CARGO_HOME="/root/.cargo"

# Set up CMake environment variables with torch with $python_version
ENV CMAKE_PREFIX_PATH="/opt/conda/lib/python${python_version}/site-packages/torch"

# Set up the correct LD_LIBRARY_PATH for PyTorch
ENV LD_LIBRARY_PATH=/opt/conda/lib/python${python_version}/site-packages/torch/lib:$LD_LIBRARY_PATH

# Ensure CUDA paths are correctly set
RUN echo "CUDA_HOME=${CUDA_HOME}" >> ~/.bashrc && \
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.bashrc

# Install flash-attn
# RUN pip install --no-build-isolation flash-attn --no-cache-dir

# Install flash-attn of correct cuda version, torch version, and python version and right abi
# RUN wget -O /tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
#     https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
#     pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-build-isolation && \
#     rm -rf /tmp/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install FlashAttention of the correct CUDA, Torch, and Python version dynamically
RUN python -c "import torch; print(f'PyTorch Version: {torch.__version__}')" && \
    python -c "import sys; print(f'Python Version: {sys.version_info.major}.{sys.version_info.minor}')" && \
    python -c "import torch; print(f'CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')" && \
    export PYTHON_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") && \
    export PYTORCH_VERSION=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))") && \
    export TORCH_VERSION_STRIPPED=$(python -c "import torch; print('.'.join(torch.__version__.replace('+cu', '').split('.')[:2]))") && \
    export CXX_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    export CUDA_MAJOR_MINOR="12" && \
    export FLASH_ATTN_WHL="flash_attn-2.7.4.post1+cu${CUDA_MAJOR_MINOR}torch${TORCH_VERSION_STRIPPED}cxx11abi${CXX_ABI}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl" && \
    export FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHL}" && \
    echo "Downloading FlashAttention from: ${FLASH_ATTN_URL}" && \
    curl -fsSL -o /tmp/${FLASH_ATTN_WHL} --write-out "%{http_code}" ${FLASH_ATTN_URL} | tee /dev/stderr | grep -q "200" && \
    pip install /tmp/${FLASH_ATTN_WHL} --no-build-isolation && \
    rm -rf /tmp/${FLASH_ATTN_WHL}

ENTRYPOINT ["/bin/bash"]