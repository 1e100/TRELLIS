#!/bin/bash

set -euo pipefail

# Configuration
VENV_DIR=".venv"
VENV_PYTHON="$VENV_DIR/bin/python"

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
LINEAR_ASSIGNMENT=false
SPCONV=false
ERROR=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false
DEMO=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new uv virtual environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install all dependencies for demo"
    exit 0
fi

# Create/Re-create virtual environment if requested
if [ "$NEW_ENV" = true ] ; then
    echo "[ENV] Creating new virtual environment..."
    if [ -d "$VENV_DIR" ]; then
        echo "[ENV] Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    uv venv --python 3.10 "$VENV_DIR"
    
    # Install PyTorch
    echo "[SYSTEM] Installing PyTorch..."
    uv pip install --python "$VENV_PYTHON" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
fi

# Verify virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_DIR. Run with --new-env first."
    exit 1
fi

# Get system information using venv's Python
WORKDIR=$(pwd)
PYTORCH_VERSION=$($VENV_PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -z "$PYTORCH_VERSION" ]; then
    echo "[ERROR] Failed to import PyTorch. Make sure --new-env was run successfully."
    exit 1
fi

PLATFORM=$($VENV_PYTHON -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
case $PLATFORM in
    cuda)
        CUDA_VERSION=$($VENV_PYTHON -c "import torch; print(torch.version.cuda)")
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    hip)
        HIP_VERSION=$($VENV_PYTHON -c "import torch; print(torch.version.hip)")
        HIP_MAJOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f1)
        HIP_MINOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f2)
        # Install pytorch 2.4.1 for hip
        if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ] ; then
            echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
            uv pip install --python "$VENV_PYTHON" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1
            mkdir -p /tmp/extensions
            sudo cp /opt/rocm/share/amd_smi /tmp/extensions/amd_smi -r
            cd /tmp/extensions/amd_smi
            sudo chmod -R 777 .
            uv pip install --python "$VENV_PYTHON" .
            cd $WORKDIR
            PYTORCH_VERSION=$($VENV_PYTHON -c "import torch; print(torch.__version__)")
        fi
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, HIP Version: $HIP_VERSION"
        ;;
    *)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, Platform: $PLATFORM"
        ;;
esac

# Install basic dependencies
if [ "$BASIC" = true ] ; then
    echo "[BASIC] Installing basic dependencies..."
    uv pip install --python "$VENV_PYTHON" pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
    uv pip install --python "$VENV_PYTHON" git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

# Install training dependencies
if [ "$TRAIN" = true ] ; then
    echo "[TRAIN] Installing training dependencies..."
    uv pip install --python "$VENV_PYTHON" tensorboard pandas lpips
    uv pip uninstall --python "$VENV_PYTHON" -y pillow
    sudo apt install -y libjpeg-dev
    uv pip install --python "$VENV_PYTHON" pillow-simd
fi

# Install xformers
if [ "$XFORMERS" = true ] ; then
    echo "[XFORMERS] Installing xformers..."
    if [ "$PLATFORM" = "cuda" ] ; then
        if [ "$CUDA_VERSION" = "11.8" ] ; then
            case $PYTORCH_VERSION in
                2.0.1) uv pip install --python "$VENV_PYTHON" https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl  ;;
                2.1.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.1.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.1.2) uv pip install --python "$VENV_PYTHON" xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.2.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.2.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.2.2) uv pip install --python "$VENV_PYTHON" xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.3.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.4.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.4.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118  ;;
                2.5.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118  ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.1" ] ; then
            case $PYTORCH_VERSION in
                2.1.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.1.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.1.2) uv pip install --python "$VENV_PYTHON" xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.2.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.2.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.2.2) uv pip install --python "$VENV_PYTHON" xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.3.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.4.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.4.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121  ;;
                2.5.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121  ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.4" ] ; then
            case $PYTORCH_VERSION in
                2.5.0) uv pip install --python "$VENV_PYTHON" xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124  ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        else
            echo "[XFORMERS] Unsupported CUDA version: $CUDA_MAJOR_VERSION"
        fi
    elif [ "$PLATFORM" = "hip" ] ; then
        case $PYTORCH_VERSION in
            2.4.1\+rocm6.1) uv pip install --python "$VENV_PYTHON" xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1  ;;
            *) echo "[XFORMERS] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi

# Install flash-attn
if [ "$FLASHATTN" = true ] ; then
    echo "[FLASHATTN] Installing flash-attn..."
    if [ "$PLATFORM" = "cuda" ] ; then
        MAX_JOBS=16 uv pip install --python "$VENV_PYTHON" flash-attn==2.7.4.post1 --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git  /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.6.3-cktile
        UV_PYTHON="$VENV_PYTHON" GPU_ARCHS=gfx942 uv pip install --python "$VENV_PYTHON" .  #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

# Install kaolin
if [ "$KAOLIN" = true ] ; then
    echo "[KAOLIN] Installing kaolin..."
    if [ "$PLATFORM" = "cuda" ] ; then
        case $PYTORCH_VERSION in
            2.0.1) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html ;;
            2.1.0) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html ;;
            2.1.1) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html ;;
            2.2.0) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html ;;
            2.2.1) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html ;;
            2.2.2) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html ;;
            2.4.0) uv pip install --python "$VENV_PYTHON" kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html ;;
            *) echo "[KAOLIN] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[KAOLIN] Unsupported platform: $PLATFORM"
    fi
fi

# Install nvdiffrast
if [ "$NVDIFFRAST" = true ] ; then
    echo "[NVDIFFRAST] Installing nvdiffrast..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone https://github.com/NVlabs/nvdiffrast.git  /tmp/extensions/nvdiffrast
        uv pip install --python "$VENV_PYTHON" /tmp/extensions/nvdiffrast
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

# Install diffoctreerast
if [ "$DIFFOCTREERAST" = true ] ; then
    echo "[DIFFOCTREERAST] Installing diffoctreerast..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git  /tmp/extensions/diffoctreerast
        uv pip install --python "$VENV_PYTHON" /tmp/extensions/diffoctreerast --no-build-isolation
    else
        echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM"
    fi
fi

# Install mipgaussian
if [ "$MIPGAUSSIAN" = true ] ; then
    echo "[MIPGAUSSIAN] Installing mip-splatting..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone https://github.com/autonomousvision/mip-splatting.git  /tmp/extensions/mip-splatting
        uv pip install --python "$VENV_PYTHON" /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation
    else
        echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM"
    fi
fi

# Install vox2seq
if [ "$VOX2SEQ" = true ] ; then
    echo "[VOX2SEQ] Installing vox2seq..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        cp -r extensions/vox2seq /tmp/extensions/vox2seq
        uv pip install --python "$VENV_PYTHON" /tmp/extensions/vox2seq
    else
        echo "[VOX2SEQ] Unsupported platform: $PLATFORM"
    fi
fi

# Install spconv
if [ "$SPCONV" = true ] ; then
    echo "[SPCONV] Installing spconv..."
    if [ "$PLATFORM" = "cuda" ] ; then
        case $CUDA_MAJOR_VERSION in
            11) uv pip install --python "$VENV_PYTHON" spconv-cu118 ;;
            12) uv pip install --python "$VENV_PYTHON" spconv-cu120 ;;
            *) echo "[SPCONV] Unsupported PyTorch CUDA version: $CUDA_MAJOR_VERSION" ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM"
    fi
fi

# Install demo dependencies
if [ "$DEMO" = true ] ; then
    echo "[DEMO] Installing demo dependencies..."
    uv pip install --python "$VENV_PYTHON" gradio==4.44.1 gradio_litmodel3d==0.0.1
fi

echo "[DONE] Setup complete!"

