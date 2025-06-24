# GroundingDINO Installation Guide

## Prerequisites

### Option 1: Using uv (Recommended)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate uv environment
uv venv
source .venv/bin/activate
```

### Option 2: Using conda
```bash
# Create conda environment with Python 3.10
conda create -n dino python=3.10
conda activate dino

# Install CUDA
conda install nvidia::cuda

# Set environment variables
conda env config vars set CUDA_HOME="/storage/home/hcoda1/1/awilcox31/miniforge3/envs/dino/"
conda env config vars set LD_LIBRARY_PATH="/storage/home/hcoda1/1/awilcox31/miniforge3/envs/dino/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
conda env config vars set CPATH="/storage/home/hcoda1/1/awilcox31/miniforge3/envs/dino/targets/x86_64-linux/include/:$CPATH"

# Install build tools
pip install pip==22.3.1
pip install "setuptools>=62.3.0,<75.9"
conda install -c conda-forge gxx_linux-64 gcc_linux-64
conda install -c conda-forge ninja
```

## Installation Methods

### Method 1: Local Development Installation
Follow the steps below if you have the GroundingDINO source code locally.

### Method 2: Installing as a Package from Another Repository
Use this method when you want to install GroundingDINO as a dependency in another project.

#### Option A: Install from Git Repository
```bash
# In your project's pyproject.toml or requirements.txt
# Add this line to your dependencies:
groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git

# Or install directly with uv:
uv add groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git

# Or with pip:
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

#### Option B: Install from Local Path
```bash
# If GroundingDINO is in a local directory
# Add to pyproject.toml dependencies:
groundingdino = {path = "../path/to/GroundingDINO", editable = true}

# Or install directly:
uv add --editable ../path/to/GroundingDINO
pip install -e ../path/to/GroundingDINO
```

#### Option C: Install from Specific Branch/Commit
```bash
# Install from a specific branch
uv add groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git@main

# Install from a specific commit
uv add groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git@commit_hash
```

## Local Development Installation Steps

### 1. Clone and Setup
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
```

### 2. Install Dependencies (uv method)
```bash
# Install dependencies from pyproject.toml
uv sync
```

### 3. Build C++/CUDA Extensions
```bash
# Set CUDA architecture for your GPU (L40S = 8.9)
export TORCH_CUDA_ARCH_LIST="8.9"

# Set PyTorch library path
export LD_LIBRARY_PATH="/storage/project/r-agarg35-0/awilcox31/GroundingDINO/.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Build the extensions
uv run python setup.py build
```

### 4. Install Package in Editable Mode
```bash
# Install with C++ extensions
uv run python setup.py develop
```

### 5. Verify Installation
```bash
# Test import
uv run python -c "from groundingdino import _C; print('Successfully imported _C extension')"
```

## Package Installation Configuration

### For uv projects (pyproject.toml)
```toml
[project]
dependencies = [
    "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git",
    # other dependencies...
]
```

### For pip projects (requirements.txt)
```txt
git+https://github.com/IDEA-Research/GroundingDINO.git
# other dependencies...
```

### Environment Setup for Package Installation
When installing as a package, you may need to set environment variables in your project:

```bash
# Add to your project's setup script or environment
export TORCH_CUDA_ARCH_LIST="8.9"  # Adjust for your GPU
export LD_LIBRARY_PATH="/path/to/your/venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
```

## Troubleshooting

### CUDA Architecture Issues
- For L40S GPU: Use `TORCH_CUDA_ARCH_LIST="8.9"`
- For other GPUs, check your GPU's compute capability and set accordingly

### Library Path Issues
If you get `libc10.so: cannot open shared object file`:
```bash
export LD_LIBRARY_PATH="/path/to/your/venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
```

### Build Issues
If extensions fail to build:
1. Clean previous builds: `rm -rf build/ groundingdino/_C*.so`
2. Rebuild: `uv run python setup.py build`
3. Reinstall: `uv run python setup.py develop`

### Package Installation Issues
If installing as a package fails:
1. Ensure you have the correct CUDA version installed
2. Set the appropriate `TORCH_CUDA_ARCH_LIST` for your GPU
3. Make sure PyTorch is installed with CUDA support
4. Check that all build dependencies are available

## Notes

- The project uses a hybrid approach: `pyproject.toml` for metadata and dependencies, `setup.py` for C++/CUDA extensions
- CUDA extensions require specific architecture flags for your GPU
- Library paths must be set correctly for runtime loading of PyTorch libraries
- Use `uv run` to ensure commands run in the correct virtual environment
- When installing as a package, the C++ extensions will be built automatically during installation
- Package installation may take longer due to C++/CUDA compilation