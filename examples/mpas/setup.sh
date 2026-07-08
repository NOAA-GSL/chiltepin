#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Setup script for MPAS multi-agent example
# This script:
# 1. Checks for or installs Miniforge (conda-forge + mamba)
# 2. Creates conda environment with uwtools
# 3. Pip installs chiltepin from PyPI into the conda environment

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_NAME="mpas-example"
MINIFORGE_VERSION="latest"
MINIFORGE_INSTALLER="Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_INSTALLER}"

echo "=========================================="
echo "MPAS Multi-Agent Example Setup"
echo "=========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for conda
if command_exists conda; then
    echo "✓ Conda found: $(which conda)"
    CONDA_EXE=$(which conda)
else
    echo "✗ Conda not found. Installing Miniforge..."
    
    # Create temporary directory for download
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download Miniforge installer
    echo "  Downloading Miniforge (conda-forge + mamba)..."
    wget -q "$MINIFORGE_URL" || curl -s -L -O "$MINIFORGE_URL"
    
    if [ ! -f "$MINIFORGE_INSTALLER" ]; then
        echo "ERROR: Failed to download Miniforge installer"
        exit 1
    fi
    
    # Install Miniforge
    echo "  Installing Miniforge to $HOME/miniforge3..."
    bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
    
    # Initialize conda
    echo "  Initializing conda..."
    "$HOME/miniforge3/bin/conda" init bash
    
    # Clean up
    cd "$SCRIPT_DIR"
    rm -rf "$TEMP_DIR"
    
    # Source bashrc to get conda in PATH
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
    
    CONDA_EXE="$HOME/miniforge3/bin/conda"
    echo "✓ Miniforge installed successfully (includes mamba for faster operations)"
fi

echo ""

# Check if environment already exists
if $CONDA_EXE env list | grep -q "^${ENV_NAME} "; then
    echo "⚠ Conda environment '${ENV_NAME}' already exists"
    read -p "  Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        $CONDA_EXE env remove -n "$ENV_NAME" -y
    else
        echo "  Keeping existing environment. Update it with:"
        echo "    conda activate ${ENV_NAME}"
        echo "    conda env update -f ${SCRIPT_DIR}/environment.yml"
        echo "    pip install -e ${SCRIPT_DIR}/../..[test]"
        exit 0
    fi
fi

echo "Creating conda environment '${ENV_NAME}'..."
$CONDA_EXE env create -f "${SCRIPT_DIR}/environment.yml" -n "$ENV_NAME"

echo ""
echo "Installing chiltepin from local source with upgraded dependencies..."
# Install from local source (not PyPI) to get latest dependency constraints
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$ENV_NAME"
pip install -e "${SCRIPT_DIR}/../..[test]"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run the MPAS example:"
echo "  cd ${SCRIPT_DIR}"
echo "  conda activate ${ENV_NAME}"
echo "  cp config/user_config.yaml.template config/user_config.yaml"
echo "  # Edit config/user_config.yaml with your settings"
echo "  python run_mpas_forecast.py config/user_config.yaml"
echo ""
echo "To run tests:"
echo "  conda activate ${ENV_NAME}"
echo "  pytest tests/ -v"
echo ""
