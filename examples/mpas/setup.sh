#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Setup script for the MPAS multi-agent example.
# Source this script so it can activate the conda environment in the
# current shell.

ERREXIT_ENABLED=0
case $- in
    *e*) ERREXIT_ENABLED=1 ;;
esac

set -e

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced, not executed."
    echo "Use: source ./setup.sh"
    return 1 2>/dev/null || exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_NAME="mpas-example"
MINIFORGE_INSTALLER="Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_INSTALLER}"
MINIFORGE_DIR="${SCRIPT_DIR}/miniforge3"

echo "=========================================="
echo "MPAS Multi-Agent Example Setup"
echo "=========================================="
echo ""

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if [ -x "${MINIFORGE_DIR}/bin/conda" ]; then
    echo "✓ Miniforge found at ${MINIFORGE_DIR}"
    CONDA_EXE="${MINIFORGE_DIR}/bin/conda"
elif command_exists conda; then
    CONDA_EXE="$(command -v conda)"
    echo "✓ Conda found: ${CONDA_EXE}"
else
    echo "✗ Conda not found. Installing Miniforge into ${MINIFORGE_DIR}..."

    TEMP_DIR="$(mktemp -d)"
    cd "${TEMP_DIR}"

    echo "  Downloading Miniforge (conda-forge + mamba)..."
    wget -q "${MINIFORGE_URL}" || curl -s -L -O "${MINIFORGE_URL}"

    if [ ! -f "${MINIFORGE_INSTALLER}" ]; then
        echo "ERROR: Failed to download Miniforge installer"
        exit 1
    fi

    echo "  Installing Miniforge..."
    bash "${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_DIR}"

    cd "${SCRIPT_DIR}"
    rm -rf "${TEMP_DIR}"

    CONDA_EXE="${MINIFORGE_DIR}/bin/conda"
    echo "✓ Miniforge installed successfully"
fi

echo ""

if "${CONDA_EXE}" env list | grep -q "^${ENV_NAME} "; then
    echo "Updating conda environment '${ENV_NAME}'..."
    "${CONDA_EXE}" env update -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml" --prune
else
    echo "Creating conda environment '${ENV_NAME}'..."
    "${CONDA_EXE}" env create -f "${SCRIPT_DIR}/environment.yml" -n "${ENV_NAME}"
fi

eval "$(${CONDA_EXE} shell.bash hook)"
conda activate "${ENV_NAME}"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "The '${ENV_NAME}' environment is now active in this shell."
echo ""
echo "To run the MPAS example:"
echo "  cd ${SCRIPT_DIR}"
echo "  cp config/user_config.yaml.template config/user_config.yaml"
echo "  # Edit config/user_config.yaml with your settings"
echo "  python run_mpas_forecast.py config/user_config.yaml"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "To install chiltepin from local source:"
echo "  pip install -e ${SCRIPT_DIR}/../..[test]"
echo ""

if [ "${ERREXIT_ENABLED}" -eq 0 ]; then
    set +e
fi
