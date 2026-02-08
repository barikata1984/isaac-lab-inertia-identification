#!/bin/bash
# =============================================================================
# setup.sh — First-time setup for Isaac Sim/Lab dev container
#
# Run this on the HOST before building the container:
#   chmod +x setup.sh && ./setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

echo "=== Isaac Sim/Lab Dev Container Setup ==="

# ---- Detect host UID/GID and write to .env ----------------------------------
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(whoami)

echo ""
echo "Detected host user: ${HOST_USER} (UID=${HOST_UID}, GID=${HOST_GID})"

# Update .env with actual host values
if [ -f "${ENV_FILE}" ]; then
    sed -i "s/^HOST_UID=.*/HOST_UID=${HOST_UID}/" "${ENV_FILE}"
    sed -i "s/^HOST_GID=.*/HOST_GID=${HOST_GID}/" "${ENV_FILE}"
    sed -i "s/^HOST_USER=.*/HOST_USER=${HOST_USER}/" "${ENV_FILE}"
    echo "Updated .env with host user info."
else
    echo "WARNING: .env not found at ${ENV_FILE}"
fi

# ---- Ensure NVIDIA Container Toolkit is available ---------------------------
if ! command -v nvidia-container-toolkit &> /dev/null && ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo ""
    echo "WARNING: nvidia-container-toolkit not found."
    echo "Please install it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# ---- Ensure Docker runtime is configured for NVIDIA -------------------------
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "NVIDIA Docker runtime detected."
else
    echo ""
    echo "NOTE: NVIDIA runtime not detected in 'docker info'."
    echo "Make sure /etc/docker/daemon.json includes the nvidia runtime,"
    echo "or use 'docker compose' v2.22+ which supports 'runtime: nvidia' natively."
fi

# ---- Allow X11 forwarding from containers ----------------------------------
echo ""
echo "Setting up X11 forwarding..."
xhost +local:docker 2>/dev/null || echo "xhost not available (headless server?)"

# ---- Login to NGC (needed to pull Isaac Sim image) --------------------------
echo ""
echo "You need NGC access to pull the Isaac Sim container image."
echo "If you haven't logged in yet, run:"
echo "  docker login nvcr.io"
echo "  Username: \$oauthtoken"
echo "  Password: <your NGC API key>"

# ---- Build ------------------------------------------------------------------
echo ""
read -p "Build the container now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "${SCRIPT_DIR}"
    docker compose -f docker/docker-compose.yml build
    echo ""
    echo "=== Build complete ==="
    echo "Start with:  docker compose -f docker/docker-compose.yml up -d"
    echo "Enter with:  docker compose -f docker/docker-compose.yml exec isaac-lab-inertia-identification zsh"
    echo "Or open in VS Code with Dev Containers extension."
fi
