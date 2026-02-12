#!/bin/bash
set -e

# =============================================================================
# Container entrypoint
# - Sets up zsh/oh-my-zsh for the runtime user
# - Ensures correct permissions
# - Configures Isaac Sim Python aliases
# =============================================================================

TARGET_USER="${HOST_USER:-developer}"
TARGET_HOME=$(eval echo "~${TARGET_USER}")
TARGET_GROUP=$(id -gn "${TARGET_USER}" 2>/dev/null || echo "${TARGET_USER}")

# ---- Setup zsh for the user if not already done ----------------------------
if [ ! -d "${TARGET_HOME}/.oh-my-zsh" ]; then
    cp -r /etc/skel/.oh-my-zsh "${TARGET_HOME}/.oh-my-zsh"
    chown -R "${TARGET_USER}:${TARGET_GROUP}" "${TARGET_HOME}/.oh-my-zsh"
fi

if [ ! -f "${TARGET_HOME}/.zshrc" ]; then
    cat > "${TARGET_HOME}/.zshrc" << 'ZSHRC'
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"
plugins=(git python zsh-autosuggestions zsh-syntax-highlighting)
source $ZSH/oh-my-zsh.sh

# --- Isaac Sim / Lab aliases ------------------------------------------------
export ISAAC_SIM_PATH=/isaac-sim
export ISAAC_LAB_PATH=/opt/isaac-lab
export ISAACSIM_PATH=/isaac-sim

# Use Isaac Sim's bundled Python
alias isaac-python='/isaac-sim/python.sh'
alias isaac-pip='/isaac-sim/python.sh -m pip'
alias isaaclab='/opt/isaac-lab/isaaclab.sh'

# Convenience: add Isaac Lab scripts and user-local bin to PATH
export PATH="$HOME/.local/bin:/opt/isaac-lab:${PATH}"
ZSHRC
    chown "${TARGET_USER}:${TARGET_GROUP}" "${TARGET_HOME}/.zshrc"
fi

# ---- Fix ownership of Isaac Sim writable dirs --------------------------------
for d in /isaac-sim/kit/data /isaac-sim/kit/cache; do
    mkdir -p "$d"
    chown -R "${TARGET_USER}:${TARGET_GROUP}" "$d" 2>/dev/null || true
done

# ---- Fix ownership of cache dirs that may be mounted as root ----------------
for d in \
    "${TARGET_HOME}/.cache" \
    "${TARGET_HOME}/.local" \
    "${TARGET_HOME}/.nvidia-omniverse" \
    "${TARGET_HOME}/.nv"; do
    if [ -d "$d" ]; then
        chown -R "${TARGET_USER}:${TARGET_GROUP}" "$d" 2>/dev/null || true
    fi
done

# ---- Install project in editable mode (if pyproject.toml exists) -----------
if [ -f /workspace/pyproject.toml ]; then
    echo "📦 Installing project in editable mode..."
    /isaac-sim/python.sh -m pip install --no-deps -e /workspace 2>&1 | tail -1
fi

# ---- Execute command as the target user -------------------------------------
if [ "$(id -u)" = "0" ] && [ "${TARGET_USER}" != "root" ]; then
    exec gosu "${TARGET_USER}" "$@"
else
    exec "$@"
fi
