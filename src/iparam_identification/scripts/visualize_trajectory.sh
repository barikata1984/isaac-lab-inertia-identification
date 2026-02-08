#!/bin/bash
# Visualize optimized excitation trajectory in Isaac Sim GUI
#
# Usage:
#   ros2 run iparam_identification visualize_trajectory.sh [--speed 0.5] [--loop]

set -e

echo "=========================================="
echo "Trajectory Visualization (Isaac Sim GUI)"
echo "=========================================="

# Find visualize_trajectory.py via Python package import (same pattern as core/isaac_sim_gui.sh)
PY_SCRIPT=$(python3 -c "import iparam_identification; import os; print(os.path.join(os.path.dirname(iparam_identification.__file__), 'visualize_trajectory.py'))" 2>/dev/null)

if [ -z "$PY_SCRIPT" ]; then
    PREFIX=$(ros2 pkg prefix iparam_identification 2>/dev/null)
    if [ -n "$PREFIX" ]; then
        PY_SCRIPT=$(find -L "$PREFIX" -name "visualize_trajectory.py" | head -n 1)
    fi
fi

if [ -z "$PY_SCRIPT" ] || [ ! -f "$PY_SCRIPT" ]; then
    echo "ERROR: Could not find visualize_trajectory.py"
    exit 1
fi

# Find package source dir for PYTHONPATH
PKG_SRC_DIR=$(python3 -c "import iparam_identification; import os; print(os.path.dirname(iparam_identification.__file__))" 2>/dev/null)

# Isaac Sim environment setup
export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy:$PYTHONPATH
if [ -n "$PKG_SRC_DIR" ]; then
    export PYTHONPATH="$PKG_SRC_DIR:$PYTHONPATH"
fi
export ISAAC_HEADLESS=false

echo "Script: $PY_SCRIPT"
echo ""

/isaac-sim/python.sh "$PY_SCRIPT" "$@"
