#!/bin/bash
# Run inertial parameter identification test in Isaac Sim

set -e

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$THIS_SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$THIS_SCRIPT_DIR")")")"

echo "=========================================="
echo "Inertial Parameter Identification Test"
echo "=========================================="
echo "Package: $PACKAGE_DIR"
echo "Workspace: $WORKSPACE_DIR"

# Source ROS 2 workspaces (underlay first, then overlay)
UNDERLAY_DIR="$(dirname "$WORKSPACE_DIR")/underlay_ws"
if [ -f "$UNDERLAY_DIR/install/setup.bash" ]; then
    source "$UNDERLAY_DIR/install/setup.bash"
    echo "Sourced: $UNDERLAY_DIR/install/setup.bash"
fi

if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
    echo "Sourced: $WORKSPACE_DIR/install/setup.bash"
fi

# Isaac Sim environment setup
ISAAC_SIM_DIR="/isaac-sim"
export PYTHONPATH=$ISAAC_SIM_DIR/exts/isaacsim.ros2.bridge/jazzy/rclpy:$PYTHONPATH

# Add package source to PYTHONPATH for estimation module
export PYTHONPATH="$PACKAGE_DIR/src:$PYTHONPATH"

# Set headless mode
export ISAAC_HEADLESS=true

echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "Running test..."
echo ""

# Run with Isaac Sim Python
$ISAAC_SIM_DIR/python.sh "$THIS_SCRIPT_DIR/run_identification_test.py" --headless "$@"

echo ""
echo "Test complete."
