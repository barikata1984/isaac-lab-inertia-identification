# Kinematics Package Implementation Summary

## Overview

Pinocchio-based kinematics computation package for UR robots.
Provides forward kinematics, velocity, acceleration, and regressor matrix computation.

## Package Structure

```
kinematics/
├── src/
│   ├── __init__.py
│   └── kinematics.py      # Main implementation
├── test/
│   └── test_kinematics.py # 26 tests
├── notes/
│   └── implementation_summary.md
├── package.xml
└── setup.py
```

## Implemented Features

### 1. Forward Kinematics

Compute tool0 position and orientation from joint angles.

```python
from kinematics import PinocchioKinematics

kin = PinocchioKinematics.for_ur5e()
position, rotation = kin.forward_kinematics(q)
```

### 2. Tool0 Velocity

Compute classical velocity (linear and angular) in LOCAL or WORLD frame.

```python
linear_vel, angular_vel = kin.tool0_velocity(q, dq, frame="local")
```

### 3. Tool0 Acceleration

Compute classical acceleration (includes centrifugal/Coriolis effects).

```python
linear_acc, angular_acc = kin.tool0_acceleration(q, dq, ddq, frame="local")
```

### 4. Complete State Computation

Efficient single-call computation of pose, velocity, and acceleration.

```python
state = kin.compute_full_state(q, dq, ddq)
# state.position, state.rotation
# state.linear_velocity, state.angular_velocity
# state.linear_acceleration, state.angular_acceleration
```

### 5. Regressor Matrix for Inertial Parameter Estimation

Based on Kubus et al. 2008, "On-line estimation of inertial parameters
using a recursive total least-squares approach", Eq. (6).

#### Equation

$$\begin{pmatrix} f \\ \tau \end{pmatrix} = A \cdot \varphi$$

where:
- $A$: 6×10 regressor matrix
- $\varphi$: 10-element parameter vector

#### Parameter Vector

$$\varphi = [m, mc_x, mc_y, mc_z, I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}]^T$$

#### Usage

**Method 1: From joint state**
```python
A = kin.compute_regressor(q, dq, ddq, gravity=[0, 0, -9.81])
```

**Method 2: From kinematics quantities directly**
```python
from kinematics import compute_regressor_matrix

A = compute_regressor_matrix(
    a=linear_acceleration,    # (3,) in tool0 frame
    alpha=angular_acceleration,  # (3,) in tool0 frame
    omega=angular_velocity,   # (3,) in tool0 frame
    g=gravity_in_tool0,       # (3,) in tool0 frame
)
```

#### Matrix Structure (Eq. 6)

| Row | Equation | Non-zero columns |
|-----|----------|------------------|
| 1-3 | Force (f_x, f_y, f_z) | m, mc_x, mc_y, mc_z |
| 4-6 | Torque (τ_x, τ_y, τ_z) | mc_x, mc_y, mc_z, I_xx...I_zz |

## Vector Convention

**[linear, angular]** ordering throughout:
- Velocity: [v, ω]
- Acceleration: [a, α]

This matches Pinocchio and Isaac Sim conventions.

## Frame Conventions

- **LOCAL**: tool0 frame (default)
- **WORLD**: base frame
- Rotation matrix R: transforms from base to tool0

## Dependencies

- `pinocchio`: Robotics library (installed via apt)
- `ur_description`: UR robot URDF (ROS 2 package)
- `numpy`: Numerical computation

## Tests

26 tests covering:
- Forward kinematics (known poses, symmetry)
- Velocity computation (zero, local vs world)
- Acceleration computation (classical vs spatial)
- Regressor matrix (shape, zero case, static case, physics)
- Full state computation

Run tests:
```bash
cd /workspaces/isaac-sim-ur5e/colcon_ws
colcon build --packages-select kinematics
source install/setup.bash
python3 -m pytest src/kinematics/test/ -v
```

## Legacy Package

The original pymlg-based implementation was renamed to `kinematics_legacy`.

## References

- Kubus, D., Kroger, T., & Wahl, F. M. (2008). On-line estimation of inertial
  parameters using a recursive total least-squares approach. IEEE/RSJ
  International Conference on Intelligent Robots and Systems.
- Pinocchio documentation: https://stack-of-tasks.github.io/pinocchio/
