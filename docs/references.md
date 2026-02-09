# References and Technical Notes

## Primary References

### Kubus et al. 2008 (主要参考論文)

**Citation**:
Kubus, D., Kroger, T., & Wahl, F. M. (2008). On-line estimation of inertial
parameters using a recursive total least-squares approach. IEEE/RSJ
International Conference on Intelligent Robots and Systems.

**Key Equations**:

| Equation | Description | Implementation |
|----------|-------------|----------------|
| Eq. (1-2) | Newton-Euler motion equations | Background |
| Eq. (4-5) | Parameter estimation form | `kinematics.compute_regressor()` |
| Eq. (6) | Regressor matrix A | `kinematics.compute_regressor_matrix()` |
| Eq. (7-9) | Sensor offset compensation | `sensor/contact_sensor.py` |
| Eq. (10-11) | Fourier trajectory | `trajectories/fourier.py` |
| Eq. (12-13) | Condition number | `trajectory/condition_optimizer.py` |
| Eq. (14-15) | Jerk limitation | `trajectory/jerk_limiter.py` |
| Eq. (17-19) | RLS algorithm | `estimation/rls.py` |
| Eq. (20-22) | RIV algorithm | (Not implemented) |
| Eq. (23) | TLS error model | `estimation/batch_tls.py` |
| Eq. (24-36) | RTLS algorithm | `estimation/rtls.py` |

**Experimental Setup**:
- Robot: Stäubli RX60
- Sensor: JR3 85M35A3-I40-D 200N12 (force-torque + acceleration)
- Estimation frequency: 1000 Hz
- Estimation duration: 1.5 s

---

## Supporting References

### Brand 2002 (Incremental SVD)

**Citation**:
Brand, M. (2002). Incremental singular value decomposition of uncertain data
with missing values. Proceedings of the ECCV 2002, Lecture Notes on Computer
Science 2350.

**Key Points**:
- Exact and stable SVD update
- Does not require right singular vectors for TLS solution
- Suitable for real-time applications

### Golub & Van Loan 1990 (Matrix Computations)

**Citation**:
Golub, G. H., & Van Loan, C. F. (1990). Matrix Computations.
Johns Hopkins University Press.

**Relevant Sections**:
- Chapter 8: Least Squares Problems
- TLS solution via SVD (Section 12.3)

---

## Equation Details

### Parameter Vector φ (Eq. 5)

$$\varphi = [m, mc_x, mc_y, mc_z, I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}]^T$$

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | $m$ | Mass |
| 1 | $mc_x$ | Mass × CoM x-coordinate |
| 2 | $mc_y$ | Mass × CoM y-coordinate |
| 3 | $mc_z$ | Mass × CoM z-coordinate |
| 4 | $I_{xx}$ | Moment of inertia (xx) |
| 5 | $I_{xy}$ | Product of inertia (xy) |
| 6 | $I_{xz}$ | Product of inertia (xz) |
| 7 | $I_{yy}$ | Moment of inertia (yy) |
| 8 | $I_{yz}$ | Product of inertia (yz) |
| 9 | $I_{zz}$ | Moment of inertia (zz) |

### Regressor Matrix A (Eq. 6)

6×10 matrix structure:
- Rows 1-3: Force equations (f_x, f_y, f_z)
- Rows 4-6: Torque equations (τ_x, τ_y, τ_z)

### Offset Compensation (Eq. 7-9)

$$A_{offs} = A + A_{g_{init}}$$

where $A_{g_{init}}$ accounts for gravitational forces eliminated by zeroing
the sensor at initial orientation.

---

## Isaac Sim Technical Notes

### Contact Sensor API

**Documentation**: [Isaac Sim Contact Sensor](https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_contact.html)

**Key Classes**:
- `omni.isaac.sensor.ContactSensor`
- `omni.isaac.core.prims.RigidContactView`

**Considerations**:
1. Contact sensors measure contact forces between bodies
2. For inertial parameter estimation, we need:
   - Forces/torques acting on the grasped object
   - OR reaction forces at the wrist/gripper

**Alternative Approach**:
- Use inverse dynamics to compute expected forces
- Compare with measured forces to isolate object contribution

### Physics Simulation Rate

**Default**: 60 Hz (rendering) / 120 Hz (physics)
**Configurable**: Up to 1000+ Hz for accurate dynamics

**Setting in Isaac Sim**:
```python
from isaacsim.core.api import World
world = World(physics_dt=1.0/1000.0)  # 1000 Hz
```

### Rigid Body Properties

**Getting inertial properties (ground truth)**:
```python
from pxr import UsdPhysics

mass_api = UsdPhysics.MassAPI.Get(stage, prim_path)
mass = mass_api.GetMassAttr().Get()
com = mass_api.GetCenterOfMassAttr().Get()
inertia = mass_api.GetDiagonalInertiaAttr().Get()
```

---

## Algorithm Implementation Notes

### Total Least Squares via SVD

Given the augmented matrix $[A | y]$:

1. Compute SVD: $[A | y] = U \Sigma V^T$
2. Let $v_{n+1}$ be the right singular vector corresponding to smallest singular value
3. TLS solution: $\hat{\varphi}_i = -\frac{v_{i,n+1}}{v_{n+1,n+1}}$ for $i = 1, ..., n$

**Condition**: Solution exists only if $v_{n+1,n+1} \neq 0$

### Incremental SVD Update (Brand 2002)

For existing SVD $M = USV^T$ and new data $N$:

1. Compute projection: $L = U^T N$
2. Compute orthogonal component: $JG = N - UL$ (QR decomposition)
3. Form middle matrix: $Z = \begin{bmatrix} S & L \\ 0 & G \end{bmatrix}$
4. Compute SVD of Z: $Z = U' S' V'^T$
5. Update: $U_{new} = [U | J] U'$, $S_{new} = S'$

### Numerical Considerations

1. **Weighting matrices** (D for data, T for parameters):
   - Scale columns to similar magnitude
   - Improves numerical stability

2. **Regularization**:
   - For ill-conditioned problems, consider Tikhonov regularization
   - $\hat{\varphi} = (A^T A + \lambda I)^{-1} A^T y$

3. **Condition number monitoring**:
   - Log $\kappa(A^T A)$ during estimation
   - High condition number → poor excitation trajectory

---

## Testing Strategies

### Synthetic Data Generation

```python
def generate_synthetic_data(
    phi_true: np.ndarray,
    n_samples: int,
    noise_std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic estimation data with known parameters."""
    # Generate random kinematics states
    # Compute A matrices
    # Compute y = A @ phi_true + noise
    pass
```

### Known Object Configurations

| Object | Mass (kg) | CoM (m) | Inertia (kg·m²) |
|--------|-----------|---------|-----------------|
| Unit cube (1m, ρ=1000) | 1000 | [0,0,0] | diag(166.7) |
| Sphere (r=0.1m, ρ=1000) | 4.19 | [0,0,0] | diag(0.0167) |

### Error Metrics

1. **Relative error** (Eq. 37):
   $$e_{rel}(\hat{x}) = \left| \frac{\hat{x} - x_{th}}{x_{th}} \right| \times 100\%$$

2. **RMS error**:
   $$e_{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - A_i \hat{\varphi})^2}$$

3. **Condition number**:
   $$\kappa(\Upsilon) = \frac{\sigma_{max}}{\sigma_{min}}$$
