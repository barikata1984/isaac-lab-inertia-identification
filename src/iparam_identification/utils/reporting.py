"""Result reporting utilities for inertial parameter identification."""

import numpy as np

PARAM_NAMES = ['m', 'm*cx', 'm*cy', 'm*cz',
               'Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']


def print_results(
    phi_true: np.ndarray,
    result_ols,
    result_tls,
    result_rtls,
    cond_number: float,
    n_samples: int,
    duration: float,
) -> None:
    """Print formatted comparison table of estimation results."""
    print("\n" + "=" * 70)
    print("  Inertial Parameter Identification Results")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s | Samples: {n_samples} | "
          f"Condition number: {cond_number:.2f}")

    # Method summary
    print(f"\n  {'Method':<8} | {'Mass [kg]':>10} | {'Error [%]':>10} | {'Residual':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for name, result in [("OLS", result_ols), ("TLS", result_tls), ("RTLS", result_rtls)]:
        mass_err = 100 * abs(result.mass - phi_true[0]) / phi_true[0]
        residual = result.residual_norm if result.residual_norm is not None else float("nan")
        print(f"  {name:<8} | {result.mass:10.4f} | {mass_err:10.2f} | {residual:10.4f}")

    # Parameter-wise detail
    print(f"\n  {'Param':<6} | {'True':>12} | {'OLS':>12} | {'TLS':>12} | {'RTLS':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for i, name in enumerate(PARAM_NAMES):
        tv = phi_true[i]
        ov = result_ols.phi[i]
        tlv = result_tls.phi[i]
        rv = result_rtls.phi[i]
        print(f"  {name:<6} | {tv:12.6f} | {ov:12.6f} | {tlv:12.6f} | {rv:12.6f}")

    print("=" * 70)
