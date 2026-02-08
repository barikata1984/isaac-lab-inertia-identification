"""Estimation algorithms module for inertial parameter identification.

Provides parameter estimation algorithms:
- Batch methods: OLS, TLS (recommended for offline)
- Recursive methods: RLS, RTLS (recommended for online)

Usage:
    # Batch estimation (offline)
    from iparam_identification.estimation import BatchLeastSquares, BatchTotalLeastSquares

    estimator = BatchTotalLeastSquares()
    result = estimator.estimate(A, y)
    print(result.mass, result.center_of_mass)

    # Recursive estimation (online)
    from iparam_identification.estimation import RecursiveTotalLeastSquares

    estimator = RecursiveTotalLeastSquares()
    for A_k, y_k in measurement_stream:
        phi = estimator.update(A_k, y_k)
        print(f"Current mass estimate: {phi[0]:.3f}")
"""

# Base classes and utilities
from .base_estimator import (
    BatchEstimatorBase,
    RecursiveEstimatorBase,
    EstimatorConfig,
    validate_estimation_input,
    compute_condition_number,
    compute_residual_norm,
)

# Batch estimators
from .batch_ls import (
    BatchLeastSquares,
    batch_least_squares,
    weighted_least_squares,
    iteratively_reweighted_ls,
)

from .batch_tls import (
    BatchTotalLeastSquares,
    batch_total_least_squares,
    generalized_total_least_squares,
    row_weighted_tls,
    truncated_tls,
    regularized_tls,
)

# Recursive estimators
from .rls import (
    RecursiveLeastSquares,
    RecursiveInstrumentalVariable,
)

from .rtls import (
    RecursiveTotalLeastSquares,
    WindowedRTLS,
    AdaptiveRTLS,
)

# SVD utilities
from .svd_update import (
    SVDState,
    initialize_svd,
    incremental_svd_update,
    incremental_svd_update_block,
    get_tls_solution_from_svd,
)

__all__ = [
    # Base classes
    "BatchEstimatorBase",
    "RecursiveEstimatorBase",
    "EstimatorConfig",
    "validate_estimation_input",
    "compute_condition_number",
    "compute_residual_norm",
    # Batch LS
    "BatchLeastSquares",
    "batch_least_squares",
    "weighted_least_squares",
    "iteratively_reweighted_ls",
    # Batch TLS
    "BatchTotalLeastSquares",
    "batch_total_least_squares",
    "generalized_total_least_squares",
    "row_weighted_tls",
    "truncated_tls",
    "regularized_tls",
    # Recursive LS
    "RecursiveLeastSquares",
    "RecursiveInstrumentalVariable",
    # Recursive TLS
    "RecursiveTotalLeastSquares",
    "WindowedRTLS",
    "AdaptiveRTLS",
    # SVD utilities
    "SVDState",
    "initialize_svd",
    "incremental_svd_update",
    "incremental_svd_update_block",
    "get_tls_solution_from_svd",
]
