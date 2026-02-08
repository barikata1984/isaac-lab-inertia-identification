"""Batch Total Least Squares estimation for inertial parameters.

This module implements Total Least Squares (TLS) for batch processing,
which accounts for errors in both the regressor matrix A and
observation vector y.

Reference:
    Kubus et al. 2008, Section IV-C, Eq. (23).
    Golub & Van Loan, "Matrix Computations", Chapter 12.
"""

from typing import Optional, Tuple

import numpy as np

from iparam_identification.sensor.data_types import EstimationResult

from .base_estimator import (
    BatchEstimatorBase,
    EstimatorConfig,
    compute_condition_number,
    compute_residual_norm,
    validate_estimation_input,
)


class BatchTotalLeastSquares(BatchEstimatorBase):
    """Batch Total Least Squares (TLS) estimator.

    Solves the errors-in-variables (EIV) problem:
        y + e = (A + E) @ φ

    where both e (observation error) and E (regressor error) are
    considered. This is more appropriate than OLS when the regressor
    matrix A contains measurement noise (e.g., from accelerations
    computed via numerical differentiation).

    The TLS solution is computed via SVD of the augmented matrix [A | y].
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        weighting: Optional[np.ndarray] = None,
    ):
        """Initialize TLS estimator.

        Args:
            config: Estimator configuration.
            weighting: Optional (11, 11) weighting matrix for generalized TLS.
                Used to weight different components of [A | y] differently.
        """
        super().__init__(config)
        self._weighting = weighting
        self._singular_values: Optional[np.ndarray] = None

    def estimate(
        self,
        A: np.ndarray,
        y: np.ndarray,
    ) -> EstimationResult:
        """Estimate parameters using Total Least Squares.

        Args:
            A: Stacked regressor matrix (N*6, 10).
            y: Stacked observation vector (N*6,).

        Returns:
            EstimationResult containing estimated parameters and metrics.
        """
        A, y, n_samples = validate_estimation_input(
            A, y, self.config.n_params
        )

        # Compute TLS solution
        if self._weighting is not None:
            phi_hat, sv = generalized_total_least_squares(
                A, y, self._weighting
            )
        else:
            phi_hat, sv = batch_total_least_squares(A, y)

        self._singular_values = sv

        # Compute quality metrics
        condition_number = compute_condition_number(A)
        # For TLS, use TLS residual (smallest singular value)
        residual_norm = float(sv[-1]) if sv is not None else np.nan

        self._result = EstimationResult(
            phi=phi_hat,
            condition_number=condition_number,
            residual_norm=residual_norm,
            n_samples=n_samples,
        )

        return self._result

    @property
    def singular_values(self) -> Optional[np.ndarray]:
        """Singular values from the most recent TLS computation."""
        return self._singular_values


def batch_total_least_squares(
    A: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch Total Least Squares estimation via SVD.

    Solves the TLS problem:
        min ||[E | e]||_F  subject to  (y + e) = (A + E) @ φ

    The solution is computed by SVD of the augmented matrix [A | y].

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).

    Returns:
        Tuple of (phi_hat, singular_values):
            - phi_hat: Estimated parameter vector (10,).
            - singular_values: All singular values of [A | y].

    Mathematical formulation:
        1. Form augmented matrix C = [A | y]  (shape: m x 11)
        2. Compute SVD: C = U @ S @ V^T
        3. Extract last column of V: v = V[:, -1]
        4. TLS solution: φ = -v[:-1] / v[-1]

    Note:
        The TLS solution exists if and only if v[-1] ≠ 0.
        If v[-1] ≈ 0, the problem is ill-posed.
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel().reshape(-1, 1)

    # Augmented matrix [A | y]
    C = np.hstack([A, y])

    # Full SVD
    U, S, Vt = np.linalg.svd(C, full_matrices=True)

    # Last column of V (last row of V^T)
    v = Vt[-1, :]

    # Check solvability
    if np.abs(v[-1]) < 1e-15:
        raise ValueError(
            "TLS problem is ill-posed: the last component of the "
            "smallest right singular vector is nearly zero."
        )

    # TLS solution: φ = -v[:-1] / v[-1]
    phi_hat = -v[:-1] / v[-1]

    return phi_hat, S


def generalized_total_least_squares(
    A: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Total Least Squares with row weighting.

    Applies weighting to account for different noise levels in
    different measurements. The weighting is applied to the
    augmented matrix before SVD.

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        W: Weighting matrix (11, 11) for column scaling of [A | y].

    Returns:
        Tuple of (phi_hat, singular_values).

    Note:
        For row weighting (different noise per measurement), use
        row_weighted_tls instead.
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel().reshape(-1, 1)
    W = np.asarray(W)

    # Augmented matrix
    C = np.hstack([A, y])

    # Apply column weighting: C_weighted = C @ W
    # This assumes W represents uncertainty in parameters + observation
    C_weighted = C @ W

    # SVD of weighted matrix
    U, S, Vt = np.linalg.svd(C_weighted, full_matrices=True)

    # Transform back: V_original = W @ V_weighted
    v_weighted = Vt[-1, :]
    v = W @ v_weighted

    if np.abs(v[-1]) < 1e-15:
        raise ValueError("GTLS problem is ill-posed.")

    phi_hat = -v[:-1] / v[-1]

    return phi_hat, S


def row_weighted_tls(
    A: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Total Least Squares with row weighting.

    Weights different measurements differently, useful when some
    measurements are more reliable than others.

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        weights: Per-row weights (N*6,). Larger = more reliable.

    Returns:
        Tuple of (phi_hat, singular_values).
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel()
    weights = np.asarray(weights).ravel()

    # Apply row weighting
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    A_weighted = A * sqrt_w
    y_weighted = y * sqrt_w.ravel()

    return batch_total_least_squares(A_weighted, y_weighted)


def truncated_tls(
    A: np.ndarray,
    y: np.ndarray,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated Total Least Squares.

    Uses only the first `rank` singular values/vectors, which can
    improve stability when the problem is nearly rank-deficient.

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        rank: Number of singular values to retain (< min(m, 11)).

    Returns:
        Tuple of (phi_hat, singular_values).

    Note:
        This is equivalent to TLS on the rank-reduced approximation
        of [A | y].
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel().reshape(-1, 1)

    C = np.hstack([A, y])
    m, n = C.shape

    if rank >= n:
        return batch_total_least_squares(A, y.ravel())

    # SVD
    U, S, Vt = np.linalg.svd(C, full_matrices=True)

    # Truncated reconstruction
    # Set small singular values to zero
    S_truncated = S.copy()
    S_truncated[rank:] = 0

    # Reconstruct
    C_truncated = U[:, :len(S)] @ np.diag(S_truncated) @ Vt[:len(S), :]

    # Apply standard TLS on truncated matrix
    A_truncated = C_truncated[:, :-1]
    y_truncated = C_truncated[:, -1]

    return batch_total_least_squares(A_truncated, y_truncated)


def regularized_tls(
    A: np.ndarray,
    y: np.ndarray,
    regularization: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Regularized Total Least Squares.

    Adds regularization to improve numerical stability for
    ill-conditioned problems.

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        regularization: Regularization parameter λ.

    Returns:
        Tuple of (phi_hat, singular_values).

    Mathematical formulation:
        min ||[E | e]||_F + λ||φ||²  subject to  (y + e) = (A + E) @ φ
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel()
    n_params = A.shape[1]

    # Augment with regularization rows
    A_reg = np.vstack([A, np.sqrt(regularization) * np.eye(n_params)])
    y_reg = np.concatenate([y, np.zeros(n_params)])

    return batch_total_least_squares(A_reg, y_reg)
