"""Batch Least Squares estimation for inertial parameters.

This module implements the standard (Ordinary) Least Squares (OLS)
estimator for batch processing of collected data.

Reference:
    Kubus et al. 2008, Section IV-A (baseline method).
"""

from typing import Optional

import numpy as np

from iparam_identification.sensor.data_types import EstimationResult

from .base_estimator import (
    BatchEstimatorBase,
    EstimatorConfig,
    compute_condition_number,
    compute_residual_norm,
    validate_estimation_input,
)


class BatchLeastSquares(BatchEstimatorBase):
    """Batch Ordinary Least Squares (OLS) estimator.

    Solves the linear estimation problem:
        y = A @ φ + e

    where e is assumed to be zero-mean noise in y only (not in A).

    The OLS solution is:
        φ_hat = (A^T A)^{-1} A^T y = A^+ y

    where A^+ is the Moore-Penrose pseudoinverse.

    Note:
        This estimator assumes no errors in the regressor matrix A,
        which is often violated in practice. For better accuracy,
        consider using BatchTotalLeastSquares which accounts for
        errors in both A and y.
    """

    def estimate(
        self,
        A: np.ndarray,
        y: np.ndarray,
    ) -> EstimationResult:
        """Estimate parameters using Ordinary Least Squares.

        Args:
            A: Stacked regressor matrix (N*6, 10).
            y: Stacked observation vector (N*6,).

        Returns:
            EstimationResult containing estimated parameters and metrics.
        """
        A, y, n_samples = validate_estimation_input(
            A, y, self.config.n_params
        )

        # Compute OLS solution
        phi_hat = batch_least_squares(
            A, y, regularization=self.config.regularization
        )

        # Compute quality metrics
        condition_number = compute_condition_number(A)
        residual_norm = compute_residual_norm(A, y, phi_hat)

        self._result = EstimationResult(
            phi=phi_hat,
            condition_number=condition_number,
            residual_norm=residual_norm,
            n_samples=n_samples,
        )

        return self._result


def batch_least_squares(
    A: np.ndarray,
    y: np.ndarray,
    regularization: float = 0.0,
) -> np.ndarray:
    """Batch Ordinary Least Squares estimation.

    Computes the OLS solution to the linear system y = A @ φ.

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        regularization: Tikhonov regularization parameter.
            If > 0, solves: (A^T A + λI)^{-1} A^T y

    Returns:
        Estimated parameter vector φ_hat (10,).

    Mathematical formulation:
        Without regularization:
            φ_hat = (A^T A)^{-1} A^T y

        With Tikhonov regularization:
            φ_hat = (A^T A + λI)^{-1} A^T y
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel()

    if regularization > 0:
        # Tikhonov regularization
        AtA = A.T @ A
        Aty = A.T @ y
        n_params = A.shape[1]
        phi_hat = np.linalg.solve(
            AtA + regularization * np.eye(n_params), Aty
        )
    else:
        # Standard OLS using pseudoinverse
        phi_hat, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    return phi_hat


def weighted_least_squares(
    A: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Weighted Least Squares estimation.

    Computes the WLS solution with weight matrix W:
        φ_hat = (A^T W A)^{-1} A^T W y

    This is equivalent to OLS on the transformed system:
        W^{1/2} y = W^{1/2} A @ φ

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        W: Weight matrix (N*6, N*6), typically diagonal.
            Larger weights give more importance to those measurements.

    Returns:
        Estimated parameter vector φ_hat (10,).

    Note:
        For measurement noise covariance Λ, use W = Λ^{-1} to get
        the Best Linear Unbiased Estimator (BLUE).
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel()
    W = np.asarray(W)

    AtWA = A.T @ W @ A
    AtWy = A.T @ W @ y

    phi_hat = np.linalg.solve(AtWA, AtWy)

    return phi_hat


def iteratively_reweighted_ls(
    A: np.ndarray,
    y: np.ndarray,
    max_iterations: int = 10,
    tol: float = 1e-6,
    huber_delta: float = 1.345,
) -> np.ndarray:
    """Iteratively Reweighted Least Squares (IRLS) for robust estimation.

    Uses Huber weighting to reduce the influence of outliers.

    Args:
        A: Stacked regressor matrix (N*6, 10).
        y: Stacked observation vector (N*6,).
        max_iterations: Maximum number of iterations.
        tol: Convergence tolerance on parameter change.
        huber_delta: Huber function threshold.

    Returns:
        Estimated parameter vector φ_hat (10,).
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel()
    n = len(y)

    # Initial OLS estimate
    phi_hat = batch_least_squares(A, y)

    for _ in range(max_iterations):
        # Compute residuals
        residuals = y - A @ phi_hat
        abs_residuals = np.abs(residuals)

        # Compute Huber weights
        weights = np.ones(n)
        mask = abs_residuals > huber_delta
        weights[mask] = huber_delta / abs_residuals[mask]

        # Weighted least squares update
        W = np.diag(weights)
        phi_new = weighted_least_squares(A, y, W)

        # Check convergence
        if np.linalg.norm(phi_new - phi_hat) < tol:
            break

        phi_hat = phi_new

    return phi_hat
