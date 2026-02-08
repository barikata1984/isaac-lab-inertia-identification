"""Base classes for inertial parameter estimators.

This module provides abstract base classes that define the interface
for parameter estimation algorithms (both batch and recursive).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from iparam_identification.sensor.data_types import EstimationResult


@dataclass
class EstimatorConfig:
    """Configuration for parameter estimators.

    Attributes:
        n_params: Number of parameters to estimate (default 10).
        Lambda: Measurement noise covariance matrix (6, 6).
            If None, identity matrix is used.
        regularization: Tikhonov regularization parameter (default 0.0).
    """

    n_params: int = 10
    Lambda: Optional[np.ndarray] = None
    regularization: float = 0.0

    def __post_init__(self):
        if self.Lambda is not None:
            self.Lambda = np.asarray(self.Lambda)
            if self.Lambda.shape != (6, 6):
                raise ValueError(
                    f"Lambda must have shape (6, 6), got {self.Lambda.shape}"
                )


class BatchEstimatorBase(ABC):
    """Abstract base class for batch parameter estimators.

    Batch estimators process all data at once to compute parameter estimates.
    They are typically more accurate but require all data to be collected
    before estimation.
    """

    def __init__(self, config: Optional[EstimatorConfig] = None):
        """Initialize batch estimator.

        Args:
            config: Estimator configuration.
        """
        self.config = config or EstimatorConfig()
        self._result: Optional[EstimationResult] = None

    @abstractmethod
    def estimate(
        self,
        A: np.ndarray,
        y: np.ndarray,
    ) -> EstimationResult:
        """Estimate parameters from batch data.

        Args:
            A: Stacked regressor matrix (N*6, 10).
            y: Stacked observation vector (N*6,).

        Returns:
            EstimationResult containing estimated parameters.
        """
        pass

    @property
    def result(self) -> Optional[EstimationResult]:
        """Get the most recent estimation result."""
        return self._result


class RecursiveEstimatorBase(ABC):
    """Abstract base class for recursive parameter estimators.

    Recursive estimators update parameter estimates incrementally
    as new data arrives, suitable for online estimation.
    """

    def __init__(self, config: Optional[EstimatorConfig] = None):
        """Initialize recursive estimator.

        Args:
            config: Estimator configuration.
        """
        self.config = config or EstimatorConfig()
        self._phi_hat: np.ndarray = np.zeros(self.config.n_params)
        self._n_updates: int = 0
        self._initialized: bool = False

    @abstractmethod
    def initialize(
        self,
        A_init: np.ndarray,
        y_init: np.ndarray,
    ) -> None:
        """Initialize estimator with initial data.

        Some recursive algorithms (e.g., RTLS) require initial data
        to compute initial estimates before incremental updates.

        Args:
            A_init: Initial regressor matrix (K*6, 10).
            y_init: Initial observation vector (K*6,).
        """
        pass

    @abstractmethod
    def update(
        self,
        A_k: np.ndarray,
        y_k: np.ndarray,
    ) -> np.ndarray:
        """Process one measurement and update parameter estimate.

        Args:
            A_k: Regressor matrix for current timestep (6, 10).
            y_k: Observation vector for current timestep (6,).

        Returns:
            Updated parameter estimate (n_params,).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset estimator to initial state."""
        pass

    @property
    def phi_hat(self) -> np.ndarray:
        """Current parameter estimate."""
        return self._phi_hat.copy()

    @property
    def n_updates(self) -> int:
        """Number of updates performed."""
        return self._n_updates

    @property
    def is_initialized(self) -> bool:
        """Whether the estimator has been initialized."""
        return self._initialized

    def get_result(self) -> EstimationResult:
        """Get current estimation result.

        Returns:
            EstimationResult with current parameter estimate.
        """
        return EstimationResult(
            phi=self._phi_hat,
            n_samples=self._n_updates,
        )


def validate_estimation_input(
    A: np.ndarray,
    y: np.ndarray,
    n_params: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate and reshape input data for estimation.

    Args:
        A: Regressor matrix (N*6, 10) or (6, 10).
        y: Observation vector (N*6,) or (6,).
        n_params: Expected number of parameters.

    Returns:
        Tuple of (A, y, n_samples) with validated shapes.

    Raises:
        ValueError: If shapes are invalid.
    """
    A = np.asarray(A)
    y = np.asarray(y).ravel()

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got {A.ndim}D")

    if A.shape[1] != n_params:
        raise ValueError(
            f"A must have {n_params} columns, got {A.shape[1]}"
        )

    if A.shape[0] % 6 != 0:
        raise ValueError(
            f"A must have rows divisible by 6, got {A.shape[0]}"
        )

    n_samples = A.shape[0] // 6

    if y.shape[0] != A.shape[0]:
        raise ValueError(
            f"y length ({y.shape[0]}) must match A rows ({A.shape[0]})"
        )

    return A, y, n_samples


def compute_condition_number(A: np.ndarray) -> float:
    """Compute condition number of the regressor matrix.

    Uses correlation matrix Υ = A^T A for condition number computation
    as suggested in Kubus et al. 2008.

    Args:
        A: Regressor matrix (N*6, 10).

    Returns:
        Condition number (ratio of largest to smallest singular value).
    """
    try:
        singular_values = np.linalg.svd(A, compute_uv=False)
        if singular_values[-1] < 1e-15:
            return np.inf
        return singular_values[0] / singular_values[-1]
    except np.linalg.LinAlgError:
        return np.inf


def compute_residual_norm(
    A: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
) -> float:
    """Compute residual norm ||y - A @ phi||.

    Args:
        A: Regressor matrix.
        y: Observation vector.
        phi: Parameter estimate.

    Returns:
        Euclidean norm of residuals.
    """
    residual = y - A @ phi
    return float(np.linalg.norm(residual))
