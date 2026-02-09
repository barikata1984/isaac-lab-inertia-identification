"""Recursive Least Squares estimation for inertial parameters.

This module implements the Recursive Least Squares (RLS) algorithm
for online parameter estimation as new measurements arrive.

Reference:
    Kubus et al. 2008, Section IV-B, Eq. (17-19).
"""

from typing import Optional

import numpy as np

from sensor.data_types import EstimationResult

from .base_estimator import (
    RecursiveEstimatorBase,
    EstimatorConfig,
    validate_estimation_input,
)


class RecursiveLeastSquares(RecursiveEstimatorBase):
    """Recursive Least Squares (RLS) estimator.

    Updates parameter estimates incrementally using the formula:

        K_k = Σ_k⁻ @ A_k^T @ (Λ + A_k @ Σ_k⁻ @ A_k^T)^{-1}
        φ_k = φ_k⁻ + K_k @ (y_k - A_k @ φ_k⁻)
        Σ_k = (I - K_k @ A_k) @ Σ_k⁻

    where:
        - φ_k: Parameter estimate after k measurements
        - Σ_k: Parameter covariance matrix
        - K_k: Kalman gain
        - Λ: Measurement noise covariance (6×6)

    This formulation is equivalent to Kalman filtering with
    constant parameters (no process noise).
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        initial_covariance: float = 1e6,
        forgetting_factor: float = 1.0,
    ):
        """Initialize RLS estimator.

        Args:
            config: Estimator configuration.
            initial_covariance: Initial diagonal value for parameter
                covariance matrix (larger = less confident in initial guess).
            forgetting_factor: Forgetting factor λ in (0, 1].
                λ = 1: Standard RLS (infinite memory)
                λ < 1: Exponentially decaying memory (for time-varying params)
        """
        super().__init__(config)

        self._initial_covariance = initial_covariance
        self._forgetting_factor = forgetting_factor

        # Parameter covariance matrix Σ
        self._Sigma: np.ndarray = (
            initial_covariance * np.eye(self.config.n_params)
        )

        # Measurement noise covariance Λ
        self._Lambda: np.ndarray = (
            self.config.Lambda if self.config.Lambda is not None
            else np.eye(6)
        )

        self._initialized = True  # RLS doesn't need special initialization

    def initialize(
        self,
        A_init: np.ndarray,
        y_init: np.ndarray,
    ) -> None:
        """Initialize with batch data (optional for RLS).

        For RLS, this simply processes initial samples sequentially.

        Args:
            A_init: Initial regressor matrix (K*6, 10).
            y_init: Initial observation vector (K*6,).
        """
        A_init, y_init, n_samples = validate_estimation_input(
            A_init, y_init, self.config.n_params
        )

        for i in range(n_samples):
            A_k = A_init[i * 6:(i + 1) * 6, :]
            y_k = y_init[i * 6:(i + 1) * 6]
            self.update(A_k, y_k)

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
            Updated parameter estimate φ_k (10,).
        """
        A_k = np.asarray(A_k)
        y_k = np.asarray(y_k).ravel()

        if A_k.shape != (6, self.config.n_params):
            raise ValueError(
                f"A_k must have shape (6, {self.config.n_params}), "
                f"got {A_k.shape}"
            )
        if y_k.shape != (6,):
            raise ValueError(f"y_k must have shape (6,), got {y_k.shape}")

        # Prediction residual (innovation)
        innovation = y_k - A_k @ self._phi_hat

        # Innovation covariance: S = Λ + A @ Σ @ A^T
        S = self._Lambda + A_k @ self._Sigma @ A_k.T

        # Kalman gain: K = Σ @ A^T @ S^{-1}
        K = self._Sigma @ A_k.T @ np.linalg.inv(S)

        # Update parameter estimate
        self._phi_hat = self._phi_hat + K @ innovation

        # Update covariance with forgetting factor
        # Σ = (1/λ) * (I - K @ A) @ Σ
        I = np.eye(self.config.n_params)
        self._Sigma = (1.0 / self._forgetting_factor) * (I - K @ A_k) @ self._Sigma

        # Ensure symmetry (numerical stability)
        self._Sigma = 0.5 * (self._Sigma + self._Sigma.T)

        self._n_updates += 1

        return self._phi_hat.copy()

    def reset(self) -> None:
        """Reset estimator to initial state."""
        self._phi_hat = np.zeros(self.config.n_params)
        self._Sigma = self._initial_covariance * np.eye(self.config.n_params)
        self._n_updates = 0

    @property
    def covariance(self) -> np.ndarray:
        """Current parameter covariance matrix."""
        return self._Sigma.copy()

    @property
    def parameter_std(self) -> np.ndarray:
        """Standard deviation of each parameter estimate."""
        return np.sqrt(np.diag(self._Sigma))

    def get_result(self) -> EstimationResult:
        """Get current estimation result with uncertainty info.

        Returns:
            EstimationResult with current parameter estimate.
        """
        return EstimationResult(
            phi=self._phi_hat,
            condition_number=np.linalg.cond(self._Sigma),
            n_samples=self._n_updates,
        )


class RecursiveInstrumentalVariable(RecursiveEstimatorBase):
    """Recursive Instrumental Variable (RIV) estimator.

    Similar to RLS but uses instrumental variables to handle
    correlated noise in the regressor matrix.

    Reference:
        Kubus et al. 2008, Section IV-C, Eq. (20-22).
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        initial_covariance: float = 1e6,
    ):
        """Initialize RIV estimator.

        Args:
            config: Estimator configuration.
            initial_covariance: Initial covariance diagonal.
        """
        super().__init__(config)

        self._initial_covariance = initial_covariance

        # Parameter covariance
        self._Sigma: np.ndarray = (
            initial_covariance * np.eye(self.config.n_params)
        )

        # Instrument matrix sum: Σ Z^T A
        self._ZtA_sum: np.ndarray = np.zeros(
            (self.config.n_params, self.config.n_params)
        )

        # Instrument-observation sum: Σ Z^T y
        self._Zty_sum: np.ndarray = np.zeros(self.config.n_params)

        # Previous regressor for instrument computation
        self._A_prev: Optional[np.ndarray] = None

        self._initialized = False

    def initialize(
        self,
        A_init: np.ndarray,
        y_init: np.ndarray,
    ) -> None:
        """Initialize with first sample (sets instrument).

        Args:
            A_init: Initial regressor matrix (6, 10) or (K*6, 10).
            y_init: Initial observation vector (6,) or (K*6,).
        """
        A_init = np.asarray(A_init)
        y_init = np.asarray(y_init).ravel()

        if A_init.shape[0] >= 12:  # Multiple samples
            # Use first sample as initial instrument
            self._A_prev = A_init[:6, :].copy()
            # Process remaining samples
            n_samples = A_init.shape[0] // 6
            for i in range(1, n_samples):
                A_k = A_init[i * 6:(i + 1) * 6, :]
                y_k = y_init[i * 6:(i + 1) * 6]
                self.update(A_k, y_k)
        else:
            # Single sample: just store as previous
            self._A_prev = A_init[:6, :].copy() if A_init.shape[0] >= 6 else None

        self._initialized = True

    def update(
        self,
        A_k: np.ndarray,
        y_k: np.ndarray,
    ) -> np.ndarray:
        """Update using instrumental variable method.

        Uses the previous regressor as instrument to reduce bias
        from correlated noise.

        Args:
            A_k: Current regressor matrix (6, 10).
            y_k: Current observation vector (6,).

        Returns:
            Updated parameter estimate.
        """
        A_k = np.asarray(A_k)
        y_k = np.asarray(y_k).ravel()

        if self._A_prev is None:
            # First update: store and return zeros
            self._A_prev = A_k.copy()
            return self._phi_hat.copy()

        # Use previous regressor as instrument
        Z_k = self._A_prev

        # Accumulate sums
        self._ZtA_sum += Z_k.T @ A_k
        self._Zty_sum += Z_k.T @ y_k

        # Solve for parameters: φ = (Z^T A)^{-1} Z^T y
        try:
            self._phi_hat = np.linalg.solve(self._ZtA_sum, self._Zty_sum)
        except np.linalg.LinAlgError:
            # Singular matrix: use pseudoinverse
            self._phi_hat = np.linalg.lstsq(
                self._ZtA_sum, self._Zty_sum, rcond=None
            )[0]

        # Store current as previous for next iteration
        self._A_prev = A_k.copy()
        self._n_updates += 1

        return self._phi_hat.copy()

    def reset(self) -> None:
        """Reset estimator to initial state."""
        self._phi_hat = np.zeros(self.config.n_params)
        self._Sigma = self._initial_covariance * np.eye(self.config.n_params)
        self._ZtA_sum = np.zeros((self.config.n_params, self.config.n_params))
        self._Zty_sum = np.zeros(self.config.n_params)
        self._A_prev = None
        self._n_updates = 0
        self._initialized = False
