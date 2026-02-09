"""Recursive Total Least Squares estimation for inertial parameters.

This module implements the Recursive Total Least Squares (RTLS) algorithm,
the main contribution of Kubus et al. 2008. It combines the advantages of:
- TLS: Accounts for errors in both regressor and observations
- Recursive: Online estimation without storing all data

Reference:
    Kubus et al. 2008, "On-line estimation of inertial parameters using
    a recursive total least-squares approach", Section IV-D.
"""

from typing import Optional

import numpy as np

from sensor.data_types import EstimationResult

from .base_estimator import (
    RecursiveEstimatorBase,
    EstimatorConfig,
    validate_estimation_input,
)
from .svd_update import (
    SVDState,
    initialize_svd,
    incremental_svd_update_block,
    get_tls_solution_from_svd,
)


class RecursiveTotalLeastSquares(RecursiveEstimatorBase):
    """Recursive Total Least Squares (RTLS) estimator.

    Uses incremental SVD updating to maintain an online TLS solution
    as new measurements arrive.

    The algorithm:
    1. Initialize with 2-3 samples to compute initial SVD of [A | y]
    2. For each new sample, update SVD incrementally
    3. Extract TLS solution from updated SVD

    Advantages over RLS:
    - Accounts for errors in regressor matrix A (from acceleration noise)
    - More accurate estimates, especially with sensor noise

    Advantages over batch TLS:
    - Online estimation (no need to store all data)
    - Constant memory usage
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        max_rank: Optional[int] = None,
        min_init_samples: int = 2,
        eps: float = 1e-12,
    ):
        """Initialize RTLS estimator.

        Args:
            config: Estimator configuration.
            max_rank: Maximum SVD rank to maintain. If None, uses n_params + 1.
                Larger values retain more information but use more memory.
            min_init_samples: Minimum samples needed for initialization.
            eps: Numerical threshold for SVD truncation.
        """
        super().__init__(config)

        self._max_rank = max_rank or (self.config.n_params + 1)
        self._min_init_samples = min_init_samples
        self._eps = eps

        # SVD state of augmented matrix [A | y]
        self._svd_state: Optional[SVDState] = None

        # Initialization buffer
        self._init_buffer_A: list[np.ndarray] = []
        self._init_buffer_y: list[np.ndarray] = []

    def initialize(
        self,
        A_init: np.ndarray,
        y_init: np.ndarray,
    ) -> None:
        """Initialize RTLS with initial data.

        Computes the initial SVD of the augmented matrix [A | y].
        Requires at least min_init_samples to proceed.

        Args:
            A_init: Initial regressor matrix (K*6, 10).
            y_init: Initial observation vector (K*6,).
        """
        A_init, y_init, n_samples = validate_estimation_input(
            A_init, y_init, self.config.n_params
        )

        if n_samples < self._min_init_samples:
            raise ValueError(
                f"RTLS initialization requires at least {self._min_init_samples} "
                f"samples, got {n_samples}"
            )

        # Form augmented matrix [A | y]
        y_col = y_init.reshape(-1, 1)
        C = np.hstack([A_init, y_col])

        # Initialize SVD
        self._svd_state = initialize_svd(
            C,
            max_rank=self._max_rank,
            eps=self._eps,
        )

        # Extract initial TLS solution
        try:
            self._phi_hat = get_tls_solution_from_svd(self._svd_state)
        except ValueError:
            # Fall back to least squares if TLS fails
            self._phi_hat = np.linalg.lstsq(A_init, y_init, rcond=None)[0]

        self._n_updates = n_samples
        self._initialized = True

    def update(
        self,
        A_k: np.ndarray,
        y_k: np.ndarray,
    ) -> np.ndarray:
        """Process one measurement and update TLS estimate.

        If not yet initialized, buffers the data until enough
        samples are available.

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

        if not self._initialized:
            # Buffer data for initialization
            self._init_buffer_A.append(A_k)
            self._init_buffer_y.append(y_k)

            if len(self._init_buffer_A) >= self._min_init_samples:
                # Initialize with buffered data
                A_stacked = np.vstack(self._init_buffer_A)
                y_stacked = np.concatenate(self._init_buffer_y)
                self.initialize(A_stacked, y_stacked)
                self._init_buffer_A = []
                self._init_buffer_y = []

            return self._phi_hat.copy()

        # Form augmented row [A_k | y_k] - this is 6 rows
        y_col = y_k.reshape(-1, 1)
        new_rows = np.hstack([A_k, y_col])  # (6, 11)

        # Update SVD incrementally
        self._svd_state = incremental_svd_update_block(
            self._svd_state,
            new_rows,
            max_rank=self._max_rank,
            eps=self._eps,
        )

        # Extract TLS solution
        try:
            self._phi_hat = get_tls_solution_from_svd(self._svd_state)
        except ValueError:
            # TLS solution undefined - keep previous estimate
            pass

        self._n_updates += 1

        return self._phi_hat.copy()

    def reset(self) -> None:
        """Reset estimator to initial state."""
        self._phi_hat = np.zeros(self.config.n_params)
        self._svd_state = None
        self._init_buffer_A = []
        self._init_buffer_y = []
        self._n_updates = 0
        self._initialized = False

    @property
    def svd_state(self) -> Optional[SVDState]:
        """Current SVD state (for debugging/analysis)."""
        return self._svd_state

    @property
    def singular_values(self) -> Optional[np.ndarray]:
        """Current singular values."""
        if self._svd_state is not None:
            return self._svd_state.S.copy()
        return None

    def get_result(self) -> EstimationResult:
        """Get current estimation result.

        Returns:
            EstimationResult with current TLS estimate.
        """
        # Condition number from singular values
        if self._svd_state is not None and len(self._svd_state.S) > 1:
            cond = self._svd_state.S[0] / max(self._svd_state.S[-1], 1e-15)
        else:
            cond = np.nan

        # Residual: smallest singular value (TLS residual)
        if self._svd_state is not None:
            residual = float(self._svd_state.S[-1])
        else:
            residual = np.nan

        return EstimationResult(
            phi=self._phi_hat,
            condition_number=cond,
            residual_norm=residual,
            n_samples=self._n_updates,
        )


class WindowedRTLS(RecursiveEstimatorBase):
    """Sliding window RTLS for time-varying parameters.

    Maintains a fixed-size window of recent measurements,
    computing TLS on the most recent data only.

    Useful when inertial parameters may change over time
    (e.g., liquid sloshing, flexible payloads).
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        window_size: int = 100,
        eps: float = 1e-12,
    ):
        """Initialize windowed RTLS.

        Args:
            config: Estimator configuration.
            window_size: Number of samples in the sliding window.
            eps: Numerical threshold.
        """
        super().__init__(config)

        self._window_size = window_size
        self._eps = eps

        # Circular buffer for windowed data
        self._A_buffer: list[np.ndarray] = []
        self._y_buffer: list[np.ndarray] = []

    def initialize(
        self,
        A_init: np.ndarray,
        y_init: np.ndarray,
    ) -> None:
        """Initialize with initial data.

        Args:
            A_init: Initial regressor matrix (K*6, 10).
            y_init: Initial observation vector (K*6,).
        """
        A_init, y_init, n_samples = validate_estimation_input(
            A_init, y_init, self.config.n_params
        )

        # Store samples in buffer
        for i in range(n_samples):
            self._A_buffer.append(A_init[i * 6:(i + 1) * 6, :])
            self._y_buffer.append(y_init[i * 6:(i + 1) * 6])

        # Trim to window size
        while len(self._A_buffer) > self._window_size:
            self._A_buffer.pop(0)
            self._y_buffer.pop(0)

        # Compute TLS on current window
        self._update_estimate()

        self._n_updates = len(self._A_buffer)
        self._initialized = True

    def update(
        self,
        A_k: np.ndarray,
        y_k: np.ndarray,
    ) -> np.ndarray:
        """Add new sample and update estimate.

        Args:
            A_k: Regressor matrix (6, 10).
            y_k: Observation vector (6,).

        Returns:
            Updated parameter estimate.
        """
        A_k = np.asarray(A_k)
        y_k = np.asarray(y_k).ravel()

        # Add to buffer
        self._A_buffer.append(A_k)
        self._y_buffer.append(y_k)

        # Remove oldest if window full
        if len(self._A_buffer) > self._window_size:
            self._A_buffer.pop(0)
            self._y_buffer.pop(0)

        # Recompute TLS on current window
        if len(self._A_buffer) >= 2:
            self._update_estimate()

        self._n_updates += 1
        self._initialized = True

        return self._phi_hat.copy()

    def _update_estimate(self) -> None:
        """Recompute TLS estimate from current window."""
        if len(self._A_buffer) < 2:
            return

        A_stacked = np.vstack(self._A_buffer)
        y_stacked = np.concatenate(self._y_buffer)

        # Form augmented matrix and compute SVD
        y_col = y_stacked.reshape(-1, 1)
        C = np.hstack([A_stacked, y_col])

        try:
            U, S, Vt = np.linalg.svd(C, full_matrices=True)
            v = Vt[-1, :]

            if np.abs(v[-1]) > self._eps:
                self._phi_hat = -v[:-1] / v[-1]
        except np.linalg.LinAlgError:
            pass

    def reset(self) -> None:
        """Reset estimator."""
        self._phi_hat = np.zeros(self.config.n_params)
        self._A_buffer = []
        self._y_buffer = []
        self._n_updates = 0
        self._initialized = False


class AdaptiveRTLS(RecursiveTotalLeastSquares):
    """Adaptive RTLS with automatic rank selection.

    Automatically adjusts the SVD rank based on the significance
    of singular values, balancing accuracy and computational cost.
    """

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        rank_threshold: float = 0.01,
        min_rank: int = 5,
        max_rank: int = 20,
    ):
        """Initialize adaptive RTLS.

        Args:
            config: Estimator configuration.
            rank_threshold: Threshold for singular value significance
                (relative to largest singular value).
            min_rank: Minimum rank to maintain.
            max_rank: Maximum rank to maintain.
        """
        super().__init__(config, max_rank=max_rank)

        self._rank_threshold = rank_threshold
        self._min_rank = min_rank
        self._adaptive_rank = max_rank

    def update(
        self,
        A_k: np.ndarray,
        y_k: np.ndarray,
    ) -> np.ndarray:
        """Update with adaptive rank selection.

        Args:
            A_k: Regressor matrix (6, 10).
            y_k: Observation vector (6,).

        Returns:
            Updated parameter estimate.
        """
        result = super().update(A_k, y_k)

        # Adapt rank based on singular value spectrum
        if self._svd_state is not None and len(self._svd_state.S) > 0:
            S = self._svd_state.S
            threshold = self._rank_threshold * S[0]
            significant = np.sum(S > threshold)
            self._adaptive_rank = max(self._min_rank, significant)

        return result

    @property
    def current_rank(self) -> int:
        """Current adaptive rank."""
        return self._adaptive_rank
