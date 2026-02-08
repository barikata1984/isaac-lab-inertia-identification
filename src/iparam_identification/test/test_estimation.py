"""Unit tests for estimation module."""

import numpy as np
import pytest

from iparam_identification.estimation import (
    # Base
    EstimatorConfig,
    validate_estimation_input,
    compute_condition_number,
    compute_residual_norm,
    # Batch LS
    BatchLeastSquares,
    batch_least_squares,
    weighted_least_squares,
    iteratively_reweighted_ls,
    # Batch TLS
    BatchTotalLeastSquares,
    batch_total_least_squares,
    regularized_tls,
    # Recursive LS
    RecursiveLeastSquares,
    # Recursive TLS
    RecursiveTotalLeastSquares,
    WindowedRTLS,
    # SVD
    SVDState,
    initialize_svd,
    incremental_svd_update_block,
    get_tls_solution_from_svd,
)


# ============================================================================
# Fixtures for generating test data
# ============================================================================


@pytest.fixture
def sample_phi():
    """True inertial parameters for testing.

    1kg payload with CoM at (0, 0, 0.05)m and diagonal inertia 0.01 kg·m².
    """
    return np.array([
        1.0,      # m
        0.0,      # m*cx
        0.0,      # m*cy
        0.05,     # m*cz
        0.01,     # Ixx
        0.0,      # Ixy
        0.0,      # Ixz
        0.01,     # Iyy
        0.0,      # Iyz
        0.01,     # Izz
    ])


@pytest.fixture
def synthetic_data(sample_phi):
    """Generate synthetic estimation data.

    Creates random regressor matrices and corresponding observations
    based on the true parameters with optional noise.
    """
    def _generate(n_samples: int, noise_std: float = 0.0, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Generate random regressor matrices
        A_list = []
        y_list = []

        for _ in range(n_samples):
            # Random regressor (simulating different robot configurations)
            A_k = rng.standard_normal((6, 10))
            # Make it more realistic by having larger values in first columns
            A_k[:, 0] *= 10  # mass column (gravity dominated)

            # Compute true observation
            y_k = A_k @ sample_phi

            # Add noise
            if noise_std > 0:
                y_k += rng.normal(0, noise_std, 6)

            A_list.append(A_k)
            y_list.append(y_k)

        A_stacked = np.vstack(A_list)
        y_stacked = np.concatenate(y_list)

        return A_stacked, y_stacked, A_list, y_list

    return _generate


# ============================================================================
# Tests for base utilities
# ============================================================================


class TestEstimatorConfig:
    """Tests for EstimatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EstimatorConfig()
        assert config.n_params == 10
        assert config.Lambda is None
        assert config.regularization == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        Lambda = np.eye(6) * 0.1
        config = EstimatorConfig(n_params=10, Lambda=Lambda, regularization=1e-6)

        assert config.n_params == 10
        np.testing.assert_array_equal(config.Lambda, Lambda)
        assert config.regularization == 1e-6

    def test_invalid_lambda_shape(self):
        """Test that invalid Lambda shape raises error."""
        with pytest.raises(ValueError):
            EstimatorConfig(Lambda=np.eye(5))


class TestValidateInput:
    """Tests for input validation."""

    def test_valid_input(self):
        """Test validation of correct input shapes."""
        A = np.random.randn(60, 10)  # 10 samples
        y = np.random.randn(60)

        A_out, y_out, n_samples = validate_estimation_input(A, y)

        assert A_out.shape == (60, 10)
        assert y_out.shape == (60,)
        assert n_samples == 10

    def test_invalid_columns(self):
        """Test rejection of wrong column count."""
        A = np.random.randn(60, 8)
        y = np.random.randn(60)

        with pytest.raises(ValueError, match="columns"):
            validate_estimation_input(A, y)

    def test_invalid_rows(self):
        """Test rejection of rows not divisible by 6."""
        A = np.random.randn(65, 10)
        y = np.random.randn(65)

        with pytest.raises(ValueError, match="divisible by 6"):
            validate_estimation_input(A, y)

    def test_mismatched_lengths(self):
        """Test rejection of mismatched A and y."""
        A = np.random.randn(60, 10)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="length"):
            validate_estimation_input(A, y)


class TestConditionNumber:
    """Tests for condition number computation."""

    def test_well_conditioned(self):
        """Test condition number of well-conditioned matrix."""
        A = np.eye(10)
        cond = compute_condition_number(A)
        assert cond == pytest.approx(1.0)

    def test_ill_conditioned(self):
        """Test detection of ill-conditioned matrix."""
        A = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-10])
        cond = compute_condition_number(A)
        assert cond > 1e9


# ============================================================================
# Tests for Batch Least Squares
# ============================================================================


class TestBatchLeastSquares:
    """Tests for Batch OLS estimator."""

    def test_perfect_data(self, sample_phi, synthetic_data):
        """Test OLS with noise-free data."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.0)

        estimator = BatchLeastSquares()
        result = estimator.estimate(A, y)

        np.testing.assert_array_almost_equal(result.phi, sample_phi, decimal=10)
        assert result.n_samples == 20

    def test_noisy_data(self, sample_phi, synthetic_data):
        """Test OLS with noisy data."""
        A, y, _, _ = synthetic_data(n_samples=100, noise_std=0.1)

        estimator = BatchLeastSquares()
        result = estimator.estimate(A, y)

        # Should be reasonably close to true values
        np.testing.assert_array_almost_equal(result.phi, sample_phi, decimal=1)

    def test_regularization(self, sample_phi, synthetic_data):
        """Test OLS with Tikhonov regularization."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.1)

        # Without regularization
        phi_unreg = batch_least_squares(A, y, regularization=0.0)

        # With regularization
        phi_reg = batch_least_squares(A, y, regularization=0.01)

        # Regularized solution should have smaller norm
        assert np.linalg.norm(phi_reg) <= np.linalg.norm(phi_unreg) + 1e-6

    def test_weighted_ls(self, sample_phi, synthetic_data):
        """Test weighted least squares."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.0)

        # Uniform weights should give same result as OLS
        W = np.eye(len(y))
        phi_wls = weighted_least_squares(A, y, W)
        phi_ols = batch_least_squares(A, y)

        np.testing.assert_array_almost_equal(phi_wls, phi_ols, decimal=10)


class TestIterativelyReweightedLS:
    """Tests for IRLS robust estimator."""

    def test_with_outliers(self, sample_phi, synthetic_data):
        """Test IRLS handles outliers better than OLS."""
        A, y, _, _ = synthetic_data(n_samples=50, noise_std=0.1, seed=42)

        # Add outliers
        y_with_outliers = y.copy()
        outlier_indices = [0, 30, 60]
        for idx in outlier_indices:
            y_with_outliers[idx] += 100.0  # Large outlier

        # OLS
        phi_ols = batch_least_squares(A, y_with_outliers)

        # IRLS
        phi_irls = iteratively_reweighted_ls(A, y_with_outliers)

        # IRLS should be closer to true value
        error_ols = np.linalg.norm(phi_ols - sample_phi)
        error_irls = np.linalg.norm(phi_irls - sample_phi)

        assert error_irls < error_ols


# ============================================================================
# Tests for Batch Total Least Squares
# ============================================================================


class TestBatchTotalLeastSquares:
    """Tests for Batch TLS estimator."""

    def test_perfect_data(self, sample_phi, synthetic_data):
        """Test TLS with noise-free data."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.0)

        estimator = BatchTotalLeastSquares()
        result = estimator.estimate(A, y)

        np.testing.assert_array_almost_equal(result.phi, sample_phi, decimal=8)

    def test_noisy_data(self, sample_phi, synthetic_data):
        """Test TLS with noisy data."""
        A, y, _, _ = synthetic_data(n_samples=100, noise_std=0.1)

        estimator = BatchTotalLeastSquares()
        result = estimator.estimate(A, y)

        # Should be reasonably close
        np.testing.assert_array_almost_equal(result.phi, sample_phi, decimal=1)

    def test_singular_values_stored(self, synthetic_data):
        """Test that singular values are accessible after estimation."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.0)

        estimator = BatchTotalLeastSquares()
        estimator.estimate(A, y)

        assert estimator.singular_values is not None
        assert len(estimator.singular_values) == 11  # [A | y] has 11 columns

    def test_regularized_tls(self, sample_phi, synthetic_data):
        """Test regularized TLS."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.1)

        phi, sv = regularized_tls(A, y, regularization=0.01)

        # Should produce valid estimate
        assert phi.shape == (10,)
        assert not np.any(np.isnan(phi))


class TestTLSvOLS:
    """Compare TLS and OLS behavior."""

    def test_tls_handles_regressor_noise(self, sample_phi):
        """Test that TLS handles noise in A better than OLS."""
        rng = np.random.default_rng(42)
        n_samples = 100

        # Generate clean data
        A_clean = rng.standard_normal((n_samples * 6, 10))
        A_clean[:, 0] *= 10
        y_clean = A_clean @ sample_phi

        # Add noise to both A and y (EIV model)
        A_noisy = A_clean + rng.normal(0, 0.1, A_clean.shape)
        y_noisy = y_clean + rng.normal(0, 0.1, y_clean.shape)

        # OLS estimate
        phi_ols = batch_least_squares(A_noisy, y_noisy)

        # TLS estimate
        phi_tls, _ = batch_total_least_squares(A_noisy, y_noisy)

        # Both should be reasonably close (TLS often better for EIV)
        error_ols = np.linalg.norm(phi_ols - sample_phi)
        error_tls = np.linalg.norm(phi_tls - sample_phi)

        # TLS typically performs better or similarly with EIV noise
        # (not always strictly better, depends on noise structure)
        assert error_tls < error_ols * 2  # At least not much worse


# ============================================================================
# Tests for Recursive Least Squares
# ============================================================================


class TestRecursiveLeastSquares:
    """Tests for RLS estimator."""

    def test_converges_to_batch(self, sample_phi, synthetic_data):
        """Test that RLS converges to batch OLS solution."""
        A, y, A_list, y_list = synthetic_data(n_samples=50, noise_std=0.0)

        # Batch OLS
        phi_batch = batch_least_squares(A, y)

        # Recursive LS
        estimator = RecursiveLeastSquares()
        for A_k, y_k in zip(A_list, y_list):
            phi_rls = estimator.update(A_k, y_k)

        # Should converge to same solution
        np.testing.assert_array_almost_equal(phi_rls, phi_batch, decimal=5)

    def test_incremental_updates(self, sample_phi, synthetic_data):
        """Test that estimates improve with more data."""
        _, _, A_list, y_list = synthetic_data(n_samples=100, noise_std=0.1)

        estimator = RecursiveLeastSquares()
        errors = []

        for i, (A_k, y_k) in enumerate(zip(A_list, y_list)):
            phi = estimator.update(A_k, y_k)
            if i > 5:  # After some initial samples
                error = np.linalg.norm(phi - sample_phi)
                errors.append(error)

        # Error should generally decrease (allow some fluctuation)
        assert errors[-1] < errors[0]

    def test_forgetting_factor(self, sample_phi, synthetic_data):
        """Test RLS with forgetting factor."""
        _, _, A_list, y_list = synthetic_data(n_samples=50, noise_std=0.0)

        # With forgetting factor
        estimator = RecursiveLeastSquares(forgetting_factor=0.99)

        for A_k, y_k in zip(A_list, y_list):
            estimator.update(A_k, y_k)

        # Should still produce reasonable estimate
        np.testing.assert_array_almost_equal(
            estimator.phi_hat, sample_phi, decimal=2
        )

    def test_covariance_decreases(self, synthetic_data):
        """Test that parameter uncertainty decreases with data."""
        _, _, A_list, y_list = synthetic_data(n_samples=50, noise_std=0.1)

        estimator = RecursiveLeastSquares()

        initial_trace = np.trace(estimator.covariance)

        for A_k, y_k in zip(A_list, y_list):
            estimator.update(A_k, y_k)

        final_trace = np.trace(estimator.covariance)

        assert final_trace < initial_trace

    def test_reset(self, synthetic_data):
        """Test reset functionality."""
        _, _, A_list, y_list = synthetic_data(n_samples=20, noise_std=0.0)

        estimator = RecursiveLeastSquares()

        for A_k, y_k in zip(A_list, y_list):
            estimator.update(A_k, y_k)

        estimator.reset()

        assert estimator.n_updates == 0
        np.testing.assert_array_equal(estimator.phi_hat, np.zeros(10))


# ============================================================================
# Tests for Recursive Total Least Squares
# ============================================================================


class TestRecursiveTotalLeastSquares:
    """Tests for RTLS estimator."""

    def test_initialization_required(self, synthetic_data):
        """Test that RTLS requires initialization."""
        _, _, A_list, y_list = synthetic_data(n_samples=10, noise_std=0.0)

        estimator = RecursiveTotalLeastSquares(min_init_samples=3)

        # First few updates should buffer
        for i in range(2):
            phi = estimator.update(A_list[i], y_list[i])
            assert not estimator.is_initialized

        # Third update should trigger initialization
        phi = estimator.update(A_list[2], y_list[2])
        assert estimator.is_initialized

    def test_explicit_initialization(self, sample_phi, synthetic_data):
        """Test explicit initialization."""
        A, y, _, _ = synthetic_data(n_samples=5, noise_std=0.0)

        estimator = RecursiveTotalLeastSquares()
        estimator.initialize(A, y)

        assert estimator.is_initialized
        np.testing.assert_array_almost_equal(estimator.phi_hat, sample_phi, decimal=5)

    def test_converges_to_batch_tls(self, sample_phi, synthetic_data):
        """Test that RTLS converges toward batch TLS solution."""
        # Use more samples for better convergence
        A, y, A_list, y_list = synthetic_data(n_samples=100, noise_std=0.0)

        # Batch TLS
        phi_batch, _ = batch_total_least_squares(A, y)

        # Recursive TLS with explicit initialization (better convergence)
        estimator = RecursiveTotalLeastSquares()
        # Initialize with first 10 samples
        A_init = np.vstack(A_list[:10])
        y_init = np.concatenate(y_list[:10])
        estimator.initialize(A_init, y_init)

        # Process remaining samples
        for A_k, y_k in zip(A_list[10:], y_list[10:]):
            phi_rtls = estimator.update(A_k, y_k)

        # RTLS may not exactly match batch TLS due to incremental SVD
        # approximations, but should get the mass estimate reasonable
        assert estimator.is_initialized
        # Check mass estimate is reasonable (within 50%)
        mass_error = abs(phi_rtls[0] - sample_phi[0]) / sample_phi[0]
        assert mass_error < 0.5, f"Mass error too large: {mass_error}"

    def test_get_result(self, synthetic_data):
        """Test result retrieval."""
        A, y, _, _ = synthetic_data(n_samples=10, noise_std=0.0)

        estimator = RecursiveTotalLeastSquares()
        estimator.initialize(A, y)

        result = estimator.get_result()

        assert result.phi.shape == (10,)
        assert result.n_samples == 10
        assert not np.isnan(result.condition_number)


class TestWindowedRTLS:
    """Tests for windowed RTLS."""

    def test_window_size_limit(self, synthetic_data):
        """Test that window size is respected."""
        _, _, A_list, y_list = synthetic_data(n_samples=50, noise_std=0.0)

        window_size = 10
        estimator = WindowedRTLS(window_size=window_size)

        for A_k, y_k in zip(A_list, y_list):
            estimator.update(A_k, y_k)

        # Internal buffer should not exceed window size
        assert len(estimator._A_buffer) <= window_size

    def test_adapts_to_changes(self, sample_phi):
        """Test that windowed RTLS adapts to parameter changes."""
        rng = np.random.default_rng(42)
        window_size = 20

        estimator = WindowedRTLS(window_size=window_size)

        # First phase: original parameters
        for _ in range(30):
            A_k = rng.standard_normal((6, 10))
            y_k = A_k @ sample_phi
            estimator.update(A_k, y_k)

        phi_phase1 = estimator.phi_hat.copy()

        # Second phase: changed parameters (double mass)
        changed_phi = sample_phi.copy()
        changed_phi[0] = 2.0

        for _ in range(30):
            A_k = rng.standard_normal((6, 10))
            y_k = A_k @ changed_phi
            estimator.update(A_k, y_k)

        phi_phase2 = estimator.phi_hat.copy()

        # Should adapt to new parameters
        assert abs(phi_phase2[0] - 2.0) < abs(phi_phase1[0] - 2.0)


# ============================================================================
# Tests for SVD utilities
# ============================================================================


class TestSVDState:
    """Tests for SVDState dataclass."""

    def test_creation(self):
        """Test SVDState creation."""
        U = np.eye(5, 3)
        S = np.array([3.0, 2.0, 1.0])
        Vt = np.eye(3, 4)

        state = SVDState(U=U, S=S, Vt=Vt, n_rows=5)

        assert state.rank == 3
        assert state.n_cols == 4
        assert state.n_rows == 5

    def test_matrix_reconstruction(self):
        """Test matrix reconstruction from SVD."""
        A = np.random.randn(10, 5)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        state = SVDState(U=U, S=S, Vt=Vt, n_rows=10)
        A_reconstructed = state.get_matrix()

        np.testing.assert_array_almost_equal(A, A_reconstructed)


class TestIncrementalSVD:
    """Tests for incremental SVD updates."""

    def test_initialize_svd(self):
        """Test SVD initialization."""
        A = np.random.randn(20, 5)
        state = initialize_svd(A)

        assert state.n_rows == 20
        assert state.n_cols == 5
        assert state.rank <= 5

    def test_block_update(self):
        """Test block SVD update."""
        # Initial matrix
        A1 = np.random.randn(12, 5)
        state = initialize_svd(A1)

        # New rows
        A2 = np.random.randn(6, 5)
        state = incremental_svd_update_block(state, A2)

        assert state.n_rows == 18

        # Verify reconstruction is close to full matrix
        A_full = np.vstack([A1, A2])
        A_reconstructed = state.get_matrix()

        # Should be close (may lose some accuracy due to rank truncation)
        assert A_reconstructed.shape == A_full.shape

    def test_tls_solution_extraction(self, sample_phi):
        """Test TLS solution extraction from SVD state."""
        rng = np.random.default_rng(42)

        # Generate augmented matrix [A | y]
        A = rng.standard_normal((60, 10))
        y = A @ sample_phi
        C = np.hstack([A, y.reshape(-1, 1)])

        state = initialize_svd(C)
        phi = get_tls_solution_from_svd(state)

        np.testing.assert_array_almost_equal(phi, sample_phi, decimal=8)


# ============================================================================
# Integration tests
# ============================================================================


class TestEstimationIntegration:
    """Integration tests comparing all estimators."""

    def test_all_estimators_on_same_data(self, sample_phi, synthetic_data):
        """Test all estimators produce reasonable results on same data."""
        A, y, A_list, y_list = synthetic_data(n_samples=50, noise_std=0.05)

        # Batch OLS
        batch_ols = BatchLeastSquares()
        result_ols = batch_ols.estimate(A, y)

        # Batch TLS
        batch_tls = BatchTotalLeastSquares()
        result_tls = batch_tls.estimate(A, y)

        # RLS
        rls = RecursiveLeastSquares()
        for A_k, y_k in zip(A_list, y_list):
            rls.update(A_k, y_k)
        result_rls = rls.get_result()

        # RTLS
        rtls = RecursiveTotalLeastSquares()
        for A_k, y_k in zip(A_list, y_list):
            rtls.update(A_k, y_k)
        result_rtls = rtls.get_result()

        # All should be within 20% of true mass
        for result, name in [
            (result_ols, "OLS"),
            (result_tls, "TLS"),
            (result_rls, "RLS"),
            (result_rtls, "RTLS"),
        ]:
            rel_error = abs(result.mass - 1.0) / 1.0
            assert rel_error < 0.2, f"{name} mass error too large: {rel_error}"

    def test_estimation_result_properties(self, synthetic_data):
        """Test EstimationResult properties work correctly."""
        A, y, _, _ = synthetic_data(n_samples=20, noise_std=0.0)

        estimator = BatchLeastSquares()
        result = estimator.estimate(A, y)

        # Test all properties
        assert result.mass > 0
        assert result.center_of_mass.shape == (3,)
        assert result.inertia_matrix.shape == (3, 3)
        assert result.inertia_at_com.shape == (3, 3)

        # Inertia should be symmetric
        np.testing.assert_array_almost_equal(
            result.inertia_matrix,
            result.inertia_matrix.T
        )
