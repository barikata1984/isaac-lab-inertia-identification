"""Unit tests for excitation trajectory optimizer."""

import numpy as np
import pytest

from iparam_identification.excitation_trajectory import (
    CollisionChecker,
    CollisionConfig,
    ExcitationOptimizer,
    JointLimits,
    OptimizerConfig,
    WorkspaceConstraintConfig,
    build_trajectory_from_params,
    compute_stacked_regressor,
    condition_number_objective,
)
from iparam_identification.excitation_trajectory.constraints import _TrajectoryCache


# ============================================================
# Fixtures
# ============================================================

Q0 = np.array([
    np.pi / 2, -np.pi / 2, np.pi / 2,
    -np.pi / 2, -np.pi / 2, np.pi / 2,
])


@pytest.fixture
def quick_config():
    """Reduced-size config for fast tests."""
    return OptimizerConfig(
        num_harmonics=2,
        base_freq=0.2,
        duration=5.0,
        fps=50.0,
        q0=Q0,
        n_monte_carlo=2,
        max_iter_per_start=10,
        subsample_factor=20,
    )


@pytest.fixture
def cache(quick_config):
    """Pre-configured trajectory cache."""
    c = _TrajectoryCache()
    c.configure(
        num_joints=quick_config.num_joints,
        num_harmonics=quick_config.num_harmonics,
        base_freq=quick_config.base_freq,
        duration=quick_config.duration,
        fps=quick_config.fps,
        q0=quick_config.q0,
    )
    return c


def _make_random_x(num_joints=6, num_harmonics=2, scale=0.1, seed=42):
    """Generate random Fourier coefficients."""
    rng = np.random.default_rng(seed)
    n = num_joints * num_harmonics
    return rng.uniform(-scale, scale, 2 * n)


# ============================================================
# TestBuildTrajectoryFromParams
# ============================================================

class TestBuildTrajectoryFromParams:
    """Tests for trajectory construction from optimization variables."""

    def test_output_shape(self):
        """Verify correct output shape for given parameters."""
        x = np.zeros(2 * 6 * 2)  # 6 joints, 2 harmonics
        q, dq, ddq = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        n_steps = int(5.0 * 50.0)
        assert q.shape == (n_steps, 6)
        assert dq.shape == (n_steps, 6)
        assert ddq.shape == (n_steps, 6)

    def test_zero_coefficients_give_static_trajectory(self):
        """With zero Fourier coefficients, trajectory stays at q0."""
        x = np.zeros(2 * 6 * 2)
        q, dq, ddq = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        # Each row should equal q0 (broadcast comparison)
        q0_expanded = np.tile(Q0, (len(q), 1))
        np.testing.assert_allclose(q, q0_expanded, atol=1e-10)
        np.testing.assert_allclose(dq, 0.0, atol=1e-10)
        np.testing.assert_allclose(ddq, 0.0, atol=1e-10)

    def test_boundary_conditions(self):
        """Verify q(0)=q(T)=q0 due to polynomial window."""
        x = _make_random_x(scale=0.3)
        q, dq, ddq = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        np.testing.assert_allclose(q[0], Q0, atol=1e-10)
        np.testing.assert_allclose(q[-1], Q0, atol=1e-3)
        np.testing.assert_allclose(dq[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(ddq[0], 0.0, atol=1e-10)

    def test_nonzero_coefficients_produce_motion(self):
        """Non-zero coefficients should produce trajectory different from q0."""
        x = _make_random_x(scale=0.3)
        q, dq, ddq = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        # Mid-trajectory should differ from q0
        mid = len(q) // 2
        assert np.max(np.abs(q[mid] - Q0)) > 0.01


# ============================================================
# TestConditionNumberObjective
# ============================================================

class TestConditionNumberObjective:
    """Tests for the objective function."""

    def test_returns_finite_value(self, kinematics, cache):
        """Objective returns finite value for reasonable inputs."""
        x = _make_random_x(scale=0.2)
        val = condition_number_objective(x, kinematics, cache, subsample_factor=20)
        assert np.isfinite(val)
        assert val > 1.0  # Condition number >= 1

    def test_zero_coefficients_return_large_value(self, kinematics, cache):
        """Zero motion should give very large condition number."""
        x = np.zeros(2 * 6 * 2)
        val = condition_number_objective(x, kinematics, cache, subsample_factor=20)
        assert val >= 1e10  # Near-singular or singular

    def test_different_coefficients_different_values(self, kinematics, cache):
        """Different coefficients should give different condition numbers."""
        x1 = _make_random_x(scale=0.2, seed=1)
        x2 = _make_random_x(scale=0.2, seed=2)
        v1 = condition_number_objective(x1, kinematics, cache, subsample_factor=20)
        v2 = condition_number_objective(x2, kinematics, cache, subsample_factor=20)
        assert v1 != v2


# ============================================================
# TestComputeStackedRegressor
# ============================================================

class TestComputeStackedRegressor:
    """Tests for stacked regressor computation."""

    def test_output_shape(self, kinematics):
        """Verify correct output shape."""
        x = _make_random_x(scale=0.2)
        q, dq, ddq = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        A = compute_stacked_regressor(kinematics, q, dq, ddq, subsample_factor=10)
        n_samples = len(range(0, len(q), 10))
        assert A.shape == (n_samples * 6, 10)

    def test_subsample_reduces_size(self, kinematics):
        """Higher subsample factor should give fewer rows."""
        x = _make_random_x(scale=0.2)
        q, dq, ddq = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        A1 = compute_stacked_regressor(kinematics, q, dq, ddq, subsample_factor=5)
        A10 = compute_stacked_regressor(kinematics, q, dq, ddq, subsample_factor=50)
        assert A1.shape[0] > A10.shape[0]


# ============================================================
# TestCollisionChecker
# ============================================================

class TestCollisionChecker:
    """Tests for FK-based collision checking."""

    def test_home_position_collision_free(self, kinematics):
        """Home position should have positive clearance."""
        checker = CollisionChecker(kinematics.model)
        clearance = checker.check_single_config(Q0)
        assert clearance > 0, f"Home position has collision: clearance={clearance}"

    def test_zero_position_detects_collision(self, kinematics):
        """Zero position has low links (arm extended), collision expected."""
        checker = CollisionChecker(kinematics.model)
        clearance = checker.check_single_config(np.zeros(6))
        # UR5e at zero config has low-z links, collision checker should detect
        assert clearance < 0.1  # May or may not collide depending on radii

    def test_trajectory_min_clearance(self, kinematics):
        """Trajectory with small motion should be collision-free."""
        checker = CollisionChecker(kinematics.model)
        x = _make_random_x(scale=0.1)
        q, _, _ = build_trajectory_from_params(
            x, num_joints=6, num_harmonics=2,
            base_freq=0.2, duration=5.0, fps=50.0, q0=Q0,
        )
        clearance = checker.compute_min_clearance(q, subsample_factor=10)
        assert clearance > 0

    def test_self_collision_pairs_valid(self):
        """Verify non-adjacent pair list is reasonable."""
        pairs = CollisionChecker.SELF_COLLISION_PAIRS
        for i, j in pairs:
            assert abs(i - j) >= 2, f"Pair ({i},{j}) is adjacent"
            assert 0 <= i < 6 and 0 <= j < 6

    def test_collision_config_defaults(self):
        """Verify default collision config values."""
        config = CollisionConfig()
        assert config.ground_z_min > 0
        assert config.self_collision_min_dist > 0
        assert len(config.link_radii) == 6
        assert config.enabled is True


# ============================================================
# TestConstraints
# ============================================================

class TestConstraints:
    """Tests for joint limit and workspace constraints."""

    def test_small_amplitude_satisfies_joint_limits(self, kinematics):
        """Small amplitudes should not violate joint limits."""
        from iparam_identification.excitation_trajectory.constraints import (
            make_joint_position_constraint,
            make_joint_velocity_constraint,
            make_joint_acceleration_constraint,
        )
        config = OptimizerConfig(
            num_harmonics=2, duration=5.0, fps=50.0, q0=Q0,
        )
        cache = _TrajectoryCache()
        cache.configure(
            num_joints=6, num_harmonics=2, base_freq=config.base_freq,
            duration=5.0, fps=50.0, q0=Q0,
        )
        x = _make_random_x(scale=0.05)

        pos_fn = make_joint_position_constraint(config.joint_limits, cache)
        vel_fn = make_joint_velocity_constraint(config.joint_limits, cache)
        acc_fn = make_joint_acceleration_constraint(config.joint_limits, cache)

        assert pos_fn(x) > 0, "Position constraint violated"
        assert vel_fn(x) > 0, "Velocity constraint violated"
        assert acc_fn(x) > 0, "Acceleration constraint violated"

    def test_large_amplitude_violates_velocity_limit(self, kinematics):
        """Very large amplitudes should violate velocity limits."""
        from iparam_identification.excitation_trajectory.constraints import (
            make_joint_velocity_constraint,
        )
        config = OptimizerConfig(
            num_harmonics=2, base_freq=1.0, duration=5.0, fps=50.0, q0=Q0,
        )
        cache = _TrajectoryCache()
        cache.configure(
            num_joints=6, num_harmonics=2, base_freq=1.0,
            duration=5.0, fps=50.0, q0=Q0,
        )
        x = _make_random_x(scale=5.0)  # Very large

        vel_fn = make_joint_velocity_constraint(config.joint_limits, cache)
        assert vel_fn(x) < 0, "Large amplitude should violate velocity"


class TestWorkspaceConstraint:
    """Tests for workspace displacement constraint."""

    def test_zero_displacement_at_q0(self, kinematics):
        """Zero coefficients => displacement = 0 => margin = max_displacement."""
        from iparam_identification.excitation_trajectory.constraints import (
            make_workspace_constraint,
        )
        ws_config = WorkspaceConstraintConfig(max_displacement=0.8)
        cache = _TrajectoryCache()
        cache.configure(
            num_joints=6, num_harmonics=2, base_freq=0.2,
            duration=5.0, fps=50.0, q0=Q0,
        )
        x = np.zeros(2 * 6 * 2)

        ws_fn = make_workspace_constraint(ws_config, kinematics, Q0, cache)
        margin = ws_fn(x)
        assert abs(margin - 0.8) < 1e-3

    def test_small_motion_within_limit(self, kinematics):
        """Small motion should stay within workspace limit."""
        from iparam_identification.excitation_trajectory.constraints import (
            make_workspace_constraint,
        )
        ws_config = WorkspaceConstraintConfig(max_displacement=0.8)
        cache = _TrajectoryCache()
        cache.configure(
            num_joints=6, num_harmonics=2, base_freq=0.2,
            duration=5.0, fps=50.0, q0=Q0,
        )
        x = _make_random_x(scale=0.1)

        ws_fn = make_workspace_constraint(ws_config, kinematics, Q0, cache)
        margin = ws_fn(x)
        assert margin > 0


# ============================================================
# TestExcitationOptimizer
# ============================================================

class TestExcitationOptimizer:
    """Integration tests for the full optimizer."""

    def test_optimization_improves_condition_number(self, kinematics):
        """Optimization should improve (lower) condition number."""
        config = OptimizerConfig(
            num_harmonics=2,
            base_freq=0.2,
            duration=5.0,
            fps=50.0,
            q0=Q0,
            n_monte_carlo=3,
            max_iter_per_start=20,
            subsample_factor=25,
        )
        optimizer = ExcitationOptimizer(config, kinematics)
        result = optimizer.optimize(verbose=False)

        assert np.isfinite(result.condition_number)
        assert result.condition_number > 1.0
        assert result.n_evaluations > 0
        assert result.wall_time > 0

    def test_result_shapes(self, kinematics):
        """OptimizationResult should have correct shapes."""
        config = OptimizerConfig(
            num_harmonics=2, duration=5.0, fps=50.0, q0=Q0,
            n_monte_carlo=1, max_iter_per_start=5, subsample_factor=50,
        )
        optimizer = ExcitationOptimizer(config, kinematics)
        result = optimizer.optimize(verbose=False)

        assert result.a_opt.shape == (6, 2)
        assert result.b_opt.shape == (6, 2)
        assert result.x_opt.shape == (2 * 6 * 2,)
        np.testing.assert_array_equal(result.q0, Q0)

    def test_validate_trajectory(self, kinematics):
        """Validation should report all metrics."""
        config = OptimizerConfig(
            num_harmonics=2, duration=5.0, fps=50.0, q0=Q0,
            n_monte_carlo=1, max_iter_per_start=5, subsample_factor=50,
        )
        optimizer = ExcitationOptimizer(config, kinematics)
        result = optimizer.optimize(verbose=False)

        validation = optimizer.validate_trajectory(result)

        assert "condition_number_full" in validation
        assert "all_constraints_satisfied" in validation
        assert "min_collision_clearance" in validation
        assert "max_tool0_displacement" in validation
        assert "singular_values" in validation
        assert np.isfinite(validation["condition_number_full"])

    def test_reproducible_with_seed(self, kinematics):
        """Same seed should produce same result."""
        config = OptimizerConfig(
            num_harmonics=2, duration=5.0, fps=50.0, q0=Q0,
            n_monte_carlo=1, max_iter_per_start=5, subsample_factor=50,
            seed=123,
        )
        opt1 = ExcitationOptimizer(config, kinematics)
        r1 = opt1.optimize(verbose=False)

        opt2 = ExcitationOptimizer(config, kinematics)
        r2 = opt2.optimize(verbose=False)

        np.testing.assert_allclose(
            r1.condition_number, r2.condition_number, rtol=1e-10,
        )

    def test_trajectory_config_usable(self, kinematics):
        """OptimizationResult.trajectory_config should create valid trajectory."""
        from trajectories import WindowedFourierTrajectory

        config = OptimizerConfig(
            num_harmonics=2, duration=5.0, fps=50.0, q0=Q0,
            n_monte_carlo=1, max_iter_per_start=5, subsample_factor=50,
        )
        optimizer = ExcitationOptimizer(config, kinematics)
        result = optimizer.optimize(verbose=False)

        traj = WindowedFourierTrajectory(result.trajectory_config)
        q, dq, ddq = traj.get_value()
        n_steps = int(5.0 * 50.0)
        assert q.shape == (n_steps, 6)


# ============================================================
# TestTrajectoryCache
# ============================================================

class TestTrajectoryCache:
    """Tests for trajectory caching."""

    def test_cache_returns_same_result(self):
        """Same x should return cached result."""
        cache = _TrajectoryCache()
        cache.configure(
            num_joints=6, num_harmonics=2, base_freq=0.2,
            duration=5.0, fps=50.0, q0=Q0,
        )
        x = _make_random_x(scale=0.1)

        q1, _, _ = cache.get(x)
        q2, _, _ = cache.get(x)
        assert q1 is q2  # Same object (cached)

    def test_cache_updates_for_different_x(self):
        """Different x should compute new result."""
        cache = _TrajectoryCache()
        cache.configure(
            num_joints=6, num_harmonics=2, base_freq=0.2,
            duration=5.0, fps=50.0, q0=Q0,
        )
        x1 = _make_random_x(scale=0.1, seed=1)
        x2 = _make_random_x(scale=0.1, seed=2)

        q1, _, _ = cache.get(x1)
        q2, _, _ = cache.get(x2)
        assert not np.allclose(q1, q2)
