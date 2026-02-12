#!/usr/bin/env python3
"""Run excitation trajectory optimization offline.

Optimizes Fourier coefficients to minimize the condition number of
the regressor matrix for inertial parameter identification.
Does NOT require Isaac Sim.

Usage:
    python3 run_optimization.py [--harmonics 5] [--duration 10.0] [--restarts 20]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from models.payloads import CuboidPayload, CylinderPayload, TwoStageCylinderPayload
from models.robots.ur.ur5e import Q0_IDENTIFICATION, URDF_PATH


def main() -> None:
    """Run trajectory optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize excitation trajectory for inertial parameter ID",
    )
    parser.add_argument(
        "--harmonics", type=int, default=5,
        help="Number of Fourier harmonics (default: 5)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Trajectory duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--base-freq", type=float, default=0.1,
        help="Base frequency in Hz (default: 0.1)",
    )
    parser.add_argument(
        "--fps", type=float, default=100.0,
        help="Sampling rate in Hz (default: 100.0)",
    )
    parser.add_argument(
        "--restarts", type=int, default=20,
        help="Number of Monte Carlo restarts (default: 20)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="Max iterations per restart (default: 200)",
    )
    parser.add_argument(
        "--subsample", type=int, default=10,
        help="Subsample factor for objective evaluation (default: 10)",
    )
    parser.add_argument(
        "--max-displacement", type=float, default=0.8,
        help="Max task-space displacement in meters (default: 0.8)",
    )
    parser.add_argument(
        "--box-lower", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Box constraint lower bounds [m] relative to FK(q0), e.g. -0.3 -0.3 -0.2",
    )
    parser.add_argument(
        "--box-upper", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Box constraint upper bounds [m] relative to FK(q0), e.g. 0.3 0.3 0.4",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: auto-select based on box constraint)",
    )
    parser.add_argument(
        "--payload", type=str, default=None,
        choices=["cuboid", "cylinder", "two-stage"],
        help="Payload type for reference (default: None)",
    )
    args = parser.parse_args()

    # Auto-select output filename based on box constraint and payload if not explicitly specified
    if args.output is None:
        base_dir = Path(__file__).parent.parent / "data"
        suffix = ""
        if args.box_lower is not None or args.box_upper is not None:
            suffix += "_box"
        if args.payload is not None:
            suffix += f"_{args.payload.replace('-', '_')}"
        if suffix:
            args.output = str(base_dir / f"optimized_trajectory{suffix}.json")
        else:
            args.output = str(base_dir / "optimized_trajectory.json")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    from kinematics import PinocchioKinematics
    from iparam_identification.excitation_trajectory import (
        ExcitationOptimizer,
        OptimizerConfig,
        WorkspaceConstraintConfig,
    )

    logger.info("=" * 60)
    logger.info("Excitation Trajectory Optimization")
    logger.info("=" * 60)

    # Instantiate payload if specified
    payload = None
    if args.payload:
        if args.payload == "two-stage":
            payload = TwoStageCylinderPayload()
        elif args.payload == "cylinder":
            payload = CylinderPayload()
        else:  # cuboid
            payload = CuboidPayload()
        logger.info(f"Payload: {args.payload}")
        logger.info(f"  Mass: {payload.mass:.3f} kg")
        logger.info(f"  COM offset: {payload.com_offset}")
        logger.info("")

    workspace_cfg = WorkspaceConstraintConfig(
        max_displacement=args.max_displacement,
        box_lower=np.array(args.box_lower) if args.box_lower is not None else None,
        box_upper=np.array(args.box_upper) if args.box_upper is not None else None,
    )

    config = OptimizerConfig(
        num_harmonics=args.harmonics,
        base_freq=args.base_freq,
        duration=args.duration,
        fps=args.fps,
        q0=Q0_IDENTIFICATION,
        workspace=workspace_cfg,
        subsample_factor=args.subsample,
        n_monte_carlo=args.restarts,
        max_iter_per_start=args.max_iter,
        seed=args.seed,
    )

    logger.info(f"  Harmonics: {config.num_harmonics}")
    logger.info(f"  Base freq: {config.base_freq} Hz")
    logger.info(f"  Duration: {config.duration} s")
    logger.info(f"  FPS: {config.fps} Hz")
    logger.info(
        f"  Variables: {2 * config.num_joints * config.num_harmonics} "
        f"({config.num_joints} joints x {config.num_harmonics} harmonics x 2)",
    )
    logger.info(f"  Restarts: {config.n_monte_carlo}")
    logger.info(f"  Max iter: {config.max_iter_per_start}")
    logger.info(f"  Subsample: {config.subsample_factor}")
    logger.info(f"  Max displacement: {config.workspace.max_displacement} m")
    if config.workspace.box_lower is not None:
        logger.info(f"  Box lower: {config.workspace.box_lower.tolist()} m")
        logger.info(f"  Box upper: {config.workspace.box_upper.tolist()} m")
    logger.info(f"  Seed: {config.seed}")
    logger.info("")

    logger.info("Loading kinematics...")
    try:
        kin = PinocchioKinematics.for_ur5e()
    except ImportError:
        logger.info("  ROS2 not available, loading from URDF directly")
        kin = PinocchioKinematics.from_urdf_path(URDF_PATH)

    logger.info("Starting optimization...")
    logger.info("")
    optimizer = ExcitationOptimizer(config, kin)
    result = optimizer.optimize(verbose=True)

    logger.info("")
    logger.info("Validating result...")
    validation = optimizer.validate_trajectory(result)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"  Condition number (subsampled): {result.condition_number:.4f}")
    logger.info(
        f"  Condition number (full): "
        f"{validation['condition_number_full']:.4f}",
    )
    logger.info(f"  Evaluations: {result.n_evaluations}")
    logger.info(f"  Wall time: {result.wall_time:.1f}s")
    logger.info(f"  Best start: {result.best_start_index + 1}")
    logger.info("")
    logger.info("Constraint margins:")
    logger.info(f"  Joint pos lower: {validation['q_margin_lower']:.4f} rad")
    logger.info(f"  Joint pos upper: {validation['q_margin_upper']:.4f} rad")
    logger.info(f"  Joint velocity: {validation['dq_margin']:.4f} rad/s")
    logger.info(f"  Joint accel: {validation['ddq_margin']:.4f} rad/s^2")
    logger.info(
        f"  Tool0 displacement: "
        f"{validation['max_tool0_displacement']:.4f} m",
    )
    logger.info(
        f"  Collision clearance: "
        f"{validation['min_collision_clearance']:.4f} m",
    )
    if "box_margin_lower" in validation:
        logger.info(
            f"  Box margin lower (xyz): "
            f"{[f'{v:.4f}' for v in validation['box_margin_lower']]} m",
        )
        logger.info(
            f"  Box margin upper (xyz): "
            f"{[f'{v:.4f}' for v in validation['box_margin_upper']]} m",
        )
        logger.info(
            f"  Box disp range (xyz): "
            f"{[f'{v:.4f}' for v in validation['box_displacement_min']]} .. "
            f"{[f'{v:.4f}' for v in validation['box_displacement_max']]} m",
        )
    logger.info(
        f"  All satisfied: {validation['all_constraints_satisfied']}",
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "num_harmonics": config.num_harmonics,
            "base_freq": config.base_freq,
            "duration": config.duration,
            "fps": config.fps,
            "q0": config.q0.tolist(),
        },
        "coefficients": {
            "a": result.a_opt.tolist(),
            "b": result.b_opt.tolist(),
        },
        "condition_number": result.condition_number,
        "validation": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in validation.items()
        },
    }

    # Add payload information if specified
    if payload is not None:
        output_data["payload"] = {
            "type": args.payload,
            "mass": payload.mass,
            "com_offset": list(payload.com_offset),
            "phi_true": payload.phi_true.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
