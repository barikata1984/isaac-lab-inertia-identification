import os
from typing import Annotated, Union

import tyro
from trajectories import (
    FourierTrajectory,
    FourierTrajectoryConfig,
    SplineTrajectory,
    SplineTrajectoryConfig,
    WindowTrajectory,
    WindowTrajectoryConfig,
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
    WindowedFourierSplineTrajectory,
    WindowedFourierSplineTrajectoryConfig,
)



ConfigType = Union[
    Annotated[SplineTrajectoryConfig, tyro.conf.subcommand(name="spline")],
    Annotated[FourierTrajectoryConfig, tyro.conf.subcommand(name="fourier")],
    Annotated[WindowTrajectoryConfig, tyro.conf.subcommand(name="window")],
    Annotated[WindowedFourierTrajectoryConfig, tyro.conf.subcommand(name="windowed_fourier")],
    Annotated[WindowedFourierSplineTrajectoryConfig, tyro.conf.subcommand(name="windowed_fourier_spline")],
]


def main(config: ConfigType) -> None:
    """Generate trajectories for Isaac Sim.

    Args:
        config: Trajectory configuration.
    """
    # Ensure output directory exists if json_path is specified
    if config.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.json_path)), exist_ok=True)
    if config.plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.plot_path)), exist_ok=True)

    traj = None
    if isinstance(config, SplineTrajectoryConfig):
        traj = SplineTrajectory(config)
    elif isinstance(config, FourierTrajectoryConfig):
        traj = FourierTrajectory(config)
        # Initialize with some random coefficients for demo if not provided
        # Note: The original CLI did this specifically for fourier if passed via CLI args
        # But here we are getting a defined config.
        # If the user provides coefficients via CLI (unlikely complex dict), we use them.
        # If not, the class __init__ handles defaults (zeros).
        # The original code did:
        # traj.a = np.random.uniform(-0.1, 0.1, (args.joints, args.harmonics))
        # We can implement a similar logic if needed, but let's stick to base behavior or maybe add a flag?
        # For now, let's keep it simple. The original code's random init might have been for quick demo.
        # Let's rely on the config.
        pass
    elif isinstance(config, WindowTrajectoryConfig):
        traj = WindowTrajectory(config)
    elif isinstance(config, WindowedFourierTrajectoryConfig):
        traj = WindowedFourierTrajectory(config)
    elif isinstance(config, WindowedFourierSplineTrajectoryConfig):
        traj = WindowedFourierSplineTrajectory(config)

    if traj:
        print(f"Generating {config.__class__.__name__}...")
        # The generate method in BaseTrajectory takes kwargs for show_plot etc,
        # but the modifications in BaseTrajectoryConfig moved these to the config object?
        # Let's check BaseTrajectory again.
        # BaseTrajectoryConfig has show_plot, plot_path, json_path.
        # BaseTrajectory.generate takes *args, **kwargs and looks for show_plot, plot_path, json_path.

        # We should pass these from config to generate()
        traj.generate(
            show_plot=config.show_plot,
            plot_path=config.plot_path,
            json_path=config.json_path,
        )


def entry_point() -> None:
    config = tyro.cli(ConfigType)
    main(config)


if __name__ == "__main__":
    entry_point()
