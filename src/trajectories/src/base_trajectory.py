import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# from factory import InstantiateConfig  # Missing dependency removed


@dataclass(kw_only=True)
class BaseTrajectoryConfig:
    duration: float  # Required, no default
    fps: float  # Required, no default
    module_name: str = "trajectories"

    # CLI-specific arguments (shared across all trajectories)
    show_plot: bool = False  # Show plot window (default: hidden)
    plot_path: Path | None = None  # Path to save plot image
    json_path: Path | None = None  # Path to save trajectory JSON
    config_class: str | None = None  # Config class name (from YAML metadata)


# 抽象クラスの定義
class BaseTrajectory(ABC):
    def __init__(self, cfg: BaseTrajectoryConfig, *args, **kwargs):
        self.duration = cfg.duration
        self.fps = cfg.fps

        self.time_steps = int(self.duration * self.fps)
        self.time_array = np.linspace(0, self.duration, self.time_steps)

    def write_to_json(self, pos, vel, acc, json_path="spline_trajectory.json"):
        """
        Save the trajectory to a JSON file.
        Structure:
        {
            "duration": float,
            "fps": float,
            "frames": [
                {
                    "qpos": [float, ...],
                    "qvel": [float, ...],
                    "qacc": [float, ...]
                },
                ...
            ]
        }
        """
        frames = []
        for i in range(len(self.time_array)):
            # frame = {"qpos": pos[i].tolist(), "qvel": vel[i].tolist(), "qacc": acc[i].tolist()}
            frame = [pos[i].tolist(), vel[i].tolist(), acc[i].tolist()]
            frames.append(frame)

        data = {"duration": self.duration, "fps": self.fps, "frames": frames}

        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

        print(f"Trajectory JSON saved to {json_path}")

    def plot(self, pos, vel, acc, show: bool = False, plot_path: str | None = None):
        """Plot the trajectory."""

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        self._plot_single_ax(axes[0], pos, "Joint Positions", "Time [s]", "Position [rad]")
        self._plot_single_ax(axes[1], vel, "Joint Velocities", "Time [s]", "Velocity [rad/s]")
        self._plot_single_ax(axes[2], acc, "Joint Accelerations", "Time [s]", "Acceleration [rad/s^2]")

        plt.tight_layout()

        if plot_path:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

        if show:
            # Try to use an interactive backend if current one is non-interactive
            if plt.get_backend().lower() == "agg":
                for backend in ["TkAgg", "Qt5Agg", "GTK3Agg"]:
                    try:
                        plt.switch_backend(backend)
                        break
                    except Exception:
                        continue

            if plt.get_backend().lower() == "agg":
                print("\n[WARNING] Cannot show plot: no interactive matplotlib backend found.")
                print("Please specify a file path with '--plot' to save the plot instead.\n")
            else:
                plt.show()

    def _plot_single_ax(self, ax: Axes, data: np.ndarray, title: str, xlabel: str, ylabel: str):
        # Plot Accelerations
        for j, d in enumerate(data.T):
            ax.plot(self.time_array, d, label=f"Joint {j + 1}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    @abstractmethod
    def _generate(self, *args, **kwargs):
        """To be implemented in each child class"""
        pass

    def generate(self, *args, **kwargs):
        show_plot = kwargs.get("show_plot", False)
        plot_path = kwargs.get("plot_path", None)
        json_path = kwargs.get("json_path", None)

        pos, vel, acc = self._generate(args, kwargs)

        if json_path is not None:
            self.write_to_json(pos, vel, acc, json_path)

        self.plot(pos, vel, acc, show=show_plot, plot_path=plot_path)
