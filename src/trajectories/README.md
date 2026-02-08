# Trajectory Generation Package

This package provides a CLI tool for generating various robot trajectories (Spline, Fourier Series, and Windowed).

## Features
- **Spline Trajectory**: Generates a 5th-order polynomial trajectory between start and end positions.
- **Fourier Trajectory**: Generates a trajectory based on a finite Fourier series with random or specified coefficients.
- **Windowed Fourier Trajectory**: Combines Fourier series oscillation with a window function to ensure start/end positions are smooth (zero velocity).

## Installation

Ensure you have the required dependencies and build the package:

```bash
cd colcon_ws
colcon build --packages-select trajectory --symlink-install
source install/setup.bash
```

## Usage

The main entry point is the `trajectory_generator` command.

### Verified Commands

#### Fourier Trajectory
```bash
trajectory_generator fourier --duration 5.0 --fps 50.0 --num-joints 6 --num-harmonics 5 --base-freq 0.1 --json-path src/results/fourier.json --show-plot --plot-path src/results/fourier.png
```

#### Window Trajectory
```bash
trajectory_generator window --duration 5.0 --fps 50.0 --num-joints 6 --show-plot --plot-path src/trajectory/results/window.png --json-path src/trajectory/results/window.json
```

#### Windowed Fourier Trajectory
```bash
trajectory_generator windowed_fourier --duration 10.0 --fps 50.0 --num-joints 6 --num-harmonics 5 --base-freq 0.5 --json-path src/trajectory/results/windowed.json --show-plot --plot-path src/trajectory/results/windowed.png
```

#### Spline Trajectory
```bash
trajectory_generator spline --start-pos 0.0 0.0 0.0 0.0 0.0 0.0 --end-pos 1.0 1.0 1.0 1.0 1.0 1.0 --duration 5.0 --fps 50 --show-plot --plot-path src/trajectory/results/spline.png --json-path src/trajectory/results/spline.json
```

## Output
- **JSON**: A JSON file containing the generated positions, velocities, and accelerations for each frame.
- **Plot**: A visual representation of the trajectory saved as an image file.
