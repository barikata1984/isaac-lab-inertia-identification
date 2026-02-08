"""Sensor integration module for inertial parameter estimation.

This module provides interfaces for acquiring force/torque and state data
from Isaac Sim simulation or external sources.

Main Components:
    - Data types: SensorData, EstimationData, EstimationResult, WrenchStamped
    - Data buffer: DataBuffer for time-series data management
    - Sensor interfaces: IsaacSimStateCollector, SimulatedForceSensor
    - Utilities: offset compensation functions

Example Usage:
    >>> from iparam_identification.sensor import (
    ...     DataBuffer, SensorData, SimulatedForceSensor
    ... )
    >>> from kinematics import PinocchioKinematics
    >>>
    >>> # Setup
    >>> kin = PinocchioKinematics.for_ur5e()
    >>> phi_true = [1.0, 0, 0, 0.05, 0.01, 0, 0, 0.01, 0, 0.01]  # Known payload
    >>> sensor = SimulatedForceSensor(kin, phi_true)
    >>>
    >>> # Collect data
    >>> buffer = DataBuffer()
    >>> for t, (q, dq, ddq) in enumerate(trajectory):
    ...     wrench = sensor.measure(q, dq, ddq, t * dt)
    ...     data = SensorData(
    ...         timestamp=t * dt, q=q, dq=dq, ddq=ddq,
    ...         force=wrench.force, torque=wrench.torque
    ...     )
    ...     buffer.add_sample(data)
    >>>
    >>> # Prepare for estimation
    >>> A, y = buffer.get_stacked_data(kin)
"""

from .data_types import (
    EstimationData,
    EstimationResult,
    SensorData,
    WrenchStamped,
)
from .data_buffer import DataBuffer
from .contact_sensor import (
    IsaacSimStateCollector,
    InverseDynamicsWrenchSource,
    SimulatedForceSensor,
    WrenchSourceBase,
    apply_offset_compensation,
    compute_offset_compensation_matrix,
)

__all__ = [
    # Data types
    "SensorData",
    "EstimationData",
    "EstimationResult",
    "WrenchStamped",
    # Data buffer
    "DataBuffer",
    # Sensor interfaces
    "IsaacSimStateCollector",
    "InverseDynamicsWrenchSource",
    "SimulatedForceSensor",
    "WrenchSourceBase",
    # Utilities
    "apply_offset_compensation",
    "compute_offset_compensation_matrix",
]
