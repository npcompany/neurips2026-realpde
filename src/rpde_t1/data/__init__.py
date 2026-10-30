"""Reading RealPDE trajectories and cutting them into scored windows."""

from .h5io import Trajectory, describe, read_trajectory
from .splits import (
    Params,
    assign_sim_splits,
    assign_splits,
    is_sim_split,
    leave_one_aoa_out,
    parse_params,
    real_trajectories,
    sim_trajectories,
    write_split_config,
)
from .windows import WindowSet, build_window_set, window_starts, windows_from_trajectory

__all__ = [
    "Params",
    "Trajectory",
    "WindowSet",
    "assign_sim_splits",
    "assign_splits",
    "is_sim_split",
    "leave_one_aoa_out",
    "parse_params",
    "real_trajectories",
    "sim_trajectories",
    "write_split_config",
    "build_window_set",
    "describe",
    "read_trajectory",
    "window_starts",
    "windows_from_trajectory",
]
