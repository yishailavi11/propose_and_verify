"""The single frozen detector configuration; thresholds are never fitted at test time."""

from dataclasses import dataclass

from .geometry.nms import ordered_plane_nms_indices
from .method import THRESHOLD


@dataclass(frozen=True)
class AngularConfig:
    pair_count: int = 250000
    prepared_point_count: int = 100000
    pair_seed: int = 20260929
    witnesses: int = 2
    witness_power: int = 2
    witness_tau_over_diagonal: float = 0.01
    witness_grid_resolution: int = 96
    nms_angle_degrees: float = 6.0
    nms_offset_over_diagonal: float = 0.01
    acceptance_threshold: float = THRESHOLD

    def d1_indices(self, normals, offsets, diagonal):
        return ordered_plane_nms_indices(
            normals, offsets, diagonal, angle_degrees=6.0, offset_over_diagonal=0.01
        )
