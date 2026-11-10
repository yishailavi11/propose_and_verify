"""Ordered, sign-invariant suppression of duplicate plane proposals."""

import numpy as np


def ordered_plane_nms_indices(
    normals,
    offsets,
    diagonal,
    *,
    angle_degrees: float,
    offset_over_diagonal: float,
) -> np.ndarray:
    """Return retained indices for unoriented planes in their input order.

    The conflict matrix is vectorized, but selection deliberately replays the
    original ordered greedy rule.  Inputs therefore remain ordered by field
    height and the highest candidate in each D1 cluster is retained.
    """
    normal_array = np.asarray(normals, dtype=np.float64)
    offset_array = np.asarray(offsets, dtype=np.float64)
    if normal_array.ndim != 2 or normal_array.shape[0] != offset_array.shape[0]:
        raise ValueError("normals and offsets must have matching leading dimensions")
    if offset_array.ndim != 1:
        raise ValueError("offsets must be a vector")
    if not float(diagonal) > 0:
        raise ValueError("diagonal must be positive")
    if not 0 <= angle_degrees <= 90:
        raise ValueError("angle_degrees must be in [0, 90]")
    if offset_over_diagonal < 0:
        raise ValueError("offset_over_diagonal must be nonnegative")
    if not len(offset_array):
        return np.empty(0, dtype=np.int64)

    dots = np.clip(normal_array @ normal_array.T, -1.0, 1.0)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    conflicts = (np.abs(dots) >= np.cos(np.deg2rad(angle_degrees))) & (
        np.abs(offset_array[:, None] - signs * offset_array[None, :]) / float(diagonal)
        <= offset_over_diagonal
    )
    blocked = np.zeros(len(offset_array), dtype=bool)
    selected = np.zeros(len(offset_array), dtype=bool)
    for index in range(len(offset_array)):
        if not blocked[index]:
            selected[index] = True
            blocked |= conflicts[index]
    return np.flatnonzero(selected)
