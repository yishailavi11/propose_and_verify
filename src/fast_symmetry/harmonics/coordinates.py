"""Convert unit Cartesian directions to the synthesis backends' spherical coordinates."""

import numpy as np


def spherical_locations(directions):
    directions = np.asarray(directions, np.float64)
    return np.ascontiguousarray(
        np.column_stack(
            (
                np.arccos(np.clip(directions[:, 2], -1, 1)),
                np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi),
            )
        )
    )
