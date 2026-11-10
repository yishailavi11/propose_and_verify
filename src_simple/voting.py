"""Paper §8.1–8.3: pair planes → witness weights → angular mass histogram."""

from itertools import product

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss

from .constants import (
    DIRECTION_NORM_EPSILON,
    LATITUDE_COUNT,
    LONGITUDE_COUNT,
    MASS_NORMALIZATION_EPSILON,
    MIN_VOXEL_STEP,
    OFFSET_GATE_OVER_DIAGONAL,
    PAIR_COUNT,
    PAIR_NORMAL_SQUARED_EPSILON,
    VOTE_DIRECTION_SHIFT,
    WITNESS_COUNT,
    WITNESS_GRID_RESOLUTION,
    WITNESS_SIGMA_OVER_DIAGONAL,
    WITNESS_WEIGHT_POWER,
)

MU_NODES = leggauss(LATITUDE_COUNT)[0]
NEIGHBOR_OFFSETS = np.array(list(product((-1, 0, 1), repeat=3)), dtype=np.int32)


def pair_planes(points, samples):
    """Perpendicular bisectors: n=(b-a)/|b-a|, offset=n·(a+b)/2."""
    first, second = points[samples[:, 0]], points[samples[:, 1]]
    difference = second - first
    normals = (
        difference
        / np.sqrt(np.sum(difference**2, axis=1) + PAIR_NORMAL_SQUARED_EPSILON)[:, None]
    )
    offsets = np.sum(normals * (first + second) / 2, axis=1)
    return normals, offsets


def occupancy_grid(points):
    """Build a 96³ Boolean occupancy array and three tables of voxel centers."""
    lower = points.min(axis=0)
    step = np.maximum(np.ptp(points, axis=0) / WITNESS_GRID_RESOLUTION, MIN_VOXEL_STEP)
    cells = np.clip(
        np.floor((points - lower) / step).astype(np.int32),
        0,
        WITNESS_GRID_RESOLUTION - 1,
    )
    occupied = np.zeros((WITNESS_GRID_RESOLUTION,) * 3, dtype=bool)
    occupied[tuple(cells.T)] = True
    centers = (
        lower[:, None]
        + (np.arange(WITNESS_GRID_RESOLUTION, dtype=points.dtype) + 0.5) * step[:, None]
    )
    return lower, step, occupied, centers


@njit(cache=True, fastmath=False)
def local_squared_distances(queries, grid):
    """Check 27 cells per query with an ordinary, compiled Python loop.

    Numba only removes interpreter overhead. fastmath=False preserves the
    float32 operation order; the body also runs as ordinary Python via py_func.
    """
    lower, step, occupied, centers = grid
    best = np.full(len(queries), np.inf, dtype=queries.dtype)
    for i in range(len(queries)):
        cx = int(np.floor((queries[i, 0] - lower[0]) / step[0]))
        cy = int(np.floor((queries[i, 1] - lower[1]) / step[1]))
        cz = int(np.floor((queries[i, 2] - lower[2]) / step[2]))
        for shift in NEIGHBOR_OFFSETS:
            x, y, z = cx + shift[0], cy + shift[1], cz + shift[2]
            if not (
                0 <= x < WITNESS_GRID_RESOLUTION
                and 0 <= y < WITNESS_GRID_RESOLUTION
                and 0 <= z < WITNESS_GRID_RESOLUTION
            ):
                continue
            if not occupied[x, y, z]:
                continue
            dx = queries[i, 0] - centers[0, x]
            dy = queries[i, 1] - centers[1, y]
            dz = queries[i, 2] - centers[2, z]
            squared = dx * dx + dy * dy + dz * dz
            best[i] = min(best[i], squared)
    return best


def witness_weights(points, samples, normals, offsets, grid):
    """Reflect both witnesses; square their mean Gaussian compatibility."""
    witnesses = points[samples[:, 2:]]
    signed = np.sum(witnesses * normals[:, None], axis=-1) - offsets[:, None]
    reflected = witnesses - 2 * signed[..., None] * normals[:, None]
    squared = local_squared_distances(reflected.reshape(-1, 3), grid)
    compatibility = np.exp(
        -squared.reshape(-1, WITNESS_COUNT) / (2 * WITNESS_SIGMA_OVER_DIAGONAL**2)
    )
    return compatibility.mean(axis=1) ** WITNESS_WEIGHT_POWER


def deposit_votes(histogram, normals, offsets, weights):
    """Cloud-in-cell: distribute each mass to four (mu,phi) grid neighbors.

    Sign-canonicalize by the pair offset. Even harmonic degrees make the field
    antipodally symmetric later; do not deposit an extra antipodal vote.
    """
    sign = np.where(offsets < 0, -1.0, 1.0).astype(normals.dtype)
    directions = (
        normals * sign[:, None] * (np.abs(offsets) + VOTE_DIRECTION_SHIFT)[:, None]
    )
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), DIRECTION_NORM_EPSILON
    )
    mu = directions[:, 2]
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi)
    nodes = MU_NODES
    upper = np.clip(np.searchsorted(nodes, mu, side="right"), 1, LATITUDE_COUNT - 1)
    latitude_weight = np.clip(
        (mu - nodes[upper - 1]) / (nodes[upper] - nodes[upper - 1]), 0, 1
    )
    longitude = phi * LONGITUDE_COUNT / (2 * np.pi)
    left = np.floor(longitude).astype(int)
    longitude_weight = longitude - left
    # Floating-point mod can round an angle just below 2pi to 2pi.
    left %= LONGITUDE_COUNT
    for row, row_weight in [(upper - 1, 1 - latitude_weight), (upper, latitude_weight)]:
        for col, col_weight in [
            (left, 1 - longitude_weight),
            ((left + 1) % LONGITUDE_COUNT, longitude_weight),
        ]:
            np.add.at(histogram, (row, col), weights * row_weight * col_weight)


def angular_histogram(points, samples):
    """Accumulate normalized voting mass on the fixed 50×193 angular grid.

    Gate all pairs before witness lookup, then normalize the total mass.
    """
    samples = np.asarray(samples)
    if samples.shape != (PAIR_COUNT, 2 + WITNESS_COUNT) or not np.issubdtype(
        samples.dtype, np.integer
    ):
        raise ValueError(
            f"samples must be an integer array of shape [{PAIR_COUNT},{2 + WITNESS_COUNT}]"
        )
    if np.any(samples < 0) or np.any(samples >= len(points)):
        raise ValueError("sample indices are out of range")
    if np.any(samples[:, 0] == samples[:, 1]) or np.any(
        samples[:, 2:, None] == samples[:, None, :2]
    ):
        raise ValueError("endpoints must differ and witnesses must exclude endpoints")
    normals, offsets = pair_planes(points, samples)
    keep = np.abs(offsets) <= OFFSET_GATE_OVER_DIAGONAL
    normals, offsets, samples = normals[keep], offsets[keep], samples[keep]
    weights = witness_weights(points, samples, normals, offsets, occupancy_grid(points))
    histogram = np.zeros((LATITUDE_COUNT, LONGITUDE_COUNT))
    deposit_votes(histogram, normals, offsets, weights)
    return histogram / max(histogram.sum(), MASS_NORMALIZATION_EPSILON)
