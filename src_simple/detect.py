"""Read this file first: FAST in the order of the paper's algorithm.

Planes use the input coordinate frame and satisfy normal @ x = offset.
NumPy/SciPy/DUCC handle geometry and fields; JAX preserves CPU random streams.
There are no production imports or custom C++/CUDA kernels.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .constants import (
    ACCEPTANCE_THRESHOLD_OVER_DIAGONAL,
    CPU_WORKERS,
    DEFAULT_PAIR_SEED,
    KDTREE_LEAF_SIZE,
    VERIFICATION_QUANTILE,
    VERIFICATION_SOURCE_COUNT,
)
from .harmonics import project_and_smooth, synthesize
from .normalization import normalize
from .readout import projective_icosphere, select_peaks
from .refinement import refine
from .sampling import sample_indices
from .voting import angular_histogram


@dataclass
class Plane:
    """A unit normal, world offset, and dimensionless exact p99 residual."""

    normal: np.ndarray
    offset: float
    p99: float


def verification_scores(points, center, diagonal, normals):
    """Reflect 512 sources through centroid planes; query the full target cloud.

    This uses unbounded exact queries for clarity. Production can stop early
    when bounds already prove rejection, with the same acceptance decision.
    """
    if len(normals) == 0:
        return np.empty(0)
    points = np.asarray(points, dtype=np.float64)
    source = points[
        np.arange(VERIFICATION_SOURCE_COUNT) * len(points) // VERIFICATION_SOURCE_COUNT
    ]
    signed = (source - center) @ normals.T
    reflected = source[None] - 2 * signed.T[:, :, None] * normals[:, None]
    tree = cKDTree(
        points, leafsize=KDTREE_LEAF_SIZE, balanced_tree=False, compact_nodes=False
    )
    distance = tree.query(reflected, workers=CPU_WORKERS)[0] / diagonal
    return np.quantile(distance, VERIFICATION_QUANTILE, axis=1)


def candidate_planes(points, seed=DEFAULT_PAIR_SEED, *, samples=None):
    """Propose, refine and exactly verify every post-NMS candidate.

    Keeping the unthresholded scores lets detection and single-plane retrieval
    share the same inference pass. Optional samples are [250000,4] integers.
    """
    points = np.asarray(points, dtype=np.float32)
    normalized, center, diagonal = normalize(points)
    if samples is None:
        samples = sample_indices(len(points), seed)
    histogram = angular_histogram(normalized, samples)
    coefficients = project_and_smooth(histogram)
    directions, graph, full_ids = projective_icosphere()
    values = synthesize(coefficients, directions)
    original = select_peaks(values, directions, graph, full_ids)
    refined = refine(coefficients, original)
    scores = verification_scores(
        points, center, diagonal, np.concatenate([original, refined])
    ).reshape(2, -1)
    # Exact verification chooses between both hypotheses; ties keep refinement.
    choose_original = scores[0] < scores[1]
    normals = np.where(choose_original[:, None], original, refined)
    best = np.min(scores, axis=0)
    return [
        Plane(n, float(n @ center), float(score)) for n, score in zip(normals, best)
    ]


def detect(points, seed=DEFAULT_PAIR_SEED, *, single_plane=False, samples=None):
    """Return accepted planes, or the minimum-p99 plane without a cutoff.

    Defaults retain production's Threefry stream and all frozen method settings.
    """
    candidates = candidate_planes(points, seed, samples=samples)
    if single_plane:
        return [min(candidates, key=lambda plane: plane.p99)] if candidates else []
    return [
        plane for plane in candidates if plane.p99 <= ACCEPTANCE_THRESHOLD_OVER_DIAGONAL
    ]
