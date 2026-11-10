"""Preserve production float32 reductions before the discontinuous voxel lookup."""

import jax
import jax.numpy as jnp
import numpy as np

from .constants import VERIFICATION_SOURCE_COUNT


@jax.jit
def _normalize_array(points):
    center = jnp.mean(points, axis=0)
    diagonal = jnp.linalg.norm(jnp.max(points, axis=0) - jnp.min(points, axis=0))
    return (points - center) / diagonal, center, diagonal


def normalize(points):
    """Center [N,3] points and divide by the observed AABB diagonal.

    The centroid reduction and coordinates must match production float32.
    Changing them can move points across voxel boundaries and change peaks.
    """
    points = np.asarray(points, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < VERIFICATION_SOURCE_COUNT
    ):
        raise ValueError(
            f"expected at least {VERIFICATION_SOURCE_COUNT} points with shape [N,3]"
        )
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        normalized, center, diagonal = jax.device_get(_normalize_array(points))
    diagonal = float(diagonal)
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("point cloud must have a finite positive diagonal")
    return normalized, center, diagonal
