"""Production centroid and bounding-box diagonal, computed in float32."""

import jax
import jax.numpy as jnp


@jax.jit
def _normalize_input_device(points):
    center = jnp.mean(points, axis=0)
    scale = jnp.linalg.norm(jnp.max(points, axis=0) - jnp.min(points, axis=0))
    return (points - center) / scale, center, scale


def normalize_input_device(vertices):
    """Normalize device geometry and compute its transform in one JAX pass.

    This is the low-overhead counterpart to :func:`normalize_input` for a
    GPU-resident readout.  The returned ``center`` and ``scale`` intentionally
    remain JAX arrays so callers can reuse the AABB diagonal without another
    host scan or synchronization.
    """
    points = jnp.asarray(vertices)
    if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0]:
        raise ValueError("vertices must be a nonempty [N,3] array")
    return _normalize_input_device(points)
