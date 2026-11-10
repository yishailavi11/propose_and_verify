"""Validate and normalize an observation while starting its exact NN index."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.normalization import normalize_input_device
from ..verification.cpu import BoundedCPUVerifier
from ..verification.cuda.quantiles import AngularExactQuantileSpatialNN
from .types import PreparedShape


def validate_points(points):
    """Enforce the canonical surface-observation contract without coercing dtype."""
    points = np.asarray(points)
    if points.shape != (100000, 3) or points.dtype != np.float32:
        raise ValueError(
            "prepared points must have shape (100000, 3) and dtype float32"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("prepared points must be finite")
    return points


def prepare_shape(runtime, points, threshold, timing, started):
    """Preserve production normalization and CPU/GPU verifier-build overlap.

    The CUDA BVH borrows the world-coordinate device points, so it can start
    before normalization. PreparedShape keeps that buffer alive until queries
    finish. The CPU KD-tree starts immediately after the diagonal is available.
    """
    device = jax.block_until_ready(jnp.asarray(points))
    future = None
    if not runtime.cpu:
        future = runtime.executor.submit(
            AngularExactQuantileSpatialNN.from_device,
            int(device.unsafe_buffer_pointer()),
            len(points),
            3,
            1.0,
            0,
            runtime.library,
        )
    normalized, center, diagonal = jax.block_until_ready(normalize_input_device(device))
    center, diagonal = jax.device_get((center, diagonal))
    diagonal = float(diagonal)
    if not np.isfinite(diagonal) or diagonal <= 0:
        if future is not None:
            future.result().close()
        raise ValueError("prepared points have zero or nonfinite AABB diagonal")
    timing["preparation_ms"] = (time.perf_counter_ns() - started) / 1e6
    if runtime.cpu:
        future = runtime.executor.submit(
            BoundedCPUVerifier, points, diagonal, threshold
        )
    return PreparedShape(points, device, normalized, center, diagonal, future)
