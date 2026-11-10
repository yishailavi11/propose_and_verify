"""Reflected-witness occupancy residuals and continuous pair-vote weights."""

from functools import partial

import jax
import jax.numpy as jnp


def _witness_mass(residuals, readout):
    compatibility = jnp.exp(
        -jnp.square(residuals) / (2 * readout.witness_tau_over_diagonal**2)
    )
    raw_weight = jnp.mean(compatibility, axis=1) ** readout.witness_power
    mass = raw_weight / jnp.maximum(jnp.sum(raw_weight), 1e-30)
    return compatibility, raw_weight, mass


@partial(jax.jit, static_argnames=("resolution",))
def _voxel_witness_residuals(points, witness, normals, offsets, *, resolution: int):
    """Approximate reflected-witness distance using a 3D occupancy grid."""
    points = jnp.asarray(points)
    dimension = 3
    lower = jnp.min(points, axis=0)
    upper = jnp.max(points, axis=0)
    step = jnp.maximum((upper - lower) / resolution, 1e-7)
    cells = jnp.clip(
        jnp.floor((points - lower) / step).astype(jnp.int32), 0, resolution - 1
    )
    flat = (cells[:, 0] * resolution + cells[:, 1]) * resolution + cells[:, 2]
    delta = jnp.asarray(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=jnp.int32,
    )
    occupancy = jnp.zeros((resolution**dimension,), dtype=jnp.bool_).at[flat].set(True)
    source = points[witness]
    signed = jnp.sum(source * normals[:, None, :], axis=-1) - offsets[:, None]
    query = source - 2 * signed[..., None] * normals[:, None, :]
    base = jnp.floor((query - lower) / step).astype(jnp.int32)
    neighbor = base[..., None, :] + delta
    valid = jnp.all((neighbor >= 0) & (neighbor < resolution), axis=-1)
    clipped = jnp.clip(neighbor, 0, resolution - 1)
    ids = (clipped[..., 0] * resolution + clipped[..., 1]) * resolution + clipped[
        ..., 2
    ]
    centers = lower + (clipped.astype(points.dtype) + 0.5) * step
    distance = jnp.linalg.norm(query[..., None, :] - centers, axis=-1)
    distance = jnp.where(valid & occupancy[ids], distance, jnp.inf)
    return jnp.min(distance, axis=-1)
