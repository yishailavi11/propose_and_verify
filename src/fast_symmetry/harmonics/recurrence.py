"""Normalized spherical harmonic recurrence and intrinsic tangent geometry."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(None)
def recurrence_constants(bandlimit):
    """Precompute normalized Legendre recurrence coefficients a[ell,m], b[ell,m]."""
    ell = np.arange(bandlimit + 1)[:, None]
    m = np.arange(bandlimit + 1)[None, :]
    denominator = np.maximum(ell * ell - m * m, 1)
    a = np.sqrt(np.maximum(4 * ell * ell - 1, 0) / denominator)
    b = np.sqrt(
        np.maximum((2 * ell + 1) * ((ell - 1) ** 2 - m * m), 0)
        / np.maximum((2 * ell - 3) * denominator, 1)
    )
    mask = m < ell
    a = np.where(mask, a, 0)
    b = np.where(mask, b, 0)
    return a, b


def _initial(directions, L):
    """Initialize Y_00 for K directions; higher degree/order slots start at zero."""
    dtype = jnp.result_type(directions.dtype, jnp.complex64)
    y = (
        jnp.zeros((len(directions), L + 1), dtype)
        .at[:, 0]
        .set(jnp.asarray(1 / np.sqrt(4 * np.pi), dtype))
    )
    return y


def _next(ell, previous, previous_two, directions, a, b):
    """Advance one degree using the prior two rows and the diagonal Y_ell,ell."""
    degree = jnp.asarray(ell, directions.dtype)
    diagonal = (
        -jnp.sqrt((2.0 * degree + 1) / (2.0 * degree))
        * (directions[:, 0] + 1j * directions[:, 1])
        * previous[:, ell - 1]
    )
    # Regular-solid-harmonic recurrence handles the tiny norm error of float32
    # unit vectors. Compute rho^2-1 without subtracting two numbers near one.
    residual = (
        directions[:, 0] ** 2
        + directions[:, 1] ** 2
        + (directions[:, 2] - 1) * (directions[:, 2] + 1)
    )[:, None]
    result = (
        a[ell] * directions[:, 2, None] * previous
        - b[ell] * previous_two
        - residual * b[ell] * previous_two
    )
    return jnp.where(
        jnp.arange(result.shape[1])[None, :] == ell, diagonal[:, None], result
    )


@jax.jit
def tangent_basis(normals):
    """Choose two orthonormal tangent axes [K,2,3], avoiding polar singularities."""
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(normals), axis=1), 3)
    e1 = jnp.cross(normals, axis)
    e1 /= jnp.linalg.norm(e1, axis=1, keepdims=True)
    return jnp.stack([e1, jnp.cross(normals, e1)], axis=1)


@jax.jit
def exp_map(normals, basis, tangent):
    """Map 2D tangent coordinates onto the sphere, with stable small-angle series."""
    q = jnp.sum(tangent * tangent, axis=-1)
    angle = jnp.sqrt(jnp.maximum(q, 1e-12))
    cosine = jnp.where(q < 1e-6, 1 - q / 2 + q * q / 24, jnp.cos(angle))
    sinc = jnp.where(q < 1e-6, 1 - q / 6 + q * q / 120, jnp.sin(angle) / angle)
    moved = cosine[..., None] * normals + sinc[..., None] * jnp.einsum(
        "...i,...ij->...j", tangent, basis
    )
    return moved / jnp.linalg.norm(moved, axis=-1, keepdims=True)


STENCIL = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]],
    np.float32,
)


@jax.jit
def constrain_to_initial(initial, proposed, radius):
    """Keep a proposed direction inside the initial normal's angular trust region."""
    dot = jnp.clip(jnp.sum(initial * proposed, axis=1), -1.0, 1.0)
    angle = jnp.arccos(dot)
    tangent = proposed - dot[:, None] * initial
    tangent /= jnp.maximum(jnp.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bound = jnp.cos(radius) * initial + jnp.sin(radius) * tangent
    return jnp.where((angle > radius)[:, None], bound, proposed)
