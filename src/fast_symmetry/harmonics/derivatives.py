"""The frozen nine-point tangent stencil for the field gradient and Hessian."""

import jax
import jax.numpy as jnp

from .evaluate import even_evaluate
from .recurrence import STENCIL, exp_map, tangent_basis


@jax.jit
def finite_derivatives(coefficients, shells, normals, step):
    """Return gradient [K,2], symmetric Hessian [K,2,2], and center values."""
    axes = tangent_basis(normals)
    count = len(normals)
    tangents = jnp.broadcast_to(jnp.asarray(STENCIL) * step, (count, 9, 2))
    directions = exp_map(
        jnp.broadcast_to(normals[:, None], (count, 9, 3)),
        jnp.broadcast_to(axes[:, None], (count, 9, 2, 3)),
        tangents,
    )
    values = even_evaluate(
        coefficients,
        jnp.repeat(shells, 9),
        directions.reshape(-1, 3),
    ).reshape(count, 9)
    g = jnp.stack(
        [
            (values[:, 1] - values[:, 2]) / (2 * step),
            (values[:, 3] - values[:, 4]) / (2 * step),
        ],
        axis=1,
    )
    h00 = (values[:, 1] + values[:, 2] - 2 * values[:, 0]) / (step * step)
    h11 = (values[:, 3] + values[:, 4] - 2 * values[:, 0]) / (step * step)
    h01 = (values[:, 5] - values[:, 6] - values[:, 7] + values[:, 8]) / (
        4 * step * step
    )
    h = jnp.stack(
        [jnp.stack([h00, h01], axis=1), jnp.stack([h01, h11], axis=1)],
        axis=1,
    )
    return g, h, values[:, 0]
