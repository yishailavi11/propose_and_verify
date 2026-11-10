"""One intrinsic R2 step: fit local curvature, try bounded updates, keep improvement.

Helpers below are ordinary Python functions traced inside refine_once's single
JIT. Splitting their source does not introduce extra JITs or rounding boundaries.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .derivatives import finite_derivatives
from .evaluate import even_evaluate
from .recurrence import constrain_to_initial, exp_map, tangent_basis


def newton_tangent(g, h, initial_value, radius):
    """Solve the 2x2 Newton system only at stable maxima; cap the tangent step."""
    trace = h[:, 0, 0] + h[:, 1, 1]
    discriminant = jnp.sqrt((h[:, 0, 0] - h[:, 1, 1]) ** 2 + 4 * h[:, 0, 1] ** 2)
    low, high = (trace - discriminant) / 2, (trace + discriminant) / 2
    condition = jnp.abs(low) / jnp.maximum(jnp.abs(high), 1e-12)
    determinant = h[:, 0, 0] * h[:, 1, 1] - h[:, 0, 1] ** 2
    safe = jnp.where(jnp.abs(determinant) > 1e-16, determinant, 1.0)
    delta = (
        jnp.stack(
            [
                -h[:, 1, 1] * g[:, 0] + h[:, 0, 1] * g[:, 1],
                h[:, 0, 1] * g[:, 0] - h[:, 0, 0] * g[:, 1],
            ],
            axis=1,
        )
        / safe[:, None]
    )
    stable = (
        (high < -jnp.maximum(jnp.abs(initial_value) * 1e-4, 1e-10))
        & (condition < 1e6)
        & jnp.all(jnp.isfinite(delta), axis=1)
        & jnp.all(jnp.isfinite(g), axis=1)
    )
    delta = jnp.where(stable[:, None], delta, 0.0)
    length = jnp.linalg.norm(delta, axis=1)
    delta *= jnp.minimum(1.0, radius / jnp.maximum(length, 1e-12))[:, None]
    return delta, stable, condition


def trial_directions(initial, delta, radius):
    """Map full, half and quarter tangent steps back to the unit sphere."""
    axes = tangent_basis(initial)
    fractions = jnp.array([1.0, 0.5, 0.25], initial.dtype)
    options = exp_map(
        jnp.broadcast_to(initial[:, None], (len(initial), 3, 3)),
        jnp.broadcast_to(axes[:, None], (len(initial), 3, 2, 3)),
        delta[:, None] * fractions[None, :, None],
    )
    options = constrain_to_initial(
        jnp.repeat(initial, 3, axis=0), options.reshape(-1, 3), radius
    ).reshape(len(initial), 3, 3)
    return options


def best_improving_update(
    coefficients, shells, initial, valid, g, initial_value, options
):
    """Choose the best trial only when its field value improves the initial one."""
    option_values = even_evaluate(
        coefficients, jnp.repeat(shells, 3), options.reshape(-1, 3)
    ).reshape(len(initial), 3)
    best = jnp.argmax(option_values, axis=1)
    next_value = jnp.take_along_axis(option_values, best[:, None], axis=1)[:, 0]
    next_normal = jnp.take_along_axis(options, best[:, None, None], axis=1)[:, 0]
    take = (
        valid
        & jnp.isfinite(next_value)
        & (next_value > initial_value)
        & (
            jnp.linalg.norm(g, axis=1)
            > jnp.maximum(jnp.abs(initial_value) * 1e-5, 1e-10)
        )
    )
    return (
        jnp.where(take[:, None], next_normal, initial),
        jnp.where(take, next_value, initial_value),
        take,
    )


@jax.jit
def refine_once(coefficients, shells, initial, valid):
    """Apply one production R2 step to a padded batch of candidate directions."""
    step = jnp.asarray(np.deg2rad(1.0), initial.dtype)
    radius = jnp.asarray(np.deg2rad(3.0), initial.dtype)
    g, h, initial_value = finite_derivatives(coefficients, shells, initial, step)
    delta, stable, condition = newton_tangent(g, h, initial_value, radius)
    options = trial_directions(initial, delta, radius)
    normals, final_value, take = best_improving_update(
        coefficients, shells, initial, valid, g, initial_value, options
    )
    return normals, {
        "initial_value": initial_value,
        "final_value": final_value,
        "iterations_with_update": take.astype(np.int32),
        "fallback": ~stable & valid,
        "hessian_condition": condition,
    }


def refine_padded(
    coefficients,
    shells,
    normals,
    *,
    method="R2",
    iterations=1,
    step_degrees=1.0,
    capacity=64,
):
    """Pad variable candidate counts to the unchanged CPU/GPU JIT capacities."""
    if method != "R2" or iterations != 1 or step_degrees != 1.0:
        raise ValueError("specialization supports frozen one-step R2 only")
    count = len(normals)
    if count > capacity:
        raise ValueError("candidate count exceeds fixed capacity")
    padded = np.zeros((capacity, 3), np.float32)
    padded[:, 2] = 1
    padded[:count] = normals
    indices = np.zeros(capacity, np.int32)
    indices[:count] = shells
    valid = np.arange(capacity) < count
    values, diagnostics = jax.device_get(
        refine_once(
            coefficients, jnp.asarray(indices), jnp.asarray(padded), jnp.asarray(valid)
        )
    )
    return values[:count], {k: v[:count] for k, v in diagnostics.items()}
