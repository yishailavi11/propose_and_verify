"""Evaluate the continuous even-degree field by normalized Cartesian recurrence.

coefficients[channel, ell, m>=0] use the orthonormal complex SciPy convention.
The channel index is always zero in the angular detector. Positive orders m>0
contribute twice their real part; odd degrees are identically zero.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .recurrence import _initial, _next, recurrence_constants


@jax.jit
def even_evaluate(coefficients, shells, directions):
    """Same recurrence state, accumulating only nonzero even degrees."""
    L = coefficients.shape[1] - 1
    a, b = (jnp.asarray(v, directions.dtype) for v in recurrence_constants(L))
    y0 = _initial(directions, L)
    weights = jnp.asarray(np.r_[1.0, np.full(L, 2.0)], directions.dtype)
    value = jnp.real(coefficients[shells, 0, 0] * y0[:, 0])

    def degree_pair(index, state):
        previous_even, previous_odd, value = state
        odd_degree = 2 * index + 1
        odd = _next(odd_degree, previous_even, previous_odd, directions, a, b)
        even_degree = odd_degree + 1
        even = _next(even_degree, odd, previous_even, directions, a, b)
        value += jnp.sum(
            jnp.real(coefficients[shells, even_degree, :] * even) * weights,
            axis=1,
        )
        return even, odd, value

    return jax.lax.fori_loop(
        0, L // 2, degree_pair, (y0, jnp.zeros_like(y0), value), unroll=2
    )[2]
