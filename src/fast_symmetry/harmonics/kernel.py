"""Frozen projective Gaussian harmonic multipliers."""

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def closed_form_multipliers(
    bandlimit: int = 48,
    quadrature: int = 192,
    sigma: float = 0.06,
    dtype=np.float32,
):
    """Return lambda_l = 2pi int_-1^1 k(t) P_l(t) dt.

    Spatial samples use the configured field dtype. Gauss--Legendre
    nodes/weights and Legendre values use
    NumPy's float64 implementation before the final cast.
    """
    dtype = np.dtype(dtype)
    mu, weights = leggauss(quadrature)
    # Production constructs spatial samples with JAX float32 before NumPy's
    # float64 quadrature contraction.  Preserve that arithmetic boundary while
    # replacing the general geodesic by the explicit projective formula.
    import jax.numpy as jnp

    t = jnp.asarray(mu, dtype=str(dtype))
    distance = jnp.asarray(0.5, dtype=t.dtype) * jnp.arccos(jnp.clip(jnp.abs(t), 0, 1))
    spatial = np.asarray(
        jnp.exp(-jnp.square(distance / jnp.asarray(sigma, dtype=t.dtype)))
    )
    legendre = legvander(mu, bandlimit)
    multipliers = (
        2 * np.pi * np.einsum("q,q,ql->l", spatial, weights, legendre)
    ).astype(dtype)
    # Projective evenness is exact; remove quadrature roundoff exactly as the
    # production builder does for a boundary endpoint.
    multipliers[1::2] = 0
    return multipliers
