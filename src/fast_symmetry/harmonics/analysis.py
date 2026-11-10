"""Construct the frozen angular histogram -> smoothed harmonic coefficient map."""

import jax
import jax.numpy as jnp

from ..method import METHOD
from .basis import angular_basis
from .kernel import closed_form_multipliers


def build_coefficient_transform():
    """Return the GL latitude nodes and a reusable JIT-compiled transform.

    The input is normalized angular mass [1,50,193]. Longitude FFT analysis,
    associated-Legendre analysis, and Funk--Hecke multipliers produce complex
    coefficients [1,49,49]. Odd degrees are explicitly zero. Operation order and
    precision are the frozen production expressions.
    """
    bandlimit = METHOD["bandlimit"]
    mu, basis = angular_basis()
    forward_basis = jnp.asarray(basis)
    multipliers = closed_form_multipliers()
    kernel_even = jnp.asarray(multipliers[::2, None, None])

    @jax.jit
    def coefficients(histogram):
        """FFT longitude, analyze latitude, then apply the even zonal multipliers."""
        longitude = jnp.fft.rfft(histogram, axis=-1)[..., : bandlimit + 1]
        forward = jnp.einsum(
            "sqm,mlq->slm",
            longitude,
            forward_basis,
            precision=jax.lax.Precision.HIGHEST,
        )
        output = jnp.einsum(
            "lrs,slm->rlm", kernel_even, forward, precision=jax.lax.Precision.HIGHEST
        )
        return (
            jnp.zeros((1, bandlimit + 1, bandlimit + 1), output.dtype)
            .at[:, ::2, :]
            .set(output)
        )

    return mu, coefficients
