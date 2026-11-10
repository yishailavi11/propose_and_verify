"""Fuse GPU pair generation, witnesses, offset gating, angular CIC and analysis."""

import jax
import jax.numpy as jnp

from ..method import OFFSET_GATE_OVER_DIAGONAL
from .histogram import angular_histogram
from .pairs import _make_pairs, _sample_witnesses
from .witnesses import _voxel_witness_residuals, _witness_mass


def build_gpu_frontend(mu, coefficients, readout):
    """Create the production whole-frontend JIT, retaining its fusion boundary.

    Points [100000,3] are already normalized. Pair and witness indices are sampled
    from independent Threefry streams derived from the caller's seed. The return
    value is the smoothed angular coefficient array, without host transfers.
    """
    mu_nodes = jnp.asarray(mu)

    @jax.jit
    def gpu_frontend(normalized, seed):
        """Return smoothed coefficients without leaving the production whole-frontend JIT."""
        pair = _make_pairs(normalized, seed)
        witness = _sample_witnesses(len(normalized), pair[0], pair[1], seed)
        residual = _voxel_witness_residuals(
            normalized, witness, pair[2], pair[3], resolution=96
        )
        raw = _witness_mass(residual, readout)[1]
        raw = jnp.where(
            jnp.abs(pair[3]) <= jnp.float32(OFFSET_GATE_OVER_DIAGONAL), raw, 0
        )
        mass = raw / jnp.maximum(jnp.sum(raw), jnp.float32(1e-30))
        return coefficients(angular_histogram(pair[4], mass, mu_nodes, 193))

    return gpu_frontend
