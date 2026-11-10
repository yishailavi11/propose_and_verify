"""DUCC CPU synthesis at fixed projective directions."""

import jax.numpy as jnp
import numpy as np

from .coordinates import spherical_locations


class CPUHarmonicSynthesis:
    """DUCC synthesis of the unchanged even-degree production coefficients."""

    def __init__(self, directions, *, bandlimit=48, threads=8):
        """Prepare DUCC locations and triangular (ell,m) packing for fixed directions."""
        import ducc0

        self.ducc = ducc0
        self.bandlimit = bandlimit
        self.threads = threads
        self.locations = spherical_locations(directions)
        self.degrees = np.concatenate(
            [np.arange(m, bandlimit + 1) for m in range(bandlimit + 1)]
        )
        self.orders = np.concatenate(
            [np.full(bandlimit + 1 - m, m) for m in range(bandlimit + 1)]
        )

    def __call__(self, coefficients, native_field=None):
        """Pack positive-order coefficients and synthesize float32 field values [1,N]."""
        del native_field
        alm = np.ascontiguousarray(
            np.asarray(coefficients)[:, self.degrees, self.orders]
        )
        values = self.ducc.sht.synthesis_general(
            alm=alm,
            spin=0,
            lmax=self.bandlimit,
            loc=self.locations,
            epsilon=2e-6,
            nthreads=self.threads,
        )
        return jnp.asarray(values, dtype=jnp.float32)
