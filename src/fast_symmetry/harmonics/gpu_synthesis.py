"""Double-Fourier-sphere and cuFINUFFT GPU synthesis at fixed projective directions."""

import jax
import jax.numpy as jnp
import numpy as np

from .coordinates import spherical_locations


class GPUHarmonicSynthesis:
    """GPU-resident double-Fourier-sphere synthesis and planned NUFFT."""

    def __init__(self, directions, *, bandlimit=48):
        """Precompute even harmonic rings and plan a double-precision arbitrary-point NUFFT."""
        import cufinufft
        import cupy as cp
        from scipy.fft import next_fast_len
        from scipy.special import sph_harm_y

        self.cp = cp
        self.stream = cp.cuda.get_current_stream()
        size = 2 * next_fast_len(bandlimit + 1)
        theta = np.arange(size // 2 + 1) * 2 * np.pi / size
        ring = np.zeros((bandlimit + 1, bandlimit + 1, len(theta)), np.float32)
        for ell in range(0, bandlimit + 1, 2):
            for m in range(ell + 1):
                ring[m, ell] = sph_harm_y(ell, m, theta, 0).real
        ring = jnp.asarray(ring)

        @jax.jit
        def frequency(coefficients):
            """Extend spherical rings to a double Fourier sphere and transform to frequencies."""
            azimuth = jnp.einsum(
                "lm,mlq->qm",
                coefficients[0],
                ring,
                precision=jax.lax.Precision.HIGHEST,
            )
            half = jnp.fft.irfft(azimuth, n=size, axis=-1) * size
            doubled = jnp.concatenate(
                [half, jnp.roll(half[-2:0:-1], size // 2, axis=1)], axis=0
            )
            return jnp.fft.fft2(doubled) / (size * size)

        self.frequency = frequency
        locations = spherical_locations(directions)
        locations = (locations + np.pi) % (2 * np.pi) - np.pi
        self.coordinates = (
            cp.asarray(np.ascontiguousarray(locations[:, 0])),
            cp.asarray(np.ascontiguousarray(locations[:, 1])),
        )
        self.plan = cufinufft.Plan(
            2,
            (size, size),
            eps=1e-10,
            isign=1,
            dtype="complex128",
            modeord=1,
            gpu_stream=self.stream.ptr,
        )
        self.plan.setpts(*self.coordinates)

    def __call__(self, coefficients, native_field=None):
        del native_field
        with self.stream:
            frequency = self.cp.from_dlpack(self.frequency(coefficients)).astype(
                self.cp.complex128
            )
            values = self.cp.ascontiguousarray(
                self.plan.execute(frequency).real, dtype=self.cp.float32
            )
            return jax.dlpack.from_dlpack(values[None], copy=False)

    def close(self):
        self.stream.synchronize()
        self.plan = None
