"""Angular analysis basis, preserving the production float32 SciPy convention."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm_y


def angular_basis():
    mu = leggauss(50)[0].astype(np.float32)
    theta = np.arccos(np.clip(mu, -1, 1))
    values = np.zeros((49, 25, 50), np.float32)
    for m in range(49):
        for ell in range(0, 49, 2):
            if ell >= m:
                values[m, ell // 2] = np.real(sph_harm_y(ell, m, theta, 0.0)).astype(
                    np.float32
                )
    return mu, values
