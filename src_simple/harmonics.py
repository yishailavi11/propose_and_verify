"""Paper §8.4: mass projection, Funk–Hecke smoothing and DUCC synthesis."""

from functools import lru_cache

import ducc0
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.special import sph_harm_y

from .constants import (
    BANDLIMIT,
    CPU_WORKERS,
    FIELD_SYNTHESIS_EPSILON,
    KERNEL_QUADRATURE_COUNT,
    KERNEL_SIGMA_RADIANS,
    LATITUDE_COUNT,
    REFINEMENT_SYNTHESIS_EPSILON,
)


@lru_cache(maxsize=1)
def kernel_multipliers():
    """Eq. (34): lambda_l = 2pi integral k(t) P_l(t) dt, with 192 GL nodes.

    Write the bandwidth directly as the paper's 0.12 radians. This equals
    production's 0.06 with its half-angle metric: (0.5*theta/0.06)^2.
    """
    nodes, weights = leggauss(KERNEL_QUADRATURE_COUNT)
    angle = np.arccos(np.abs(nodes))
    kernel = np.exp(-((angle / KERNEL_SIGMA_RADIANS) ** 2))
    multipliers = 2 * np.pi * ((weights * kernel) @ legvander(nodes, BANDLIMIT))
    multipliers[1::2] = 0
    return multipliers


@lru_cache(maxsize=1)
def latitude_basis():
    """Y_lm(theta,0), with the orthonormal complex SciPy convention."""
    theta = np.arccos(leggauss(LATITUDE_COUNT)[0])
    basis = np.zeros((BANDLIMIT + 1, BANDLIMIT + 1, LATITUDE_COUNT))
    for ell in range(0, BANDLIMIT + 1, 2):
        orders = np.arange(ell + 1)
        basis[ell, : ell + 1] = sph_harm_y(ell, orders[:, None], theta, 0).real
    return basis


def project_and_smooth(histogram):
    """Eq. (33–34): project masses, then multiply each degree by lambda_l.

    H contains masses, not density samples: no latitude quadrature weights or
    FFT normalization belong here. Negative orders follow conjugate symmetry.
    Coefficients [ell,m>=0] use the convention F=sum c_lm Y_lm, doubled for m>0.
    """
    longitude_fft = np.fft.rfft(histogram, axis=1)[:, : BANDLIMIT + 1]
    projected = np.einsum("qm,lmq->lm", longitude_fft, latitude_basis())
    return (projected * kernel_multipliers()[:, None]).astype(np.complex64)


def synthesize(coefficients, directions, *, epsilon=FIELD_SYNTHESIS_EPSILON):
    """Evaluate the continuous field at arbitrary unit directions using DUCC.

    DUCC packs coefficients by increasing m, then ell=m,...,L. Production's
    sampling epsilon and complex64 field storage are retained.
    """
    if len(directions) == 0:
        return np.empty(0)
    packed = np.concatenate([coefficients[m:, m] for m in range(BANDLIMIT + 1)])
    locations = np.column_stack(
        [
            np.arccos(np.clip(directions[:, 2], -1, 1)),
            np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi),
        ]
    )
    return ducc0.sht.synthesis_general(
        alm=packed[None],
        spin=0,
        lmax=BANDLIMIT,
        loc=locations,
        epsilon=epsilon,
        nthreads=CPU_WORKERS,
    )[0]


def evaluate(coefficients, directions):
    """Use the same DUCC evaluator in float64 for the refinement stencil.

    Tight accuracy keeps finite differences stable and agrees with the direct
    SciPy harmonic expansion; no hand-written harmonic recurrence is needed.
    """
    return synthesize(
        np.asarray(coefficients, np.complex128),
        directions,
        epsilon=REFINEMENT_SYNTHESIS_EPSILON,
    )
