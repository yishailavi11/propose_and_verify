"""Independent numerical checks of the selected method's critical stages."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y

from fast_symmetry.geometry.normalization import normalize_input_device
from fast_symmetry.harmonics.basis import angular_basis
from fast_symmetry.harmonics.cpu_synthesis import CPUHarmonicSynthesis
from fast_symmetry.harmonics.gpu_synthesis import GPUHarmonicSynthesis
from fast_symmetry.method import OFFSET_GATE_OVER_DIAGONAL, THRESHOLD
from fast_symmetry.verification.cpu import BoundedCPUVerifier, quantiles
from fast_symmetry.voting.cpu import FusedAngularHistogram
from fast_symmetry.voting.pairs import _make_pairs, _sample_witnesses


def test_synthesis_against_independent_scipy_harmonics():
    directions = np.asarray(trimesh.creation.icosphere(subdivisions=3).vertices)
    coefficients = np.zeros((1, 49, 49), np.complex64)
    coefficients[0, 0, 0] = 1
    coefficients[0, 8, 3] = 0.025 + 0.01j
    coefficients[0, 32, 4] = 1e-3 - 1e-3j
    coefficients[0, 48, 48] = 1e-6j
    theta = np.arccos(np.clip(directions[:, 2], -1, 1))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    expected = np.zeros(len(directions))
    for ell, m in zip(*np.nonzero(coefficients[0])):
        expected += (1 if m == 0 else 2) * np.real(
            coefficients[0, ell, m] * sph_harm_y(ell, m, theta, phi)
        )
    cls = (
        CPUHarmonicSynthesis if jax.default_backend() == "cpu" else GPUHarmonicSynthesis
    )
    synthesis = cls(directions, bandlimit=48)
    try:
        actual = np.asarray(jax.block_until_ready(synthesis(jnp.asarray(coefficients))))
        assert (
            np.linalg.norm(actual.ravel() - expected) / np.linalg.norm(expected) < 2e-6
        )
    finally:
        if hasattr(synthesis, "close"):
            synthesis.close()


def test_cpp_streams_equal_jax_threefry(tmp_path):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU voting RNG comparison")
    points = np.random.default_rng(19).uniform(-1, 1, (100000, 3)).astype(np.float32)
    normalized, _, _ = normalize_input_device(points)
    mu, _ = angular_basis()
    histogram = FusedAngularHistogram(
        mu, 193, cutoff=OFFSET_GATE_OVER_DIAGONAL, build_root=tmp_path
    )
    _, first, second, witnesses = histogram(normalized, 1771, audit=True)
    reference = _make_pairs(normalized, jnp.int32(1771))
    np.testing.assert_array_equal(first, reference[0])
    np.testing.assert_array_equal(second, reference[1])
    np.testing.assert_array_equal(
        witnesses,
        _sample_witnesses(100000, reference[0], reference[1], jnp.int32(1771)),
    )
    assert np.all(first != second)
    assert np.all(witnesses != first[:, None]) and np.all(witnesses != second[:, None])


def test_bounded_verification_matches_unbounded_exact_queries():
    points = np.random.default_rng(31).normal(size=(1000, 3))
    diagonal = np.linalg.norm(np.ptp(points, axis=0))
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = np.array([0.0, 0.1, -0.2])
    sources = np.arange(512)
    reflected = (
        points[sources][None]
        - 2
        * (np.einsum("sj,cj->cs", points[sources], normals) - offsets[:, None])[
            :, :, None
        ]
        * normals[:, None]
    )
    exact = (
        cKDTree(points).query(reflected.reshape(-1, 3))[0].reshape(3, 512) / diagonal
    )
    expected = np.quantile(exact, 0.99, axis=1)
    partial_bound = float(np.quantile(exact[0], 0.995))
    for threshold in [THRESHOLD, partial_bound, np.inf]:
        verifier = BoundedCPUVerifier(points, diagonal, threshold)
        try:
            distances, _, _ = verifier.query(sources, normals, offsets)
            scores, _ = quantiles(verifier, distances)
            np.testing.assert_array_equal(scores <= threshold, expected <= threshold)
            if threshold == partial_bound:
                assert verifier.resolved_sources > 0
            finite = np.isfinite(scores)
            np.testing.assert_allclose(
                scores[finite], expected[finite], rtol=1e-14, atol=1e-14
            )
        finally:
            verifier.close()


def test_cuda_quantiles_against_independent_cpu_tree(tmp_path):
    if jax.default_backend() == "cpu":
        pytest.skip("requires a CUDA JAX backend")
    from fast_symmetry.verification.cuda.build import build_angular_gpu_library
    from fast_symmetry.verification.cuda.quantiles import AngularExactQuantileSpatialNN
    from fast_symmetry.verification.sources import deterministic_stride_indices

    points = np.random.default_rng(812).normal(size=(4096, 3)).astype(np.float32)
    device = jax.block_until_ready(jnp.asarray(points))
    diagonal = float(np.linalg.norm(np.ptp(points.astype(float), axis=0)))
    library = build_angular_gpu_library(build_root=tmp_path)
    verifier = AngularExactQuantileSpatialNN.from_device(
        int(device.unsafe_buffer_pointer()), len(points), 3, diagonal, 0, library
    )
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    offsets = np.array([0.0, 0.2, -0.1])
    sources = deterministic_stride_indices(len(points))
    try:
        scores, _, _ = verifier.query(sources, normals, offsets)
        selected = points[sources].astype(float)
        reflected = (
            selected[None]
            - 2
            * (np.einsum("sj,cj->cs", selected, normals) - offsets[:, None])[:, :, None]
            * normals[:, None]
        )
        exact = (
            cKDTree(points.astype(float))
            .query(reflected.reshape(-1, 3))[0]
            .reshape(3, 512)
            / diagonal
        )
        np.testing.assert_allclose(
            scores.p99, np.quantile(exact, 0.99, axis=1), rtol=2e-6, atol=2e-7
        )
        np.testing.assert_allclose(
            scores.p90, np.quantile(exact, 0.90, axis=1), rtol=2e-6, atol=2e-7
        )
    finally:
        verifier.close()
