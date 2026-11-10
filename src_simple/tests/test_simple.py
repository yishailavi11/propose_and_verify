"""Independent checks for the readable example; no production imports."""

import networkx as nx
import numpy as np
import trimesh
from scipy.special import sph_harm_y

from src_simple.detect import detect, normalize, verification_scores
from src_simple.harmonics import evaluate, kernel_multipliers, synthesize
from src_simple.readout import projective_icosphere, select_peaks
from src_simple.sampling import sample_indices
from src_simple.voting import deposit_votes, witness_weights


def test_samples_exclude_endpoints():
    samples = sample_indices(512, 31)
    assert np.all(samples[:, 0] != samples[:, 1])
    assert not np.any(samples[:, 2:, None] == samples[:, None, :2])
    assert np.all((samples >= 0) & (samples < 512))
    assert np.any(samples[:, 2] == samples[:, 3])  # Replacement is intentional.
    np.testing.assert_array_equal(samples, sample_indices(512, 31))
    # Frozen production Threefry values, not just distributional equivalence.
    np.testing.assert_array_equal(
        samples[:3], [[203, 378, 354, 15], [84, 197, 328, 388], [506, 164, 310, 305]]
    )


def test_synthesis_matches_direct_scipy_expansion_and_is_even():
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coefficients = np.zeros((49, 49), complex)
    coefficients[0, 0] = 1
    coefficients[8, 3] = 0.2 + 0.1j
    coefficients[48, 48] = 1e-5j
    theta = np.arccos(directions[:, 2])
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    direct = sum(
        (1 if m == 0 else 2)
        * (coefficients[ell, m] * sph_harm_y(ell, m, theta, phi)).real
        for ell, m in zip(*np.nonzero(coefficients))
    )
    np.testing.assert_allclose(evaluate(coefficients, directions), direct, atol=1e-13)
    np.testing.assert_allclose(evaluate(coefficients, -directions), direct, atol=1e-13)
    assert (
        np.linalg.norm(synthesize(coefficients, directions) - direct)
        / np.linalg.norm(direct)
        < 2e-6
    )
    assert np.all(kernel_multipliers()[1::2] == 0)


def test_cic_preserves_mass_and_longitude_periodicity():
    histogram = np.zeros((50, 193))
    normals = np.array([[1.0, -1e-9, 0], [0, 0, 1.0], [0, 0, -1.0]])
    deposit_votes(histogram, normals, np.zeros(3), np.array([2.0, 3.0, 5.0]))
    np.testing.assert_allclose(histogram.sum(), 10)
    np.testing.assert_allclose(histogram[-1].sum(), 3)
    np.testing.assert_allclose(histogram[0].sum(), 5)
    assert histogram[:, 0].sum() > 1.9
    # atan2 followed by mod can round a tiny negative angle to exactly 2pi.
    periodic = np.zeros_like(histogram)
    deposit_votes(periodic, np.array([[1.0, -1e-20, 0.0]]), np.zeros(1), np.ones(1))
    np.testing.assert_allclose(periodic[:, 0].sum(), 1)


def test_witness_search_is_local_not_global_nearest_center():
    step = np.full(3, 0.001)
    occupied = np.zeros((96, 96, 96), bool)
    occupied[0, 0, 0] = True
    centers = (np.arange(96) + 0.5)[None, :] * step[:, None]
    grid = np.zeros(3), step, occupied, centers
    points = np.array(
        [[0.0, 0, 0], [0.001, 0, 0], [0.0005, 0.0005, 0.0005], [0.0025, 0.0005, 0.0005]]
    )
    samples = np.array([[0, 1, 2, 3]])
    weight = witness_weights(
        points, samples, np.array([[0.0, 1, 0]]), np.array([0.0005]), grid
    )
    # Witness 2 is just two cells away: globally close, outside the local window.
    # A global nearest query would give nonzero compatibility, changing the mass.
    np.testing.assert_allclose(weight, [0.25])  # Compatibility [1,0], mean squared.


def test_projective_graph_and_plateau_representative():
    directions, graph, full_ids = projective_icosphere()
    assert len(directions) == 20481
    assert nx.number_of_selfloops(graph) == 0
    assert set(dict(graph.degree).values()) == {5, 6}
    from scipy.spatial import cKDTree

    assert cKDTree(directions).query(-directions)[0].min() > 1e-4
    values = np.zeros(len(directions))
    a, b = list(graph.edges)[100]
    values[[a, b]] = 1
    peaks = select_peaks(values, directions, graph, full_ids)
    assert len(peaks) == 1
    assert np.max(peaks @ directions[[a, b]].T) > 1 - 1e-14
    assert select_peaks(np.zeros(len(values)), directions, graph, full_ids).shape == (
        0,
        3,
    )


def test_box_detection_world_coordinates_and_single_plane():
    mesh = trimesh.creation.box(extents=[1, 1.7, 2.3])
    points, _ = trimesh.sample.sample_surface(mesh, 100000, seed=0)
    points += [7, -3, 4]
    points = points.astype(np.float32)
    predictions = detect(points)
    assert len(predictions) == 3
    normals = np.array([p.normal for p in predictions])
    assert len(set(np.argmax(np.abs(normals), axis=1))) == 3
    assert np.all(np.max(np.abs(normals), axis=1) > np.cos(np.deg2rad(1)))
    _, center, diagonal = normalize(points)
    np.testing.assert_allclose([p.offset for p in predictions], normals @ center)
    scores = verification_scores(points, center, diagonal, normals)
    np.testing.assert_allclose(scores, [p.p99 for p in predictions], atol=1e-14)
    single = detect(points, single_plane=True)
    assert len(single) == 1
    np.testing.assert_allclose(single[0].p99, scores.min(), atol=1e-14)


def test_numpy_gradient_preserves_the_paper_stencil():
    from src_simple.refinement import exp_map, local_quadratic, tangent_basis

    normal = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    basis = tangent_basis(normal)
    coefficients = np.zeros((49, 49), complex)
    coefficients[0, 0] = 1
    coefficients[8, 3] = 0.2 + 0.1j
    h = np.deg2rad(1)
    stencil = (
        np.array(
            [
                [0, 0],
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
            ]
        )
        * h
    )
    center, xp, xm, yp, ym, pp, pm, mp, mm = evaluate(
        coefficients, exp_map(normal, basis, stencil)
    )
    expected_gradient = np.array([xp - xm, yp - ym]) / (2 * h)
    mixed = (pp - pm - mp + mm) / (4 * h**2)
    expected_hessian = np.array(
        [[(xp + xm - 2 * center) / h**2, mixed], [mixed, (yp + ym - 2 * center) / h**2]]
    )
    value, gradient, hessian = local_quadratic(coefficients, normal, basis)
    np.testing.assert_allclose(value, center, atol=1e-14)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(hessian, expected_hessian, atol=1e-10)


def test_normalization_preserves_production_voxel_precision():
    points = np.random.default_rng(19).normal(size=(1000, 3)).astype(np.float32)
    points += np.array([7, -3, 4], np.float32)
    normalized, center, diagonal = normalize(points)
    assert normalized.dtype == np.float32
    assert center.dtype == np.float32
    assert diagonal == float(np.float32(diagonal))


def test_compiled_neighborhood_matches_exhaustive_occupied_centers():
    from src_simple.voting import local_squared_distances, occupancy_grid

    rng = np.random.default_rng(71)
    points = rng.uniform(-0.5, 0.5, (1000, 3)).astype(np.float32)
    grid = occupancy_grid(points)
    lower, step, occupied, _ = grid
    cells = np.argwhere(occupied)
    centers = lower + (cells.astype(points.dtype) + 0.5) * step
    queries = np.concatenate(
        [points[:30], points[:30] + step, points[:30] + 3 * step, [lower - 2 * step]]
    )
    expected = []
    for query in queries:
        cell = np.floor((query - lower) / step).astype(int)
        nearby = centers[np.all(np.abs(cells - cell) <= 1, axis=1)]
        expected.append(np.sum((nearby - query) ** 2, axis=1).min(initial=np.inf))
    # Exact equality also catches compiler changes to float32 operation order.
    np.testing.assert_array_equal(local_squared_distances(queries, grid), expected)
