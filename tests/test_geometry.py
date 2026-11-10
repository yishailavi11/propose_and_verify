"""Geometric invariants and projective quotient topology."""

import jax
import numpy as np
import pytest

from fast_symmetry import AngularCPUDetector3D, AngularDetector3D
from fast_symmetry.config import ordered_plane_nms_indices
from fast_symmetry.geometry.projective import icosphere_quotient_extractor


def test_projective_graph_and_plateau():
    base, extractor, geometry, directions = icosphere_quotient_extractor(6)
    assert len(base.directions) == 40962 and directions.shape == (20481, 3)
    np.testing.assert_array_equal(
        geometry.full_to_quotient, geometry.full_to_quotient[geometry.aliases]
    )
    values = np.zeros(20481, np.float32)
    q = 10
    neighbor = next(iter(geometry.adjacent[q]))
    values[[q, neighbor]] = 1
    result = extractor._propose_values(values, 256)
    assert len(result["indices"]) == 1
    assert extractor.last_diagnostic["added_components"] == 1


def test_centered_nms_and_sign_equivalence():
    a = np.deg2rad(2)
    normals = np.array([[1, 0, 0], [np.cos(a), np.sin(a), 0], [-1, 0, 0]])
    assert ordered_plane_nms_indices(
        normals, np.zeros(3), 1, angle_degrees=6, offset_over_diagonal=0.01
    ).tolist() == [0]


def test_translation_and_centroid_offsets(tmp_path):
    cls = AngularCPUDetector3D if jax.default_backend() == "cpu" else AngularDetector3D
    half = np.random.default_rng(20261021).uniform(-1, 1, (50000, 3)) * [1.2, 0.7, 0.4]
    half[:, 2] = np.abs(half[:, 2])
    points = np.r_[half, half * [1, 1, -1]].astype(np.float32)
    diagonal = np.linalg.norm(np.ptp(points, axis=0))
    outputs = []
    with cls(cache_dir=tmp_path) as detector:
        for delta in [np.zeros(3), diagonal * np.array([0.5, -0.25, 0.75])]:
            result = detector.detect((points + delta).astype(np.float32), seed=1771)
            record = result.to_dict()
            assert detector.state["coeff"].shape == (1, 49, 49)
            for candidate in record["candidates"]:
                np.testing.assert_allclose(
                    candidate["offset_refined"],
                    np.dot(candidate["normal_refined"], record["center"]),
                    rtol=0,
                    atol=2e-7 * diagonal,
                )
            assert result.predictions
            outputs.append(
                (max(result.predictions, key=lambda p: abs(p.normal[2])), delta)
            )
        single = detector.detect_single(points, seed=1771)
        assert single is not None and np.isfinite(single["p99"])
        with pytest.raises(ValueError):
            detector.detect(points[:10])
    a, _ = outputs[0]
    b, delta = outputs[1]
    na = np.array(a.normal)
    nb = np.array(b.normal)
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    assert np.degrees(np.arccos(np.clip(abs(na @ nb), 0, 1))) < 0.02
    sign = 1 if na @ nb >= 0 else -1
    assert abs(a.offset - sign * (b.offset - np.dot(b.normal, delta))) / diagonal < 2e-6
