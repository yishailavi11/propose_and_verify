"""Construct the level-6 icosphere and contract antipodal vertices to RP2."""

from types import SimpleNamespace

import numpy as np
import trimesh

from .peaks import QuotientExtractor


def antipodal_mapping(source_base):
    """Choose the lower full-sphere ID in each antipodal pair, preserving numbering."""
    directions = np.asarray(source_base.directions, np.float32)
    aliases = np.asarray(source_base.aliases, np.int32)
    count = len(directions)
    subdivisions = int(round(np.log((count - 2) / 10) / np.log(4)))
    assert count == 10 * 4**subdivisions + 2
    quotient_count = count // 2
    assert np.array_equal(aliases[aliases], np.arange(count))
    assert np.all(aliases != np.arange(count))
    assert np.max(np.linalg.norm(directions + directions[aliases], axis=1)) < 1e-6
    representatives = np.flatnonzero(np.arange(count) < aliases).astype(np.int32)
    assert len(representatives) == quotient_count
    full_to_quotient = np.empty(count, np.int32)
    full_to_quotient[representatives] = np.arange(len(representatives), dtype=np.int32)
    full_to_quotient[aliases[representatives]] = np.arange(
        len(representatives), dtype=np.int32
    )
    assert np.array_equal(
        np.bincount(full_to_quotient, minlength=quotient_count),
        np.full(quotient_count, 2),
    )

    return (
        directions,
        aliases,
        count,
        subdivisions,
        quotient_count,
        representatives,
        full_to_quotient,
    )


def project_edges(full_edges, full_to_quotient, quotient_count):
    """Contract edges and build the degree-5/6 projective neighbor tables."""
    projected = np.sort(full_to_quotient[full_edges], axis=1)
    self_edges = projected[:, 0] == projected[:, 1]
    assert not np.any(self_edges)
    edges = np.unique(projected, axis=0)
    adjacent = [set() for _ in range(quotient_count)]
    for a, b in edges:
        adjacent[int(a)].add(int(b))
        adjacent[int(b)].add(int(a))
    assert all(i not in row for i, row in enumerate(adjacent))
    assert all(i in adjacent[j] for i, row in enumerate(adjacent) for j in row)
    degrees = np.asarray([len(row) for row in adjacent], np.int32)
    assert set(degrees.tolist()) == {5, 6}
    width = int(degrees.max())
    neighbors = np.empty((quotient_count, width), np.int32)
    for i, row in enumerate(adjacent):
        values = np.asarray(sorted(row), np.int32)
        neighbors[i] = np.resize(values, width)

    return edges, adjacent, neighbors, degrees


def quotient_geometry(source_base):
    """Assemble and validate the antipodal quotient from an icosphere graph."""
    (
        directions,
        aliases,
        count,
        subdivisions,
        quotient_count,
        representatives,
        full_to_quotient,
    ) = antipodal_mapping(source_base)
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    assert np.max(np.abs(np.asarray(sphere.vertices, np.float32) - directions)) == 0
    full_edges = np.unique(
        np.sort(
            np.concatenate(
                [
                    sphere.faces[:, [0, 1]],
                    sphere.faces[:, [1, 2]],
                    sphere.faces[:, [2, 0]],
                ],
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    edges, adjacent, neighbors, degrees = project_edges(
        full_edges, full_to_quotient, quotient_count
    )
    # Each quotient neighborhood must equal the union of the two full-sphere
    # neighborhoods after antipodal contraction.
    ordinary = np.asarray(source_base.tables[0], np.int32)
    for q, rep in enumerate(representatives):
        expected = set(full_to_quotient[ordinary[rep]].tolist())
        expected.update(full_to_quotient[ordinary[aliases[rep]]].tolist())
        assert expected == adjacent[q]
    return SimpleNamespace(
        directions=directions,
        aliases=aliases,
        representatives=representatives,
        full_to_quotient=full_to_quotient,
        full_edges=full_edges,
        edges=edges,
        adjacent=adjacent,
        neighbors=neighbors,
        degrees=degrees,
    )


def quotient_extractor(source_base):
    geometry = quotient_geometry(source_base)
    return QuotientExtractor(source_base, geometry), geometry


def icosphere_quotient_extractor(subdivisions):
    """Build the projective graph without materializing a dense SH basis.

    Production installs its DUCC/NUFFT sampler immediately after construction.
    Returning float64 sampling directions preserves the measured backend input.
    """

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    directions = np.asarray(sphere.vertices, np.float32)
    lookup = {tuple(-point): i for i, point in enumerate(directions)}
    aliases = np.array([lookup[tuple(point)] for point in directions], np.int32)
    edges = np.unique(
        np.sort(
            np.concatenate(
                [
                    sphere.faces[:, [0, 1]],
                    sphere.faces[:, [1, 2]],
                    sphere.faces[:, [2, 0]],
                ]
            ),
            axis=1,
        ),
        axis=0,
    )
    neighbors, _ = neighbors_array(len(directions), edges)
    base = SimpleNamespace(
        directions=directions,
        aliases=aliases,
        tables=(neighbors,),
        basis=None,
    )
    extractor, geometry = quotient_extractor(base)
    sampling_directions = np.asarray(sphere.vertices)[geometry.representatives]
    return base, extractor, geometry, sampling_directions


def neighbors_array(count, edges, ignore_special=False):
    """Construct deterministic padded neighbor rows from sorted, undirected edge sets."""
    adjacent = [set() for _ in range(count)]
    for a, b in edges:
        if a != b:
            adjacent[int(a)].add(int(b))
            adjacent[int(b)].add(int(a))
    ordinary = count - 2 if ignore_special else count
    maximum = max(len(row) for row in adjacent[:ordinary])
    table = np.empty((ordinary, maximum), np.int32)
    for i, row in enumerate(adjacent[:ordinary]):
        values = sorted(row)
        table[i] = np.resize(values, maximum) if values else i
    return table, [np.array(sorted(row), np.int32) for row in adjacent[ordinary:]]
