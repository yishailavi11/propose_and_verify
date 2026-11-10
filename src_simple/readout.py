"""Paper §9.1: use a NetworkX graph for projective peaks and plateaus."""

from functools import lru_cache

import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .constants import (
    CANDIDATE_PEAK_COUNT,
    ICOSPHERE_MAX_DEGREE,
    ICOSPHERE_SUBDIVISIONS,
    MAX_PLATEAU_VERTICES,
    NMS_ANGLE_DEGREES,
    PLATEAU_CENTER_MIN_NORM,
    REFERENCE_PEAK_COUNT,
    RELATIVE_PEAK_FLOOR,
)


@lru_cache(maxsize=1)
def projective_icosphere():
    """Return 20,481 directions, their graph, and original full-sphere IDs.

    Trimesh constructs the sphere. The KD-tree finds antipodes; NetworkX merges
    duplicate edges when they are mapped onto the projective vertex IDs.
    """
    sphere = trimesh.creation.icosphere(subdivisions=ICOSPHERE_SUBDIVISIONS)
    vertices = sphere.vertices
    antipodes = cKDTree(vertices).query(-vertices)[1]
    representatives = np.flatnonzero(np.arange(len(vertices)) < antipodes)
    quotient_id = np.empty(len(vertices), dtype=int)
    quotient_id[representatives] = np.arange(len(representatives))
    quotient_id[antipodes[representatives]] = np.arange(len(representatives))
    graph = nx.from_edgelist(quotient_id[sphere.edges_unique])
    # Degree-five vertices repeat a neighbor to make a rectangular gather table.
    graph.graph["neighbors"] = np.array(
        [
            np.resize(list(graph[i]), ICOSPHERE_MAX_DEGREE)
            for i in range(len(representatives))
        ],
        dtype=np.int32,
    )
    return np.asarray(vertices[representatives]), graph, representatives


def plateau_representative(members, directions, full_ids):
    """Align signs, average directions, choose nearest member; ties use full ID."""
    aligned = directions[members].copy()
    aligned *= np.where(aligned @ aligned[0] >= 0, 1, -1)[:, None]
    center = aligned.sum(axis=0)
    if np.linalg.norm(center) < PLATEAU_CENTER_MIN_NORM:
        return None
    alignment = aligned @ (center / np.linalg.norm(center))
    return members[np.lexsort((full_ids[members], -alignment))[0]]


def plateau_peaks(values, directions, graph, full_ids, neighbor_max):
    """An eligible plateau is an equal-valued component above its boundary."""
    weak = np.isfinite(values) & (values == neighbor_max)
    weak &= np.abs(values) >= np.finfo(values.dtype).tiny
    if not weak.any():
        return []
    equal_edges = [
        (a, b) for a, b in graph.edges if weak[a] and weak[b] and values[a] == values[b]
    ]
    peaks = []
    for component in nx.connected_components(nx.from_edgelist(equal_edges)):
        if not 2 <= len(component) <= MAX_PLATEAU_VERTICES:
            continue
        boundary = list(nx.node_boundary(graph, component))
        members = np.array(sorted(component))
        if not boundary or values[members[0]] <= values[boundary].max():
            continue
        representative = plateau_representative(members, directions, full_ids)
        if representative is not None:
            peaks.append(representative)
    return peaks


def select_peaks(values, directions, graph, full_ids):
    """Strict/plateau maxima → top256 → 1% floor → top64 → greedy 6° NMS."""
    neighbor_max = values[graph.graph["neighbors"]].max(axis=1)
    nonzero = np.isfinite(values) & (np.abs(values) >= np.finfo(values.dtype).tiny)
    strict = np.flatnonzero(nonzero & (values > neighbor_max))
    plateau = plateau_peaks(values, directions, graph, full_ids, neighbor_max)
    peaks = np.asarray([*strict, *plateau], dtype=int)
    peaks = peaks[np.lexsort((full_ids[peaks], -values[peaks]))][:REFERENCE_PEAK_COUNT]
    if not len(peaks):
        return np.empty((0, 3))
    peaks = peaks[values[peaks] >= RELATIVE_PEAK_FLOOR * values[peaks[0]]][
        :CANDIDATE_PEAK_COUNT
    ]
    selected = []
    for index in peaks:
        normal = directions[index]
        if not selected or np.max(np.abs(np.asarray(selected) @ normal)) < np.cos(
            np.deg2rad(NMS_ANGLE_DEGREES)
        ):
            selected.append(normal)
    return np.asarray(selected).reshape(-1, 3)
