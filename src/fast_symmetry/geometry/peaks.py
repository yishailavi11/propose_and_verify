"""Find strict field maxima and connected, exactly equal plateaus on RP2."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@jax.jit
def quotient_strict_and_weak(values, neighbors):
    other = jnp.max(values[neighbors], axis=1)
    positive = jnp.isfinite(values) & (jnp.abs(values) >= jnp.finfo(values.dtype).tiny)
    strict = positive & (values > other)
    weak = positive & (values == other)
    return strict, weak


def equal_components(values, weak_vertices, adjacent):
    """Connected components use exact field equality, with no ULP tolerance."""
    lookup = {int(vertex): i for i, vertex in enumerate(weak_vertices)}
    edges = []
    for vertex in weak_vertices:
        for neighbor in adjacent[int(vertex)]:
            slot = lookup.get(int(neighbor))
            if slot is not None and values[neighbor] == values[vertex]:
                edges.append((lookup[int(vertex)], slot))
    array = np.asarray(edges, int).reshape(-1, 2)
    matrix = coo_matrix(
        (np.ones(len(array)), (array[:, 0], array[:, 1])),
        shape=(len(weak_vertices), len(weak_vertices)),
    ).tocsr()
    count, labels = connected_components(matrix, directed=False)
    return [weak_vertices[np.flatnonzero(labels == label)] for label in range(count)]


def plateau_representative(values, members, adjacent, directions, representatives):
    """Reject nonmaximal/degenerate components; break valid plateau ties geometrically."""
    if len(members) == 1:
        return None, False
    if len(members) > 64 or np.ptp(values[members]) != 0:
        return None, True
    member_set = set(members.tolist())
    external = [
        values[n]
        for vertex in members
        for n in adjacent[int(vertex)]
        if n not in member_set
    ]
    if not external or values[members[0]] <= max(external):
        return None, False
    dirs = directions[members].astype(float)
    dirs *= np.where(dirs @ dirs[0] >= 0, 1.0, -1.0)[:, None]
    center = dirs.sum(axis=0)
    norm = np.linalg.norm(center)
    if norm < 1e-8:
        return None, True
    full_ids = representatives[members]
    representative = members[np.lexsort((full_ids, -(dirs @ (center / norm))))[0]]
    return int(representative), False


def extract_plateaus(values, weak_vertices, adjacent, directions, representatives):
    """Return one vertex per valid plateau, plus the unchanged degeneracy count."""
    emitted = []
    degenerate = 0
    for members in equal_components(values, weak_vertices, adjacent):
        representative, invalid = plateau_representative(
            values, members, adjacent, directions, representatives
        )
        degenerate += int(invalid)
        if representative is not None:
            emitted.append(
                (representative, float(values[representative]), len(members))
            )
    return emitted, degenerate


class QuotientExtractor:
    """Direct topology-correct maxima and plateau extraction on RP2."""

    def __init__(self, source_base, geometry):
        self.geometry = geometry
        self.directions = geometry.directions[geometry.representatives]
        self.representatives = geometry.representatives
        self.neighbors = geometry.neighbors
        self.device_neighbors = jnp.asarray(self.neighbors)
        self.sample = None
        self.last_diagnostic = {}

    def _propose_values(self, values, limit):
        """Return strict/plateau maxima ordered by height, then original sphere ID.

        `seed_count` counts all maxima before the output limit is applied.
        """
        values = np.asarray(values).reshape(-1)
        strict, weak = map(
            np.asarray,
            jax.device_get(
                quotient_strict_and_weak(jnp.asarray(values), self.device_neighbors)
            ),
        )
        strict_q = np.flatnonzero(strict)
        items = [(int(q), float(values[q])) for q in strict_q]
        weak_q = np.flatnonzero(weak)
        emitted, degenerate = [], 0
        if len(weak_q):
            emitted, degenerate = extract_plateaus(
                values,
                weak_q,
                self.geometry.adjacent,
                self.directions,
                self.representatives,
            )
            items.extend((vertex, score) for vertex, score, _ in emitted)
        items.sort(key=lambda x: (-x[1], int(self.representatives[x[0]])))
        items = items[:limit]
        qids = np.asarray([q for q, _ in items], np.int32)
        full_ids = self.representatives[qids]
        self.last_diagnostic = {
            "weak_vertices": int(len(weak_q)),
            "added_components": len(emitted),
            "degenerate_components": degenerate,
            "components": [
                (int(self.representatives[q]), score, size)
                for q, score, size in emitted
            ],
            "direct_quotient": True,
        }
        return {
            "normals": self.directions[qids],
            "shells": np.zeros(len(qids), np.int32),
            "scores": np.asarray([score for _, score in items]),
            "indices": full_ids,
            "seed_count": int(np.sum(strict)) + len(emitted),
            "field_sampling_vertices": len(self.directions),
        }

    def propose(self, coefficients, native_field, limit=256):
        return self._propose_values(
            jax.device_get(self.sample(coefficients, native_field))[0], limit
        )
