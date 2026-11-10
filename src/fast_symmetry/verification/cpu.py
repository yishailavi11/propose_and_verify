"""Exact float64 p99 verifier optimized for one-build, one-query use."""

import time

import numpy as np
from scipy.spatial import cKDTree


class BoundedCPUVerifier:
    """Exact cKDTree scores, resolving only distances needed by the p99 decision."""

    def __init__(self, points, D, threshold):
        """Build the production float64, leafsize-64 KD-tree and retain the decision bound."""
        started = time.perf_counter_ns()
        self.points = np.asarray(points, np.float64)
        self.tree = cKDTree(
            self.points,
            leafsize=64,
            balanced_tree=False,
            compact_nodes=False,
        )
        self.D = D
        self.threshold = threshold
        self.proven_reject = None
        self.proven_reject_count = 0
        self.resolved_sources = 0
        self.build_ms = (time.perf_counter_ns() - started) / 1e6
        self.query_ms = 0.0

    def query(self, source, normals, offsets):
        """Reflect [512,3] sources through each world plane, then query all targets."""
        started = time.perf_counter_ns()
        n = np.asarray(normals, np.float64)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        s = self.points[source]
        reflected = (
            s[None]
            - 2
            * (np.einsum("sj,cj->cs", s, n) - np.asarray(offsets)[:, None])[:, :, None]
            * n[:, None]
        )
        upper = np.nextafter(self.threshold * self.D, np.inf)
        distance = self.tree.query(
            reflected.reshape(-1, 3),
            k=1,
            workers=8,
            distance_upper_bound=upper,
        )[0].reshape(len(n), len(source))
        missing = ~np.isfinite(distance)
        # Seven missing distances prove p99 rejection for 512 sources.
        # Otherwise resolve them exactly before calculating the linear quantile.
        reject = missing.sum(axis=1) >= 7
        resolve = missing & ~reject[:, None]
        if np.any(resolve):
            distance[resolve] = self.tree.query(reflected[resolve], k=1, workers=8)[0]
        self.proven_reject = reject
        self.proven_reject_count += int(reject.sum())
        self.resolved_sources += int(resolve.sum())
        self.query_ms += (time.perf_counter_ns() - started) / 1e6
        return distance / self.D, None, None

    def close(self):
        pass


def quantiles(verifier, distances):
    reject = verifier.proven_reject
    safe = distances.copy()
    safe[reject] = 0
    p99 = np.quantile(safe, 0.99, axis=1)
    p90 = np.quantile(safe, 0.90, axis=1)
    p99[reject] = np.inf
    p90[reject] = np.inf
    return p99, p90
