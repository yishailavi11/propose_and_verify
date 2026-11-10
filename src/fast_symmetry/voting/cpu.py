"""Compiled CPU frontend for the selected one-channel angular detector."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..native_build import (
    cached_binary_matches,
    compile_shared_library,
    source_fingerprint,
)


def _default_build_root() -> Path:
    root = os.environ.get("FAST_SYMMETRY_BUILD_ROOT")
    if root:
        return Path(root) / "angular_cpu"
    return Path.home() / ".cache" / "fast_symmetry" / "angular_cpu"


def build_angular_cpu_library(build_root: Path | None = None) -> Path:
    """Build or reuse the OpenMP fused pair/witness/angular-CIC frontend."""
    build_root = _default_build_root() if build_root is None else Path(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "native" / "angular_witness.cpp"
    source_hash = source_fingerprint(source.parent, {".cpp", ".hpp"})
    compiler = os.environ.get("CXX", "g++")
    output = build_root / "libfast_angular_witness.so"
    build = {
        "schema_version": 1,
        "source_sha256": source_hash,
        "compiler": compiler,
        "flags": [
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            "-fopenmp",
        ],
    }
    if cached_binary_matches(output, build):
        return output
    return compile_shared_library(source, output, build)


@lru_cache(maxsize=None)
def _load(path: Path):
    """Register the fused C ABI once per shared-library path."""
    library = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    function = library.witness_histogram_all_fused
    function.argtypes = [
        fp,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        fp,
        ctypes.c_int,
        ctypes.c_int,
        fp,
        ip,
        ip,
        ip,
    ]
    function.restype = ctypes.c_int
    return function


class FusedAngularHistogram:
    """Callable exact-RNG CPU implementation of pair voting and angular CIC."""

    def __init__(self, mu_nodes, n_phi, *, cutoff, library_path=None, build_root=None):
        self.mu_nodes = np.ascontiguousarray(np.asarray(mu_nodes), dtype=np.float32)
        self.n_phi = int(n_phi)
        self.cutoff = np.float32(cutoff)
        self.library_path = (
            build_angular_cpu_library(build_root)
            if library_path is None
            else Path(library_path)
        )
        self.function = _load(self.library_path)

    def __call__(self, points, seed, *, resolution=96, audit=False):
        """Build normalized angular mass [1,50,193] with the frozen C++/OpenMP frontend.

        audit=True additionally returns endpoint and witness indices for exact
        comparison with JAX Threefry; it does not change vote weights.
        """
        points = np.ascontiguousarray(np.asarray(points), dtype=np.float32)
        output = np.empty((1, len(self.mu_nodes), self.n_phi), dtype=np.float32)
        first = np.empty(250000, np.int32) if audit else None
        second = np.empty(250000, np.int32) if audit else None
        witnesses = np.empty((250000, 2), np.int32) if audit else None
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int32)
        null = ctypes.cast(0, ip)
        status = self.function(
            points.ctypes.data_as(fp),
            len(points),
            ctypes.c_uint32(seed),
            250000,
            int(resolution),
            self.cutoff,
            np.float32(0.01),
            self.mu_nodes.ctypes.data_as(fp),
            len(self.mu_nodes),
            self.n_phi,
            output.ctypes.data_as(fp),
            first.ctypes.data_as(ip) if audit else null,
            second.ctypes.data_as(ip) if audit else null,
            witnesses.ctypes.data_as(ip) if audit else null,
        )
        if status:
            raise RuntimeError(f"fused angular histogram failed with status {status}")
        histogram = jnp.asarray(output)
        return (histogram, first, second, witnesses) if audit else histogram
