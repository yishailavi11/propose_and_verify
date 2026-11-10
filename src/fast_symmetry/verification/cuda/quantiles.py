"""Exact, GPU-resident linear p90/p99 queries; only scores cross to the host."""

import ctypes
from dataclasses import dataclass

import numpy as np

from .abi import (
    MAX_CANDIDATES,
    SOURCE_COUNT,
    FullCloudSpatialNN,
    SpatialTimings,
    _Timings,
)


@dataclass(frozen=True)
class QuantileScores:
    p99: np.ndarray
    p90: np.ndarray


def _register(library):
    """Describe the exact-quantile C ABI and its host output buffers."""
    function = library.spatial_cubql_query_quantiles
    float_p = ctypes.POINTER(ctypes.c_float)
    int_p = ctypes.POINTER(ctypes.c_int)
    function.argtypes = [
        ctypes.c_void_p,
        int_p,
        float_p,
        float_p,
        ctypes.c_int,
        ctypes.c_float,
        float_p,
        float_p,
        ctypes.POINTER(_Timings),
    ]
    function.restype = ctypes.c_int
    return function


class AngularExactQuantileSpatialNN(FullCloudSpatialNN):
    """Return exact NumPy-linear p99/p90 without copying all distances."""

    def query(self, source_indices, normals, offsets):
        """Reflect every source for both hypotheses; return exact linear quantile arrays."""
        sources, gpu_normals, gpu_offsets = self._prepare_query(
            source_indices, normals, offsets
        )
        count = len(gpu_normals)
        if not 0 < count <= MAX_CANDIDATES or sources.shape != (SOURCE_COUNT,):
            raise ValueError("invalid exact-quantile query")
        p99 = np.empty(count, dtype=np.float32)
        p90 = np.empty(count, dtype=np.float32)
        timings = _Timings()
        function = _register(self._library)
        result = function(
            self._context,
            sources.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            gpu_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gpu_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            count,
            self.diagonal,
            p99.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p90.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(timings),
        )
        if result:
            raise RuntimeError(self._library.spatial_cubql_last_error().decode())
        return QuantileScores(p99, p90), p90, SpatialTimings.from_c(timings)


def quantiles(_verifier, result):
    if not isinstance(result, QuantileScores):
        raise TypeError("exact quantile verifier received an unexpected result")
    return result.p99, result.p90
