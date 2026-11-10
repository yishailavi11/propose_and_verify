"""ctypes ABI and lifetime management for the selected 3D CUDA verifier."""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CUBQL_REVISION = "20f9db19cdb160e2b83d8f05dfdb0437fe11fe24"
SOURCE_COUNT = 512
MAX_CANDIDATES = 128


class _Timings(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_float)
        for name in (
            "setup_ms",
            "boxes_ms",
            "build_ms",
            "input_ms",
            "reflection_ms",
            "traversal_ms",
            "percentile_ms",
            "output_ms",
            "total_ms",
        )
    ]


@dataclass(frozen=True)
class SpatialTimings:
    setup_ms: float
    boxes_ms: float
    build_ms: float
    input_ms: float
    reflection_ms: float
    traversal_ms: float
    percentile_ms: float
    output_ms: float
    total_ms: float

    @classmethod
    def from_c(cls, value: _Timings) -> "SpatialTimings":
        return cls(**{name: float(getattr(value, name)) for name, _ in value._fields_})


@lru_cache(maxsize=None)
def _load(path: Path) -> ctypes.CDLL:
    """Register context construction/destruction and verify the pinned cuBQL revision."""
    library = ctypes.CDLL(str(path))
    library.spatial_cubql_revision.restype = ctypes.c_char_p
    library.spatial_cubql_last_error.restype = ctypes.c_char_p
    library.spatial_cubql_build_device.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.POINTER(_Timings),
    ]
    library.spatial_cubql_build_device.restype = ctypes.c_void_p
    library.spatial_cubql_destroy.argtypes = [ctypes.c_void_p]
    if library.spatial_cubql_revision().decode() != CUBQL_REVISION:
        raise RuntimeError("loaded spatial library has the wrong cuBQL revision")
    return library


class FullCloudSpatialNN:
    @classmethod
    def from_device(
        cls,
        device_pointer: int,
        point_count: int,
        dimension: int,
        diagonal: float,
        stream_pointer: int = 0,
        library_path: Path | None = None,
    ) -> "FullCloudSpatialNN":
        """Build from an N-by-3 contiguous float32 CUDA buffer.

        The contiguous float32 device points are borrowed until close().
        """
        if library_path is None:
            raise ValueError("a built angular CUDA library is required")
        if not device_pointer or point_count < SOURCE_COUNT or dimension != 3:
            raise ValueError("invalid device point table")
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("point-cloud diagonal must be positive")
        self = cls.__new__(cls)
        self._library = _load(library_path)
        self.dimension = dimension
        self.points = None
        self.diagonal = float(diagonal)
        timings = _Timings()
        self._context = self._library.spatial_cubql_build_device(
            int(device_pointer),
            point_count,
            dimension,
            int(stream_pointer),
            ctypes.byref(timings),
        )
        if not self._context:
            raise RuntimeError(self._library.spatial_cubql_last_error().decode())
        self.build_timings = SpatialTimings.from_c(timings)
        return self

    def close(self) -> None:
        if getattr(self, "_context", None):
            self._library.spatial_cubql_destroy(self._context)
            self._context = None

    def _prepare_query(
        self,
        source_indices: np.ndarray,
        normals: np.ndarray,
        offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize each plane equation and create contiguous float32 CUDA inputs.

        Offsets are divided by normal length too, preserving the world plane.
        """
        sources = np.ascontiguousarray(source_indices, dtype=np.int32)
        raw_normals = np.asarray(normals, dtype=np.float64)
        raw_offsets = np.asarray(offsets, dtype=np.float64)
        if sources.shape != (SOURCE_COUNT,):
            raise ValueError("source_indices must have exactly 512 entries")
        if raw_normals.ndim != 2 or raw_normals.shape[1] != self.dimension:
            raise ValueError("normals have the wrong shape")
        count = len(raw_normals)
        if not 0 < count <= MAX_CANDIDATES or raw_offsets.shape != (count,):
            raise ValueError("require 1..128 normals and one offset per normal")
        lengths = np.linalg.norm(raw_normals, axis=1)
        if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("normals must be finite and nonzero")
        raw_offsets = raw_offsets / lengths
        raw_normals = raw_normals / lengths[:, None]
        gpu_normals = np.ascontiguousarray(raw_normals, dtype=np.float32)
        gpu_offsets = np.ascontiguousarray(raw_offsets, dtype=np.float32)
        return sources, gpu_normals, gpu_offsets
