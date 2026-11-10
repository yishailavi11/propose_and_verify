"""Exact score-only cuBQL verifier for the selected angular detector."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ...native_build import (
    cached_binary_matches,
    compile_shared_library,
    source_fingerprint,
)
from .abi import (
    CUBQL_REVISION,
)


def _default_cubql_root() -> Path:
    root = os.environ.get("FAST_SYMMETRY_CUBQL_ROOT")
    if not root:
        raise RuntimeError(
            "set FAST_SYMMETRY_CUBQL_ROOT to the pinned cuBQL checkout; "
            f"required revision: {CUBQL_REVISION}"
        )
    return Path(root)


def _default_build_root() -> Path:
    root = os.environ.get("FAST_SYMMETRY_BUILD_ROOT")
    if root:
        return Path(root) / "angular_cubql"
    return Path.home() / ".cache" / "fast_symmetry" / "angular_cubql"


def build_angular_gpu_library(
    cubql_root: Path | None = None,
    build_root: Path | None = None,
) -> Path:
    """Build the selected exact p99/p90 cuBQL implementation."""
    cubql_root = _default_cubql_root() if cubql_root is None else Path(cubql_root)
    build_root = _default_build_root() if build_root is None else Path(build_root)
    revision = subprocess.check_output(
        ["git", "-C", str(cubql_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CUBQL_REVISION:
        raise RuntimeError(f"cuBQL revision {revision}, expected {CUBQL_REVISION}")
    build_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "native" / "angular_spatial_cubql.cu"
    source_hash = source_fingerprint(source.parent, {".cu", ".cuh"})
    compiler = os.environ.get("NVCC", "nvcc")
    architecture = os.environ.get("FAST_SYMMETRY_CUDA_ARCH", "sm_86")
    output = build_root / "libfast_angular_cubql.so"
    flags = [
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        f"-arch={architecture}",
        "-DFAST_SYMMETRY_INITIAL_SKETCH=512",
        "-DFAST_SYMMETRY_LEAF_THRESHOLD=10",
        f"-I{cubql_root}",
    ]
    build = {
        "schema_version": 1,
        "cubql_revision": revision,
        "source_sha256": source_hash,
        "compiler": compiler,
        "architecture": architecture,
        "flags": flags,
    }
    if cached_binary_matches(output, build):
        return output
    return compile_shared_library(source, output, build)
