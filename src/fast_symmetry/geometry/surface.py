"""Canonical mesh preparation used for the paper's 100k-point observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

PAPER_SURFACE_POINT_COUNT = 100_000


def load_mesh_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh arrays with the exact benchmark loader and conversion rules."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            return np.asarray(data["vertices"], dtype=np.float64), np.asarray(
                data["faces"], dtype=np.int64
            )
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(
        loaded.faces, dtype=np.int64
    )


def sample_mesh_surface(
    path: str | Path,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replay the benchmark's float64 area-weighted folded-barycentric sampler."""
    vertices, faces = load_mesh_arrays(path)
    triangles = vertices[faces]
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        / 2
    )
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError(f"invalid triangle areas in {Path(path)}")
    chosen = rng.choice(len(faces), size=count, replace=True, p=areas / areas.sum())
    u = rng.random(count)
    v = rng.random(count)
    reflected = u + v > 1
    u[reflected] = 1 - u[reflected]
    v[reflected] = 1 - v[reflected]
    selected = triangles[chosen]
    return (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    )


def prepare_surface_points(
    path: str | Path,
    *,
    seed: int,
    n_points: int = PAPER_SURFACE_POINT_COUNT,
) -> np.ndarray:
    """Prepare the exact float32 surface observation consumed by production."""
    return sample_mesh_surface(path, int(n_points), np.random.default_rng(seed)).astype(
        np.float32
    )
