"""Data passed between detector stages.

World coordinates are used only for final plane equations and exact verification.
Voting and the harmonic field use centroid-centered, diagonal-normalized points.
A PlaneRecord is the existing JSON-compatible diagnostic dictionary; its schema
is deliberately preserved by this organizational refactor.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

from ..config import AngularConfig

PlaneRecord = dict[str, Any]
Timings = dict[str, float]


@dataclass
class DetectorRuntime:
    """Reusable, shape-independent resources; one instance is used serially."""

    cpu: bool
    readout: AngularConfig
    cache_dir: Path
    executor: ThreadPoolExecutor
    library: Path | None
    base: Any
    extractor: Any
    quotient: Any
    harmonic_synthesis: Callable
    coefficients: Callable
    frontend: Callable | None
    cpu_histogram: Callable | None
    setup_ms: float
    state: dict | None = None


@dataclass
class PreparedShape:
    """One observation in both coordinate frames, plus its pending NN index."""

    points: np.ndarray
    device_points: jax.Array
    normalized: jax.Array
    center: np.ndarray
    diagonal: float
    verifier_future: Future


@dataclass
class HarmonicField:
    """Complex coefficients [1,49,49] and values [1,20481] on RP2."""

    coefficients: jax.Array
    values: jax.Array


@dataclass
class CandidateSelection:
    """Ordered maxima before the floor, after the floor, and after NMS."""

    proposal: dict
    maximum: float
    raw: list[PlaneRecord]
    above_floor: list[PlaneRecord]
    selected: list[PlaneRecord]


@dataclass
class RefinedCandidates:
    """Original/refined float32 normals [K,3] and one R2 diagnostic per plane."""

    original: np.ndarray
    refined: np.ndarray
    diagnostics: dict[str, np.ndarray]
    capacity: int


@dataclass
class VerifiedCandidates:
    """All scored candidates and the accepted subset, in proposal order."""

    candidates: list[PlaneRecord]
    predictions: list[PlaneRecord]
