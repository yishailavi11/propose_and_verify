"""Apply the unchanged one-step intrinsic R2 update to the surviving candidates."""

import time

import numpy as np

from ..harmonics.refinement import refine_padded
from .types import RefinedCandidates


def refine_candidates(runtime, field, selection, timing):
    """Prepare fixed-size JIT batches, retaining both original and refined normals."""
    started = time.perf_counter_ns()
    selected = selection.selected
    original = np.asarray([plane["normal"] for plane in selected], np.float32).reshape(
        -1, 3
    )
    channels = np.asarray([plane["shell"] for plane in selected], np.int32)
    capacities = (4, 8, 16, 32, 64) if runtime.cpu else (16, 32, 64)
    capacity = next(value for value in capacities if len(selected) <= value)
    refined, diagnostics = refine_padded(
        field.coefficients, channels, original, capacity=capacity
    )
    timing["R2_ms"] = (time.perf_counter_ns() - started) / 1e6
    return RefinedCandidates(original, refined, diagnostics, capacity)
