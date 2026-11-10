"""One observation through FAST, expressed in the same order as the method.

Start reading at run_detection. Each call below is one named numerical stage;
its module documents coordinate frames, array shapes and backend differences.
"""

import time

import numpy as np

from ..method import THRESHOLD
from .field import build_field
from .preparation import prepare_shape, validate_points
from .proposals import extract_maxima, select_candidates
from .records import build_record
from .refinement import refine_candidates
from .verification import verify_candidates


def run_detection(runtime, points, seed, *, single_plane=False):
    """Run the frozen detector; only single-plane retrieval disables the cutoff."""
    threshold = np.inf if single_plane else THRESHOLD
    points = validate_points(points)
    started = time.perf_counter_ns()
    timing = {}
    shape = prepare_shape(runtime, points, threshold, timing, started)
    field = build_field(runtime, shape, seed, timing)
    maxima = extract_maxima(runtime, field, timing)
    selection = select_candidates(runtime, shape, maxima, timing)
    refinement = refine_candidates(runtime, field, selection, timing)
    verified = verify_candidates(
        runtime, shape, selection, refinement, threshold, timing
    )
    timing["total_ms"] = (time.perf_counter_ns() - started) / 1e6
    runtime.state = dict(
        values=field.values,
        coeff=field.coefficients,
        pair=None,
        witness=None,
        proposal=maxima,
    )
    return build_record(runtime, shape, selection, refinement, verified, seed, timing)
