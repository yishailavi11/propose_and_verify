"""Turn field maxima into an ordered, bounded set of centroid-plane candidates."""

import time

import numpy as np

from ..method import FLOOR
from .types import CandidateSelection


def extract_maxima(runtime, field, timing):
    """Run exact projective maxima/plateau handling, reusing the sampled field."""
    started = time.perf_counter_ns()
    extractor = runtime.extractor
    sample = extractor.sample
    # The extractor's convenience API samples a field itself. Reuse the already
    # computed values to preserve its tie rules without evaluating twice.
    extractor.sample = lambda *_: field.values
    try:
        proposal = extractor.propose(field.coefficients, None, limit=256)
    finally:
        extractor.sample = sample
    timing["maxima_ms"] = (time.perf_counter_ns() - started) / 1e6
    return proposal


def make_centroid_planes(proposal, center):
    """Convert graph vertices into world-plane records, preserving peak order.

    A searched normal n defines the centered plane n·x=0, hence its world offset
    is n·center. The preserved `shell` diagnostic is always the one angular
    channel, indexed zero; it is not an offset/radial search coordinate.
    """
    maximum = float(proposal["scores"][0]) if len(proposal["scores"]) else 0.0
    raw = []
    for rank, (direction, channel, height, index) in enumerate(
        zip(
            proposal["normals"],
            proposal["shells"],
            proposal["scores"],
            proposal["indices"],
        )
    ):
        normal = direction.astype(float)
        normal /= np.linalg.norm(normal)
        raw.append(
            dict(
                native_id=int(index),
                raw_rank=rank + 1,
                shell=int(channel),
                normal=normal.tolist(),
                offset=float(normal @ center),
                height=float(height),
                relative_height=float(height / maximum) if maximum else 0.0,
            )
        )
    return raw, maximum


def select_candidates(runtime, shape, proposal, timing):
    """Apply the frozen 1% floor, top-64 cap, then ordered six-degree NMS."""
    started = time.perf_counter_ns()
    raw, maximum = make_centroid_planes(proposal, shape.center)
    above = [plane for plane in raw if plane["height"] >= FLOOR * maximum]
    selected = above[:64]
    timing["floor_top64_ms"] = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    if selected:
        # NMS operates in the centered frame: every candidate offset is zero.
        keep = runtime.readout.d1_indices(
            [plane["normal"] for plane in selected],
            [0.0 for plane in selected],
            shape.diagonal,
        )
        selected = [selected[int(index)] for index in keep]
    timing["NMS_ms"] = (time.perf_counter_ns() - started) / 1e6
    return CandidateSelection(proposal, maximum, raw, above, selected)
