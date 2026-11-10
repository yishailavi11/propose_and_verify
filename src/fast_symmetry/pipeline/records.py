"""Preserve the existing raw result schema independently of stage implementation."""

import numpy as np


def build_record(runtime, shape, selection, refinement, verified, seed, timing):
    """Collect counts, world-coordinate planes and diagnostics without recomputation."""
    count = len(selection.selected)
    return dict(
        seed=int(seed),
        center=np.asarray(shape.center).astype(float).tolist(),
        diagonal=shape.diagonal,
        raw_maxima=int(selection.proposal["seed_count"]),
        maximum_extracted_peak_height=selection.maximum,
        above_floor_in_top256=len(selection.above_floor),
        top64_count=min(64, len(selection.above_floor)),
        NMS_count=count,
        refined_count=count,
        verifier_candidate_count=count,
        verifier_hypotheses=2 * count,
        accepted_count=len(verified.predictions),
        raw_top256=selection.raw,
        candidates=verified.candidates,
        predictions=verified.predictions,
        plateau=runtime.extractor.last_diagnostic,
        padding_capacity=refinement.capacity,
        timing=timing,
    )
