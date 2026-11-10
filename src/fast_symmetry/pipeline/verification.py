"""Score original/refined world planes and select the exact lower-p99 hypothesis."""

import time
from dataclasses import dataclass

import numpy as np

from ..verification.cpu import quantiles as cpu_quantiles
from ..verification.cuda.quantiles import quantiles as gpu_quantiles
from ..verification.sources import deterministic_stride_indices
from .types import VerifiedCandidates


@dataclass
class DualHypotheses:
    """Two world-plane equations per candidate, with backend-preserved precision."""

    original: np.ndarray
    refined: np.ndarray
    offsets: np.ndarray
    refined_offsets: np.ndarray


@dataclass
class DualScores:
    """Arrays ordered [all refined hypotheses, all original hypotheses]."""

    p99: np.ndarray
    p90: np.ndarray
    proven_reject: np.ndarray


def make_hypotheses(runtime, shape, selected, refinement):
    """Keep CPU float64/GPU float32 verification boundaries exactly as evaluated."""
    dtype = np.float64 if runtime.cpu else np.float32
    offsets = np.asarray([plane["offset"] for plane in selected], dtype)
    original = np.asarray(refinement.original, dtype)
    refined = np.asarray(refinement.refined, dtype)
    refined_offsets = np.asarray(refined @ shape.center, dtype)
    return DualHypotheses(original, refined, offsets, refined_offsets)


def query_hypotheses(runtime, shape, verifier, hypotheses, timing):
    """Query the same 512 sources against the full target cloud for both versions."""
    count = len(hypotheses.original)
    if not count:
        return DualScores(np.empty(0), np.empty(0), np.empty(0, bool))
    distances, _, _ = verifier.query(
        deterministic_stride_indices(len(shape.points)),
        np.r_[hypotheses.refined, hypotheses.original],
        np.r_[hypotheses.refined_offsets, hypotheses.offsets],
    )
    p99, p90 = (
        cpu_quantiles(verifier, distances)
        if runtime.cpu
        else gpu_quantiles(verifier, distances)
    )
    if runtime.cpu:
        timing["verifier_query_actual_ms"] = verifier.query_ms
    reject = np.asarray(getattr(verifier, "proven_reject", np.zeros(2 * count, bool)))
    return DualScores(p99, p90, reject)


def choose_hypothesis(index, plane, hypotheses, scores, diagnostics, threshold):
    """Build one existing diagnostic record; exact p99 ties retain refinement.

    The bounded CPU verifier can prove both versions exceed the multi-plane
    cutoff without resolving their exact residuals. Such scores remain null in
    JSON and the candidate is marked rejected. Single-plane mode is unbounded.
    """
    count = len(hypotheses.original)
    s1, s0 = float(scores.p99[index]), float(scores.p99[index + count])
    accepted = min(s0, s1) <= threshold
    choose0 = s0 < s1
    choice = (
        ("original" if choose0 else "refined")
        if np.isfinite(min(s0, s1))
        else "unresolved_both_proven_rejected"
    )
    original = hypotheses.original
    refined = hypotheses.refined
    normal = original[index] if choose0 else refined[index]
    original_unit = original[index] / np.linalg.norm(original[index])
    refined_unit = refined[index] / np.linalg.norm(refined[index])
    update = float(np.degrees(np.arccos(np.clip(original_unit @ refined_unit, -1, 1))))
    return {
        **plane,
        "normal_original": original[index].astype(float).tolist(),
        "normal_refined": refined[index].astype(float).tolist(),
        "offset_original": float(hypotheses.offsets[index]),
        "offset_refined": float(hypotheses.refined_offsets[index]),
        "offset": float(
            hypotheses.offsets[index] if choose0 else hypotheses.refined_offsets[index]
        ),
        "normal": normal.astype(float).tolist(),
        "selected_hypothesis": choice,
        "accepted": bool(accepted),
        "p90": float(scores.p90[index + count] if choose0 else scores.p90[index])
        if accepted
        else None,
        "p99_original": s0 if np.isfinite(s0) else None,
        "p99_refined": s1 if np.isfinite(s1) else None,
        "original_proven_rejected": bool(scores.proven_reject[index + count]),
        "refined_proven_rejected": bool(scores.proven_reject[index]),
        "p99": min(s0, s1) if np.isfinite(min(s0, s1)) else None,
        "R2_update_degrees": update,
        "optimization": {
            key: np.asarray(value[index]).item() for key, value in diagnostics.items()
        },
    }


def verify_candidates(runtime, shape, selection, refinement, threshold, timing):
    """Join the overlapped index build, score dual planes, select and accept."""
    started = time.perf_counter_ns()
    verifier = shape.verifier_future.result()
    if not runtime.cpu:
        verifier.diagonal = shape.diagonal
    timing["verifier_build_or_wait_ms"] = (time.perf_counter_ns() - started) / 1e6
    if runtime.cpu:
        timing["verifier_build_actual_ms"] = verifier.build_ms
    started = time.perf_counter_ns()
    selected = selection.selected
    hypotheses = make_hypotheses(runtime, shape, selected, refinement)
    try:
        scores = query_hypotheses(runtime, shape, verifier, hypotheses, timing)
        candidates = [
            choose_hypothesis(
                index, plane, hypotheses, scores, refinement.diagnostics, threshold
            )
            for index, plane in enumerate(selected)
        ]
        predictions = [plane for plane in candidates if plane["accepted"]]
    finally:
        verifier.close()
    timing["dual_verification_selection_ms"] = (time.perf_counter_ns() - started) / 1e6
    return VerifiedCandidates(candidates, predictions)
