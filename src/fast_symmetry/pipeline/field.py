"""Build the weighted angular field and evaluate it on the projective graph."""

import time

import jax
import jax.numpy as jnp

from .types import HarmonicField


def build_field(runtime, shape, seed, timing):
    """Run the selected frontend, then synthesize 20,481 orientation scores.

    CPU voting and harmonic analysis are separately timed. The production GPU
    frontend fuses them into one JIT, so its coefficient-only timer stays zero.
    """
    started = time.perf_counter_ns()
    if runtime.cpu:
        histogram = runtime.cpu_histogram(shape.normalized, int(seed), resolution=96)
        timing["pair_witness_ms"] = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        coefficients = jax.block_until_ready(runtime.coefficients(histogram))
        timing["harmonic_coefficients_ms"] = (time.perf_counter_ns() - started) / 1e6
    else:
        coefficients = jax.block_until_ready(
            runtime.frontend(shape.normalized, jnp.asarray(seed, jnp.int32))
        )
        timing["pair_witness_ms"] = (time.perf_counter_ns() - started) / 1e6
        timing["harmonic_coefficients_ms"] = 0.0
    started = time.perf_counter_ns()
    values = jax.block_until_ready(runtime.extractor.sample(coefficients, None))
    timing["icosphere_synthesis_ms"] = (time.perf_counter_ns() - started) / 1e6
    return HarmonicField(coefficients, values)
