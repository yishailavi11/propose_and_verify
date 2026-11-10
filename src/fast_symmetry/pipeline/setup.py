"""Create reusable resources once; no observation-specific inference lives here."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax

from ..config import AngularConfig
from ..geometry.projective import icosphere_quotient_extractor
from ..harmonics.analysis import build_coefficient_transform
from ..harmonics.cpu_synthesis import CPUHarmonicSynthesis
from ..harmonics.gpu_synthesis import GPUHarmonicSynthesis
from ..method import METHOD, OFFSET_GATE_OVER_DIAGONAL
from ..verification.cuda.build import build_angular_gpu_library
from ..voting.cpu import FusedAngularHistogram
from ..voting.gpu import build_gpu_frontend
from .types import DetectorRuntime


def _check_configuration(readout):
    """Fail early if a supposedly frozen production setting has drifted."""
    assert (
        readout.pair_count == 250000
        and readout.prepared_point_count == 100000
        and readout.witnesses == 2
    )
    assert (
        readout.witness_power == 2
        and readout.witness_tau_over_diagonal == 0.01
        and readout.witness_grid_resolution == 96
    )
    assert readout.nms_angle_degrees == 6 and readout.nms_offset_over_diagonal == 0.01
    assert jax.config.jax_default_prng_impl == "threefry2x32"


def create_runtime(
    *, cpu, cache_dir, cubql_library=None, cubql_root=None, cpu_histogram_library=None
):
    """Build the graph, harmonic transforms, native backend and overlap worker.

    CPU uses C++ voting plus DUCC synthesis. GPU uses a fused JAX frontend,
    cuFINUFFT synthesis and the exact cuBQL verifier. Both use the same graph,
    coefficients, candidate rules and one-step R2 math.
    """
    if (jax.default_backend() == "cpu") != cpu:
        raise RuntimeError(
            "CPU inference requires JAX_PLATFORMS=cpu; GPU inference requires a CUDA JAX backend"
        )
    started = time.perf_counter_ns()
    cache = Path(cache_dir)
    readout = AngularConfig()
    library = None
    if not cpu:
        library = (
            Path(cubql_library)
            if cubql_library is not None
            else build_angular_gpu_library(
                Path(cubql_root) if cubql_root is not None else None,
                cache / "angular_cubql",
            )
        )
    executor = ThreadPoolExecutor(max_workers=1)
    base, extractor, quotient, directions = icosphere_quotient_extractor(
        METHOD["icosphere_level"]
    )
    assert len(extractor.directions) == METHOD["quotient_directions"]
    synthesis_class = CPUHarmonicSynthesis if cpu else GPUHarmonicSynthesis
    synthesis = synthesis_class(directions, bandlimit=METHOD["bandlimit"])
    extractor.sample = synthesis
    _check_configuration(readout)
    mu, coefficients = build_coefficient_transform()
    frontend = None if cpu else build_gpu_frontend(mu, coefficients, readout)
    cpu_histogram = (
        FusedAngularHistogram(
            mu,
            193,
            cutoff=OFFSET_GATE_OVER_DIAGONAL,
            library_path=cpu_histogram_library,
            build_root=cache / "angular_cpu",
        )
        if cpu
        else None
    )
    return DetectorRuntime(
        cpu=cpu,
        readout=readout,
        cache_dir=cache,
        executor=executor,
        library=library,
        base=base,
        extractor=extractor,
        quotient=quotient,
        harmonic_synthesis=synthesis,
        coefficients=coefficients,
        frontend=frontend,
        cpu_histogram=cpu_histogram,
        setup_ms=(time.perf_counter_ns() - started) / 1e6,
    )
