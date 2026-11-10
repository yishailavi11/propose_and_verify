"""Public API and resource lifetime; the numerical recipe is pipeline/detection.py."""

import hashlib

import numpy as np

from .pipeline.detection import run_detection
from .pipeline.setup import create_runtime
from .results import ProductionDetectionResult, ProductionPrediction, from_record


class _Detector:
    """Shared CPU/GPU API; backend resources are created once and reused serially."""

    def __init__(self, runtime):
        self._runtime = runtime

    @property
    def state(self):
        """Last field/coefficients/proposal, retained for numerical diagnostics."""
        return self._runtime.state

    @property
    def readout(self):
        """The immutable selected detector configuration."""
        return self._runtime.readout

    @property
    def setup_ms(self):
        return self._runtime.setup_ms

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Finish outstanding work and release synthesis resources."""
        self._runtime.executor.shutdown()
        synthesis = self._runtime.harmonic_synthesis
        if hasattr(synthesis, "close"):
            synthesis.close()

    def _detect(self, points, seed, *, single_plane=False):
        """Raw-record entry point retained for the existing reproduction scripts."""
        return run_detection(self._runtime, points, seed, single_plane=single_plane)

    def detect(self, points, *, seed=None):
        """Recover all verified planes in the input's world coordinates.

        Args:
            points: Finite float32 observation with shape (100000, 3).
            seed: Pair/witness RNG seed; defaults to the frozen configuration.

        Returns:
            ProductionDetectionResult with accepted planes and stage diagnostics.
        """
        pair_seed = self.readout.pair_seed if seed is None else int(seed)
        return from_record(self._detect(points, pair_seed))

    def detect_single(self, points, *, seed=None):
        """Return the lowest exact-p99 candidate without an acceptance threshold.

        Returns the existing candidate dictionary, or None if no maxima survive.
        The same geometric proposal/refinement pipeline is used by detect().
        """
        pair_seed = self.readout.pair_seed if seed is None else int(seed)
        record = self._detect(points, pair_seed, single_plane=True)
        return min(
            record["candidates"], key=lambda candidate: candidate["p99"], default=None
        )

    def diagnostics(self, result):
        """Attach the coefficient hash and return the ordered maxima arrays."""
        record = (
            result.to_dict()
            if isinstance(result, ProductionDetectionResult)
            else result
        )
        proposal = self.state["proposal"]
        record["coefficient_sha256"] = hashlib.sha256(
            np.asarray(self.state["coeff"]).tobytes()
        ).hexdigest()
        maxima = {
            "indices": np.asarray(proposal["indices"], np.int32),
            "heights": np.asarray(proposal["scores"], np.float32),
        }
        if isinstance(result, ProductionDetectionResult):
            result.diagnostics.update(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ["predictions", "graph"]
                }
            )
        return maxima


class AngularDetector3D(_Detector):
    """The selected FAST detector on a CUDA JAX device."""

    def __init__(
        self,
        *,
        cache_dir=".fast_symmetry_cache/angular",
        cubql_library=None,
        cubql_root=None,
    ):
        super().__init__(
            create_runtime(
                cpu=False,
                cache_dir=cache_dir,
                cubql_library=cubql_library,
                cubql_root=cubql_root,
            )
        )


class AngularCPUDetector3D(_Detector):
    """The same method with fused C++ voting, DUCC and exact CPU verification."""

    def __init__(
        self, *, cache_dir=".fast_symmetry_cache/angular", histogram_library=None
    ):
        super().__init__(
            create_runtime(
                cpu=True, cache_dir=cache_dir, cpu_histogram_library=histogram_library
            )
        )


def detect_angular_reflection_symmetries_3d(
    points,
    *,
    seed=None,
    cache_dir=".fast_symmetry_cache/angular",
    cubql_library=None,
    cubql_root=None,
):
    """One-shot CUDA convenience API; reuse a detector for repeated inference."""
    with AngularDetector3D(
        cache_dir=cache_dir, cubql_library=cubql_library, cubql_root=cubql_root
    ) as detector:
        return detector.detect(points, seed=seed)


def detect_angular_reflection_symmetries_3d_cpu(
    points,
    *,
    seed=None,
    cache_dir=".fast_symmetry_cache/angular",
    histogram_library=None,
):
    """One-shot CPU convenience API; reuse a detector for repeated inference."""
    with AngularCPUDetector3D(
        cache_dir=cache_dir, histogram_library=histogram_library
    ) as detector:
        return detector.detect(points, seed=seed)


__all__ = [
    "AngularDetector3D",
    "AngularCPUDetector3D",
    "ProductionDetectionResult",
    "ProductionPrediction",
    "detect_angular_reflection_symmetries_3d",
    "detect_angular_reflection_symmetries_3d_cpu",
]
