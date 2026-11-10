"""Run FAST on a mesh or canonical 100,000-point float32 observation."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=20260929, help="Pair/witness RNG seed"
    )
    parser.add_argument(
        "--surface-seed", type=int, default=0, help="Mesh surface sampling seed"
    )
    parser.add_argument(
        "--single-plane",
        action="store_true",
        help="Lowest-p99 candidate, without an acceptance threshold",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".fast_symmetry_cache"))
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu" if args.backend == "cpu" else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import numpy as np

    from fast_symmetry import (
        AngularCPUDetector3D,
        AngularDetector3D,
        prepare_surface_points,
    )

    if args.input.suffix == ".npy":
        points = np.load(args.input)
    elif args.input.suffix == ".npz":
        with np.load(args.input) as data:
            points = data["points"] if "points" in data else None
        if points is None:
            points = prepare_surface_points(args.input, seed=args.surface_seed)
    else:
        points = prepare_surface_points(args.input, seed=args.surface_seed)
    cls = AngularCPUDetector3D if args.backend == "cpu" else AngularDetector3D
    with cls(cache_dir=args.cache_dir) as detector:
        # First call compiles; repeat calls on a reused instance amortize setup.
        if args.single_plane:
            result = detector.detect_single(points, seed=args.seed)
        else:
            result = detector.detect(points, seed=args.seed).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
