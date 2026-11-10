"""Run a point cloud or a three-plane box example: python -m src_simple [points.npy]."""

import argparse
import json
from dataclasses import asdict

import numpy as np
import trimesh

from .constants import DEFAULT_PAIR_SEED, PREPARED_POINT_COUNT
from .detect import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "points", nargs="?", help=".npy [N,3], or .npz with a points array"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_PAIR_SEED)
    parser.add_argument("--single-plane", action="store_true")
    args = parser.parse_args()
    if args.points is None:
        mesh = trimesh.creation.box(extents=[1, 1.7, 2.3])
        points, _ = trimesh.sample.sample_surface(mesh, PREPARED_POINT_COUNT, seed=0)
    elif args.points.endswith(".npz"):
        with np.load(args.points) as table:
            points = table["points"]
    else:
        points = np.load(args.points)
    planes = detect(points, seed=args.seed, single_plane=args.single_plane)
    records = [{**asdict(plane), "normal": plane.normal.tolist()} for plane in planes]
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
