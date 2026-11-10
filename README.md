# Propose & Verify: Fast Reflection Symmetry Detection via Harmonic Fields

FAST detects reflection symmetry planes in 3D shapes by proposing planes from a smoothed harmonic field and verifying them geometrically. This repository provides the paper's main method on **CPU and NVIDIA GPU**, together with a simpler CPU implementation for reading and experimentation. No training or pretrained weights are required.

## Installation

Tested on Linux x86-64 with **Python 3.11**. Run the commands below from the repository root. CPU and GPU installations use separate environments; choose the one you need.

### CPU

Requires a C++17 compiler with OpenMP (`g++`).

```bash
python3.11 -m venv .venv-cpu
.venv-cpu/bin/pip install -r requirements/cpu.txt
.venv-cpu/bin/pip install .
```

### GPU

Requires an NVIDIA GPU, a CUDA 12-compatible driver, and a CUDA 12 toolkit with `nvcc`. The setup script downloads the pinned [cuBQL](https://github.com/NVIDIA/cuBQL) dependency.

```bash
python3.11 -m venv .venv-gpu
.venv-gpu/bin/pip install -r requirements/gpu.txt
.venv-gpu/bin/pip install '.[gpu]'
bash scripts/setup_gpu.sh
export FAST_SYMMETRY_CUBQL_ROOT="$PWD/external/cuBQL"
export NVCC=/usr/local/cuda/bin/nvcc
export FAST_SYMMETRY_CUDA_ARCH=sm_86  # RTX A5000; adjust for your GPU
```

Native helpers compile automatically on first use. First-call timings include setup and compilation; reuse a detector when processing multiple shapes.

## Quick start

Run the included box example, which has three reflection symmetry planes:

```bash
.venv-cpu/bin/python scripts/detect.py examples/box.obj \
  --backend cpu --output outputs/box.json
```

For GPU inference, use `.venv-gpu/bin/python` and `--backend gpu`.

Run on your own mesh, or retrieve just the best plane:

```bash
.venv-cpu/bin/python scripts/detect.py shape.obj \
  --backend cpu --output outputs/planes.json
.venv-cpu/bin/python scripts/detect.py shape.obj \
  --backend cpu --single-plane --output outputs/one-plane.json
```

Mesh inputs are sampled to 100,000 surface points. Prepared point clouds are also supported: `.npy` arrays of shape `(100000, 3)` and dtype `float32`, or `.npz` files containing a `points` array.

Each output plane has a unit `normal` and an `offset` in the input coordinate frame, satisfying `normal @ x = offset`. The `p99` score is the 99th percentile of reflected nearest-neighbor distances, divided by the input bounding-box diagonal; lower is better. Single-plane mode returns the lowest-score candidate without applying the multi-plane acceptance cutoff.

### Python API

In the CPU environment (`JAX_PLATFORMS=cpu`):

```python
from fast_symmetry import AngularCPUDetector3D, prepare_surface_points

points = prepare_surface_points("examples/box.obj", seed=0)
with AngularCPUDetector3D() as detector:
    result = detector.detect(points)
    for plane in result.predictions:
        print(plane.normal, plane.offset, plane.p99)
    best_plane = detector.detect_single(points)
```

For GPU inference, use `AngularDetector3D` in the GPU environment with `JAX_PLATFORMS=cuda`.

## Simpler implementation

[`src_simple`](src_simple/README.md) follows the same algorithm in short Python functions using NumPy, SciPy and DUCC, with library support for geometry, sampling and a compiled neighborhood lookup. Start with [`detect.py`](src_simple/detect.py); the method's settings are documented in [`constants.py`](src_simple/constants.py).

```bash
python3.11 -m venv .venv-simple
.venv-simple/bin/pip install -r src_simple/requirements.txt
JAX_PLATFORMS=cpu .venv-simple/bin/python -m src_simple
```

This runs a built-in box example. The simpler version favors readability over speed; use the optimized CPU or GPU backend for faster inference.

## License

[MIT](LICENSE). Third-party dependencies retain their own [licenses](THIRD_PARTY.md).
