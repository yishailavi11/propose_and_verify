# FAST, written for reading

A standalone CPU implementation of the same FAST algorithm, organized into short Python functions. NumPy, SciPy and DUCC handle geometry and harmonic fields; trimesh and NetworkX provide the projective graph. JAX preserves the optimized implementation's random samples and float32 normalization. Numba compiles one neighborhood lookup loop.

## Run

From the repository root with Python 3.11:

```bash
python3.11 -m venv .venv-simple
.venv-simple/bin/pip install -r src_simple/requirements.txt
JAX_PLATFORMS=cpu .venv-simple/bin/python -m src_simple
```

This generates 100,000 surface samples from an unequal-sided box and prints its three reflection planes. To use a prepared point cloud:

```bash
JAX_PLATFORMS=cpu .venv-simple/bin/python -m src_simple points.npy
JAX_PLATFORMS=cpu .venv-simple/bin/python -m src_simple points.npz --single-plane
```

An NPZ must contain a `points` array. Use 100,000 points for the paper's input setting. Returned planes satisfy `normal @ x = offset` in the input coordinate frame. The `p99` score is normalized by the observed bounding-box diagonal; single-plane mode selects the lowest-score candidate without an acceptance cutoff.

## Read the code

Start with [`detect.py`](detect.py), which calls the algorithm's stages in order. Settings and numerical tolerances are named in [`constants.py`](constants.py).

| File | Purpose |
|---|---|
| [`normalization.py`](normalization.py) | Center and normalize the point cloud |
| [`sampling.py`](sampling.py) | Sample point pairs and independent witnesses |
| [`voting.py`](voting.py) | Weight pair-induced planes and accumulate angular votes |
| [`harmonics.py`](harmonics.py) | Project, smooth and evaluate the harmonic field |
| [`readout.py`](readout.py) | Find peaks and plateaus; suppress nearby candidates |
| [`refinement.py`](refinement.py) | Refine candidate orientations |
| [`detect.py`](detect.py) | Verify candidates and return planes |

The package has no imports from `fast_symmetry` and no custom C++ or CUDA source. Run Python from the repository root, or copy `src_simple` and run from its parent directory. This version prioritizes readability; the optimized backends provide faster inference.
