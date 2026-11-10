# Third-party dependencies

cuBQL: [NVIDIA/cuBQL](https://github.com/NVIDIA/cuBQL), revision `20f9db19cdb160e2b83d8f05dfdb0437fe11fe24`, Apache-2.0. It is fetched separately by `scripts/setup_gpu.sh`; its license remains in the checkout. This repository contains the FAST CUDA wrapper, not a vendored cuBQL implementation.

Numerical dependencies are installed as Python packages: NumPy, SciPy, JAX/jaxlib, trimesh and DUCC; GPU installations additionally use CuPy, cuFINUFFT and NVIDIA CUDA runtime packages. Each dependency retains its own license. Pinned package versions are listed in `requirements/`.

The standalone educational CPU example additionally uses NetworkX for projective graph components and boundaries. Its independently tested versions are in `src_simple/requirements.txt`.

The readable CPU implementation uses Numba and llvmlite to compile one ordinary Python neighborhood loop. Their tested versions are pinned in `src_simple/requirements.txt`; no external compiler or hand-written native source is required for that helper.
