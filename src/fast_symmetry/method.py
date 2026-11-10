"""Frozen L48 angular FAST configuration with the original DEV100 cutoff.

Only the selected L48 angular method is implemented.
"""

METHOD_ID = "FAST_L48_FRAMEFIX_V1"
OFFSET_GATE_OVER_DIAGONAL = 0.010826736688613892
THRESHOLD = 0.014664287354662298
FLOOR = 0.01

METHOD = {
    "voting": "V2V-W2 Gaussian witness mass",
    "points": 100000,
    "pairs": 250000,
    "witnesses": 2,
    "tau_w": 0.01,
    "witness_grid": 96,
    "offset_gate_over_diagonal": OFFSET_GATE_OVER_DIAGONAL,
    "input_channels": 1,
    "input_grid": "50 Gauss-Legendre latitude x 193 longitude",
    "kernel": "closed-form projective boundary Gaussian",
    "kernel_formula": "exp(-(0.5*acos(abs(dot))/0.06)^2)",
    "bandlimit": 48,
    "active_degrees": list(range(0, 49, 2)),
    "icosphere_level": 6,
    "quotient_directions": 20481,
    "plateau": "exact projective connected components",
    "floor": FLOOR,
    "reference_topK": 256,
    "topK": 64,
    "NMS_angle_degrees": 6,
    "NMS_offset_D": 0.01,
    "NMS_coordinate_frame": "centered; all candidate offsets zero",
    "refined_world_offset": "dot(refined_normal, input_centroid)",
    "refinement": "one-step angular R2",
    "verification": "dual original/refined exact 512-source p99",
    "threshold": THRESHOLD,
    "CPU_backend": "fused C++ pair/witness/angular CIC + overlapped cKDTree",
    "GPU_backend": "fused JAX frontend + exact cuBQL p99/p90",
    "CPU_synthesis": "DUCC 0.41.0 synthesis_general, epsilon=2e-6, 8 threads",
    "GPU_synthesis": "double Fourier sphere + cuFINUFFT 2.5.1 FP64 NUFFT",
    "implementation_freeze": "angular-l48-framefix-v1-20261028",
    "rng": "JAX Threefry-compatible pair and witness streams",
}
