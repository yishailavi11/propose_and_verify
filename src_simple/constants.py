"""Frozen settings for the readable FAST implementation.

Keep independent of fast_symmetry so this example remains self-contained.
Distances are fractions of the observed AABB diagonal unless stated otherwise.
The gate and acceptance cutoff retain the production freeze's full precision.
"""

# Input protocol and deterministic pair/witness sampling.
PREPARED_POINT_COUNT = 100000
DEFAULT_PAIR_SEED = 20260929
PAIR_COUNT = 250000
WITNESS_COUNT = 2
WITNESS_RNG_TAG = 0x57544E53  # Production Threefry fold-in tag: ASCII "WTNS".

# Pair voting and local witness compatibility.
OFFSET_GATE_OVER_DIAGONAL = 0.010826736688613892
WITNESS_GRID_RESOLUTION = 96
MIN_VOXEL_STEP = 1e-7
WITNESS_SIGMA_OVER_DIAGONAL = 0.01
WITNESS_WEIGHT_POWER = 2
VOTE_DIRECTION_SHIFT = 0.5  # Positive scale before normalization; preserves rounding.
PAIR_NORMAL_SQUARED_EPSILON = 1e-16
DIRECTION_NORM_EPSILON = 1e-30
MASS_NORMALIZATION_EPSILON = 1e-30

# Angular mass grid, even harmonic field and Funk–Hecke quadrature.
LATITUDE_COUNT = 50
LONGITUDE_COUNT = 193
BANDLIMIT = 48
KERNEL_QUADRATURE_COUNT = 192
KERNEL_SIGMA_RADIANS = 0.12  # Production's half-angle sigma 0.06 in full-angle units.
FIELD_SYNTHESIS_EPSILON = 2e-6
REFINEMENT_SYNTHESIS_EPSILON = 1e-12

# Projective graph, peak selection and nonmaximum suppression.
ICOSPHERE_SUBDIVISIONS = 6
ICOSPHERE_MAX_DEGREE = 6
MAX_PLATEAU_VERTICES = 64
PLATEAU_CENTER_MIN_NORM = 1e-8
REFERENCE_PEAK_COUNT = 256
RELATIVE_PEAK_FLOOR = 0.01
CANDIDATE_PEAK_COUNT = 64
NMS_ANGLE_DEGREES = 6

# One intrinsic R2 update: stencil, curvature/gradient guards and trial lengths.
REFINEMENT_STENCIL_DEGREES = 1
REFINEMENT_TRUST_DEGREES = 3
REFINEMENT_TRIAL_SCALES = (1, 0.5, 0.25)
REFINEMENT_DENOMINATOR_EPSILON = 1e-12
CURVATURE_RELATIVE_TOLERANCE = 1e-4
CURVATURE_ABSOLUTE_TOLERANCE = 1e-10
HESSIAN_CONDITION_LIMIT = 1e6
GRADIENT_RELATIVE_TOLERANCE = 1e-5
GRADIENT_ABSOLUTE_TOLERANCE = 1e-10
HESSIAN_DETERMINANT_TOLERANCE = 1e-16

# Exact geometric verification and CPU library settings.
VERIFICATION_SOURCE_COUNT = 512
VERIFICATION_QUANTILE = 0.99
ACCEPTANCE_THRESHOLD_OVER_DIAGONAL = 0.014664287354662298
KDTREE_LEAF_SIZE = 64
CPU_WORKERS = 8
