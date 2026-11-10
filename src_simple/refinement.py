"""Paper §9.2: one intrinsic R2 step, expressed with small NumPy operations."""

import numpy as np

from .constants import (
    CURVATURE_ABSOLUTE_TOLERANCE,
    CURVATURE_RELATIVE_TOLERANCE,
    GRADIENT_ABSOLUTE_TOLERANCE,
    GRADIENT_RELATIVE_TOLERANCE,
    HESSIAN_CONDITION_LIMIT,
    HESSIAN_DETERMINANT_TOLERANCE,
    REFINEMENT_DENOMINATOR_EPSILON,
    REFINEMENT_STENCIL_DEGREES,
    REFINEMENT_TRIAL_SCALES,
    REFINEMENT_TRUST_DEGREES,
)
from .harmonics import evaluate


def tangent_basis(normal):
    """Two orthonormal tangent axes, using the least-aligned Cartesian axis."""
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return np.stack([first, np.cross(normal, first)])


def exp_map(normal, basis, tangent):
    """Map [...,2] tangent vectors onto the sphere without pole singularities."""
    angle = np.linalg.norm(tangent, axis=-1)
    moved = np.cos(angle)[..., None] * normal + np.sinc(angle / np.pi)[..., None] * (
        tangent @ basis
    )
    return moved / np.linalg.norm(moved, axis=-1, keepdims=True)


def local_quadratic(coefficients, normal, basis):
    """Sample a 3×3 tangent grid; NumPy differentiates its local quadratic.

    With three samples per axis, edge_order=2 gives the same centered gradient,
    diagonal second differences, and four-corner mixed difference as the paper.
    """
    step = np.deg2rad(REFINEMENT_STENCIL_DEGREES)
    axis = np.array([-step, 0, step])
    tangents = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    directions = exp_map(normal, basis, tangents).reshape(-1, 3)
    values = evaluate(coefficients, directions).reshape(3, 3)
    first = np.array(np.gradient(values, step, edge_order=2))
    second = np.array([np.gradient(part, step, edge_order=2) for part in first])
    return values[1, 1], first[:, 1, 1], second[:, :, 1, 1]


def refine_normal(coefficients, normal):
    """Fit a stable concave quadratic; try full/half/quarter bounded steps."""
    basis = tangent_basis(normal)
    center, gradient, hessian = local_quadratic(coefficients, normal, basis)
    if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
        return normal
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition = abs(eigenvalues[0]) / max(
        abs(eigenvalues[1]), REFINEMENT_DENOMINATOR_EPSILON
    )
    if (
        eigenvalues[1]
        >= -max(
            abs(center) * CURVATURE_RELATIVE_TOLERANCE, CURVATURE_ABSOLUTE_TOLERANCE
        )
        or condition >= HESSIAN_CONDITION_LIMIT
    ):
        return normal
    if np.linalg.norm(gradient) <= max(
        abs(center) * GRADIENT_RELATIVE_TOLERANCE, GRADIENT_ABSOLUTE_TOLERANCE
    ):
        return normal
    # Production's guarded determinant rejects a singular Newton update.
    if abs(np.linalg.det(hessian)) <= HESSIAN_DETERMINANT_TOLERANCE:
        return normal
    delta = np.linalg.solve(hessian, -gradient)
    radius = np.deg2rad(REFINEMENT_TRUST_DEGREES)
    delta *= min(1, radius / max(np.linalg.norm(delta), REFINEMENT_DENOMINATOR_EPSILON))
    trials = exp_map(normal, basis, np.array(REFINEMENT_TRIAL_SCALES)[:, None] * delta)
    values = evaluate(coefficients, trials)
    best = np.argmax(values)
    return (
        trials[best] if np.isfinite(values[best]) and values[best] > center else normal
    )


def refine(coefficients, normals):
    """Refine each peak once; geometric verification may still keep its original."""
    return np.asarray(
        [refine_normal(coefficients, normal) for normal in normals]
    ).reshape(-1, 3)
