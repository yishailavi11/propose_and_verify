"""Mass-preserving cloud-in-cell deposition on the angular GL/longitude grid."""

import jax.numpy as jnp


def angular_histogram(votes, mass, mu_nodes, n_phi):
    """CIC-deposit sign-canonical pair normals on the angular grid."""
    safe = jnp.maximum(jnp.linalg.norm(votes, axis=1), jnp.finfo(votes.dtype).tiny)
    mu = jnp.clip(votes[:, 2] / safe, -1, 1)
    phi = jnp.mod(jnp.arctan2(votes[:, 1], votes[:, 0]), 2 * jnp.pi)
    upper = jnp.clip(jnp.searchsorted(mu_nodes, mu, side="right"), 1, len(mu_nodes) - 1)
    q0, q1 = upper - 1, upper
    wq1 = jnp.clip((mu - mu_nodes[q0]) / (mu_nodes[q1] - mu_nodes[q0]), 0, 1)
    wq0 = 1 - wq1
    longitude = phi * n_phi / (2 * jnp.pi)
    p0 = jnp.floor(longitude).astype(jnp.int32)
    p1 = jnp.mod(p0 + 1, n_phi)
    wp1 = longitude - jnp.floor(longitude)
    wp0 = 1 - wp1
    flat = jnp.stack(
        (q0 * n_phi + p0, q0 * n_phi + p1, q1 * n_phi + p0, q1 * n_phi + p1),
        axis=1,
    ).reshape(-1)
    weight = jnp.stack((wq0 * wp0, wq0 * wp1, wq1 * wp0, wq1 * wp1), axis=1).reshape(
        -1
    ) * jnp.repeat(mass, 4)
    return jnp.bincount(flat, weights=weight, length=len(mu_nodes) * n_phi).reshape(
        1, len(mu_nodes), n_phi
    )
