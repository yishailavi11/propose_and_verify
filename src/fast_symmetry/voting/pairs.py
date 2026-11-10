"""JAX Threefry streams for ordered-distinct pairs and independent witnesses."""

import jax
import jax.numpy as jnp

PAIR_COUNT = 250000


@jax.jit
def _make_pairs(points, seed):
    """Sample 250,000 ordered-distinct endpoint indices and their normalized pair planes.

    Returns endpoint arrays [P], pair normals [P,3], signed offsets [P], and
    the frozen sign-canonical vote vectors [P,3]. Keep their positive scale
    before angular normalization to preserve production float32 rounding.
    """
    key = jax.random.key(seed)
    _, key_first, key_compressed_second = jax.random.split(key, 3)
    point_count = len(points)
    point_first = jax.random.randint(
        key_first, (PAIR_COUNT,), 0, point_count, dtype=jnp.int32
    )
    compressed_second = jax.random.randint(
        key_compressed_second,
        (PAIR_COUNT,),
        0,
        point_count - 1,
        dtype=jnp.int32,
    )
    point_second = compressed_second + (compressed_second >= point_first).astype(
        jnp.int32
    )
    first = points[point_first]
    second = points[point_second]
    difference = second - first
    length = jnp.sqrt(jnp.sum(difference * difference, axis=1) + 1e-16)
    normal = difference / length[:, None]
    offset = jnp.sum(0.5 * (first + second) * normal, axis=1)
    sign = jnp.where(offset < 0, -1.0, 1.0)
    votes = normal * sign[:, None] * (jnp.abs(offset) + jnp.float32(0.5))[:, None]
    return point_first, point_second, normal, offset, votes


@jax.jit
def _sample_witnesses(point_count, point_first, point_second, seed):
    key = jax.random.fold_in(jax.random.key(seed), 0x57544E53)
    raw = jax.random.randint(key, (PAIR_COUNT, 2), 0, point_count - 2, dtype=jnp.int32)
    return _exclude_two(raw, point_first, point_second)


def _exclude_two(raw, first, second):
    """Map integers in ``[0,N-2)`` around two excluded endpoint indices."""
    lo = jnp.minimum(first, second)[:, None]
    hi = jnp.maximum(first, second)[:, None]
    value = raw + (raw >= lo).astype(jnp.int32)
    return value + (value >= hi).astype(jnp.int32)
