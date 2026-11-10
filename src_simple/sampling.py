"""Use JAX's public CPU random API to retain the production Threefry stream.

The same integer seed must generate the same pairs and witnesses. NumPy's
PCG64 generator has the right distribution but changes those hypotheses.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .constants import PAIR_COUNT, WITNESS_COUNT, WITNESS_RNG_TAG


@partial(jax.jit, static_argnums=(0,))
def _sample_indices(point_count, seed):
    """Draw ordered-distinct endpoints, then two witnesses excluding both."""
    key = jax.random.key(seed)
    _, first_key, second_key = jax.random.split(key, 3)
    first = jax.random.randint(
        first_key, (PAIR_COUNT,), 0, point_count, dtype=jnp.int32
    )
    second = jax.random.randint(
        second_key, (PAIR_COUNT,), 0, point_count - 1, dtype=jnp.int32
    )
    second += second >= first
    witness_key = jax.random.fold_in(key, WITNESS_RNG_TAG)
    witnesses = jax.random.randint(
        witness_key, (PAIR_COUNT, WITNESS_COUNT), 0, point_count - 2, dtype=jnp.int32
    )
    witnesses += witnesses >= jnp.minimum(first, second)[:, None]
    witnesses += witnesses >= jnp.maximum(first, second)[:, None]
    return jnp.column_stack([first, second, witnesses])


def sample_indices(point_count, seed):
    """Return host integers [250000,4], identical to production for this seed."""
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        return np.asarray(_sample_indices(point_count, np.int32(seed)))
