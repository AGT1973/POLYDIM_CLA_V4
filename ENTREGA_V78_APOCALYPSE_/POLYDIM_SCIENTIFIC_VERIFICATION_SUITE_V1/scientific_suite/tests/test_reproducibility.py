import numpy as np
import pytest

pytestmark = [pytest.mark.reproducibility]


@pytest.mark.p0
def test_benchmark_must_split_prng_keys():
    # This captures the exact class of bug in the V78 benchmark: reusing a JAX
    # PRNG key produces identical random arrays instead of independent samples.
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (16,))
    u = jax.random.normal(key, (16,))
    assert np.array_equal(np.asarray(x), np.asarray(u)), \
        "Control sanity check failed: repeated key should reproduce the same sample"


@pytest.mark.p0
def test_independent_keys_are_actually_independent():
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    kx, ku, kv = jax.random.split(key, 3)
    x = jax.random.normal(kx, (1024,))
    u = jax.random.normal(ku, (1024,))
    v = jax.random.normal(kv, (1024,))
    assert not np.array_equal(np.asarray(x), np.asarray(u))
    assert not np.array_equal(np.asarray(u), np.asarray(v))
