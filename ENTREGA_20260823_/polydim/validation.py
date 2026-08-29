"""
POLYDIM V57 VALIDATION & ISOMETRY AUDITING
Módulo de certificación para verificar la preservación isométrica rigurosa:
- Norma: ||f(x)|| = ||x||
- Producto Interno: <f(x), f(y)> = <x, y> (Ángulo y escala conservados)
"""

import jax
import jax.numpy as jnp

def assert_isometry(fn, x: jnp.ndarray, *args, atol: float = 1e-4, num_samples: int = 5) -> bool:
    """
    Verifica empíricamente que la transformación fn preserve tanto la norma ||f(x)|| = ||x||
    como el producto interno <f(x), f(y)> = <x, y> a lo largo de num_samples muestras independientes.
    """
    all_passed = True
    for i in range(num_samples):
        key = jax.random.PRNGKey(42 + i)
        y = x + jax.random.normal(key, x.shape, dtype=x.dtype) * 0.1
        y = y / jnp.linalg.norm(y)

        fx = fn(x, *args)
        fy = fn(y, *args)

        norm_x_before = jnp.linalg.norm(x)
        norm_fx_after = jnp.linalg.norm(fx)
        norm_preserved = jnp.abs(norm_x_before - norm_fx_after) < atol

        dot_before = jnp.einsum('i,i->', x, y)
        dot_after = jnp.einsum('i,i->', fx, fy)
        dot_preserved = jnp.abs(dot_before - dot_after) < atol

        if not bool(norm_preserved and dot_preserved):
            all_passed = False
            break

    return all_passed
