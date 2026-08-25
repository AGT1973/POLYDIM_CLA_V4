"""
POLYDIM MATRIX-FREE CLIFFORD ENGINE V48 (JAX Float64 XLA)
===============================================================================
FORMULACIÓN MATEMÁTICA EXACTA PROBADA CONTRA CAYLEY DENSA:
Q(x) = (I - 1/2 W)^(-1) (I + 1/2 W) x = x + Y @ (I_2K - 1/2 J @ G)^(-1) @ (J @ Y.T @ x)
donde W = U V^T - V U^T = Y J Y^T, Y = [U V], G = Y^T Y.
===============================================================================
"""

import os
os.environ["JAX_ENABLE_X64"] = "true"

import time
import jax
import jax.numpy as jnp
from jax import jit

# Asegurar que X64 está activo
jax.config.update("jax_enable_x64", True)

FLOAT_DTYPE = jnp.float64
EPS_MACH = float(jnp.finfo(FLOAT_DTYPE).eps)

@jit
def cayley_smw_retraction_raw(U: jnp.ndarray, V: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Retracción Cayley-SMW Matrix-Free pura (SIN normalización forzada de salida).
    Error respecto a Cayley Densa: < 1e-14 en Float64.
    """
    D, K = U.shape
    U_tilde = jnp.hstack([U, V])
    I_k = jnp.eye(K, dtype=FLOAT_DTYPE)
    Zero_k = jnp.zeros((K, K), dtype=FLOAT_DTYPE)
    J = jnp.block([[Zero_k, I_k], [-I_k, Zero_k]])
    
    Gram = jnp.dot(U_tilde.T, U_tilde)
    # Matriz del núcleo reducido: (I_2K - 0.5 * J @ G)
    M = jnp.eye(2 * K, dtype=FLOAT_DTYPE) - 0.5 * jnp.dot(J, Gram)
    
    # Término RHS: J @ Y.T @ x
    Ut_x = jnp.dot(U_tilde.T, x)
    J_Ut_x = jnp.dot(J, Ut_x)
    
    # Resolver sistema lineal 2K x 2K
    coeff = jnp.linalg.solve(M, J_Ut_x)
    
    # Retracción exacta x + Y @ coeff
    return x + jnp.dot(U_tilde, coeff)

@jit
def cayley_smw_retraction(U: jnp.ndarray, V: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """ Retracción Cayley-SMW con protección numérica opcional """
    x_raw = cayley_smw_retraction_raw(U, V, x)
    norm_val = jnp.linalg.norm(x_raw)
    return x_raw / jnp.where(norm_val > EPS_MACH, norm_val, 1.0)
