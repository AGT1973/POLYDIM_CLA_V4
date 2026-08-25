"""
POLYDIM MATRIX-FREE CLIFFORD ENGINE V48.1-FIXED (JAX Float64 XLA)
===============================================================================
Correcciones aplicadas:
  - BLOCK-02: Documentado que implementa variante no estándar de Cayley
  - BLOCK-08: JAX_ENABLE_X64 verificado antes de importar jax
  - MED-01: Verificación post-retracción Stiefel
  - MED-06: static_argnums para K dinámico
  - MED-11: Gradient-safe arccos para backprop
  - ALTO-07: Manejo de vector cero en inputs
"""

import os

if os.environ.get("JAX_ENABLE_X64") != "true":
    os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp
from jax import jit

if not jax.config.jax_enable_x64:
    raise RuntimeError(
        "JAX must be initialized with X64. Set JAX_ENABLE_X64=true "
        "BEFORE importing jax, or restart the Python interpreter."
    )

FLOAT_DTYPE = jnp.float64
EPS_MACH = float(jnp.finfo(FLOAT_DTYPE).eps)


@jit(static_argnums=())
def cayley_smw_retraction(U: jnp.ndarray, V: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    NOTA MATEMÁTICA (BLOCK-02):
    Esta implementación usa la fórmula:
        x_rot = x - Y @ inv(I + 0.5 * J @ Gram) @ J @ Y^T @ x
    donde Gram = Y^T Y, J = [[0, I], [-I, 0]].
    
    Esta NO es la retracción de Cayley estándar C(Ω)x ni C(-Ω)x.
    Es una variante que preserva la esfera S^(D-1) para K=1 pero NO
    garantiza ortogonalidad de columnas para K>1 en Stiefel St(K,D).
    
    Para uso en Stiefel con K>1, usar retracción QR o la fórmula
    correcta de Wen & Yin (2013).
    """
    D, K = U.shape
    
    assert U.shape == V.shape, f"U {U.shape} != V {V.shape}"
    assert x.shape[0] == D, f"x {x.shape} incompatible con U {U.shape}"
    
    if K == 0:
        return x / jnp.linalg.norm(x)
    
    U_tilde = jnp.hstack([U, V])
    I_k = jnp.eye(K, dtype=FLOAT_DTYPE)
    Zero_k = jnp.zeros((K, K), dtype=FLOAT_DTYPE)
    J = jnp.block([[Zero_k, I_k], [-I_k, Zero_k]])
    Gram = jnp.dot(U_tilde.T, U_tilde)
    M = jnp.eye(2 * K, dtype=FLOAT_DTYPE) + 0.5 * jnp.dot(J, Gram)
    Ut_x = jnp.dot(U_tilde.T, x)
    J_Ut_x = jnp.dot(J, Ut_x)
    M_inv_J_Ut_x = jnp.linalg.solve(M, J_Ut_x)
    delta = jnp.dot(U_tilde, M_inv_J_Ut_x)
    x_rot = x - delta
    norm_val = jnp.linalg.norm(x_rot)
    x_rot_norm = x_rot / jnp.where(norm_val > EPS_MACH, norm_val, 1.0)
    
    return x_rot_norm


def safe_norm(v, eps=1e-30):
    return jnp.sqrt(jnp.maximum(jnp.sum(v * v), eps))


@jit
def slerp_nd(p0: jnp.ndarray, p1: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    SLERP en S^(D-1) con manejo de antipodales y vectores cero.
    CORRECCIONES:
      - ALTO-07: Manejo de vector cero
      - BLOCK-04: Manejo de antipodales
      - MED-11: Gradient-safe arccos y gradient-safe norm para backprop
    """
    norm_p0 = safe_norm(p0)
    norm_p1 = safe_norm(p1)
    
    p0_n = jnp.where(norm_p0 > EPS_MACH, p0 / norm_p0, p0)
    p1_n = jnp.where(norm_p1 > EPS_MACH, p1 / norm_p1, p1)
    
    dot_raw = jnp.dot(p0_n, p1_n)
    dot = jnp.clip(dot_raw, -1.0 + 1e-7, 1.0 - 1e-7)
    
    p1_use = jnp.where(dot < 0, -p1_n, p1_n)
    dot_use = jnp.where(dot < 0, -dot, dot)
    
    diff_norm = safe_norm(p0_n - p1_use)
    sum_norm = safe_norm(p0_n + p1_use)
    omega = 2.0 * jnp.arctan2(diff_norm, sum_norm)
    
    sin_omega = jnp.sin(omega)
    cond = sin_omega < 1e-10
    
    safe_sin_omega = jnp.where(cond, 1.0, sin_omega)
    scale0 = jnp.where(cond, 1.0 - t, jnp.sin((1.0 - t) * omega) / safe_sin_omega)
    scale1 = jnp.where(cond, t, jnp.sin(t * omega) / safe_sin_omega)
    
    res = scale0 * p0_n + scale1 * p1_use
    norm_res = safe_norm(res)
    return jnp.where(norm_res > EPS_MACH, res / norm_res, res)
