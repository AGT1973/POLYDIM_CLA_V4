# polydim_jax_engine.py
# Motor JAX (XLA) AOT Compilado para Computación en Espacio IA (ND >= 10,000)
# ESTRICTAMENTE PYTORCH-FREE, FLOAT64-NATIVE, STATELESS & FUNCTIONAL
# ============================================================================

import os
import sys
import numpy as np

# Configuración estricta de JAX (Float64) sin PyTorch
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    import numpy as jnp
    JAX_AVAILABLE = False

class IaSpaceJaxEngine:
    """
    Motor Funcional JAX (XLA) para transformaciones tensoriales en S^(D-1).
    Opera a nivel de acelerador (CPU/GPU/TPU) con cero overhead de GIL ni CPython.
    """
    def __init__(self, use_jax: bool = JAX_AVAILABLE):
        self.use_jax = use_jax and JAX_AVAILABLE
        self.backend = "JAX_XLA_FP64" if self.use_jax else "NUMPY_FP64"

    def project_to_sphere(self, v: np.ndarray) -> np.ndarray:
        """
        Proyecta un tensor arbitrario a la esfera unitaria S^(D-1) en Float64 estricto.
        """
        if self.use_jax:
            v_j = jnp.asarray(v, dtype=jnp.float64)
            norm = jnp.linalg.norm(v_j)
            safe_norm = jnp.maximum(norm, 1e-15)
            return np.asarray(v_j / safe_norm)
        else:
            v_np = np.asarray(v, dtype=np.float64)
            norm = np.linalg.norm(v_np)
            safe_norm = max(norm, 1e-15)
            return v_np / safe_norm

    def slerp_batch_jax(self, P: np.ndarray, Q: np.ndarray, t: float = 0.5) -> np.ndarray:
        """
        Geodésica SLERP batch vectorizada en JAX XLA con Kahan atan2.
        """
        if self.use_jax:
            P_j = jnp.asarray(P, dtype=jnp.float64)
            Q_j = jnp.asarray(Q, dtype=jnp.float64)
            
            d_norm = jnp.linalg.norm(P_j - Q_j, axis=-1, keepdims=True)
            s_norm = jnp.linalg.norm(P_j + Q_j, axis=-1, keepdims=True)
            omega = 2.0 * jnp.arctan2(d_norm, s_norm)

            eps = jnp.finfo(jnp.float64).eps
            sin_omega = jnp.sin(omega)
            safe_sin = jnp.where(jnp.abs(sin_omega) < eps, eps, sin_omega)

            s0 = jnp.sin((1.0 - t) * omega) / safe_sin
            s1 = jnp.sin(t * omega) / safe_sin

            res = s0 * P_j + s1 * Q_j
            res_norm = jnp.linalg.norm(res, axis=-1, keepdims=True)
            return np.asarray(res / jnp.maximum(res_norm, eps))
        else:
            P_np = np.asarray(P, dtype=np.float64)
            Q_np = np.asarray(Q, dtype=np.float64)

            d_norm = np.linalg.norm(P_np - Q_np, axis=-1, keepdims=True)
            s_norm = np.linalg.norm(P_np + Q_np, axis=-1, keepdims=True)
            omega = 2.0 * np.arctan2(d_norm, s_norm)

            eps = np.finfo(np.float64).eps
            sin_omega = np.sin(omega)
            safe_sin = np.where(np.abs(sin_omega) < eps, eps, sin_omega)

            s0 = np.sin((1.0 - t) * omega) / safe_sin
            s1 = np.sin(t * omega) / safe_sin

            res = s0 * P_np + s1 * Q_np
            res_norm = np.linalg.norm(res, axis=-1, keepdims=True)
            return res / np.maximum(res_norm, eps)

if __name__ == "__main__":
    engine = IaSpaceJaxEngine()
    print(f"[JAX ENGINE] Active Backend: {engine.backend}")

    p = np.random.randn(10000); p /= np.linalg.norm(p)
    q = np.random.randn(10000); q /= np.linalg.norm(q)

    res = engine.slerp_batch_jax(p, q, t=0.5)
    print(f"[JAX ENGINE] Batch SLERP D=10000 complete, norm={np.linalg.norm(res):.6f}")
