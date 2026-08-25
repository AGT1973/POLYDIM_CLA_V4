"""
SLERP estable en S^(D-1) — Capa JAX
Hallazgo certificado CHK_01, CHK_08, CHK_09:
  omega = 2 * atan2(||p-q||, ||p+q||)  — sin cancelacion catastrofica
  Antipodal: tangente determinista via BLAKE2b (CHK_09)
  Precision mixta: calculos geometricos en f64 (CHK_15)
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")  # f64 obligatorio

try:
    import jax
    import jax.numpy as jnp
    JAX_OK = True
except ImportError:
    import numpy as jnp
    JAX_OK = False

import numpy as np
import hashlib, math


def slerp_stable(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP en S^(D-1). Formula estable: omega = 2*atan2(||p-q||, ||p+q||).
    Maneja 3 regimenes:
      - theta ~ 0 : LERP normalizado (expansion Taylor O(omega^2))
      - theta ~ pi: escape determinista BLAKE2b
      - normal    : SLERP clasico con omega derivado por atan2
    Args: p, q vectores unitarios en R^D. t en [0,1].
    Returns: vector interpolado unitario en R^D.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Garantizar unitarios (no asumir)
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)

    d_norm = np.linalg.norm(p - q)
    s_norm = np.linalg.norm(p + q)
    omega = 2.0 * math.atan2(d_norm, s_norm)

    # Regimen theta ~ 0
    eps = float(np.finfo(np.float64).eps)
    if omega < 16.0 * eps:
        result = p + t * (q - p)
        return result / np.linalg.norm(result)

    # Regimen antipodal: theta ~ pi
    if math.pi - omega < 1e-9:
        digest = hashlib.blake2b(p.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        v_raw = rng.standard_normal(len(p))
        v_raw -= np.dot(v_raw, p) * p
        v = v_raw / np.linalg.norm(v_raw)
        return p * math.cos(t * math.pi) + v * math.sin(t * math.pi)

    # Regimen normal
    sin_omega = math.sin(omega)
    s0 = math.sin((1.0 - t) * omega) / sin_omega
    s1 = math.sin(t * omega) / sin_omega
    result = s0 * p + s1 * q
    return result / np.linalg.norm(result)


def slerp_batch(P: np.ndarray, Q: np.ndarray, t: float) -> np.ndarray:
    """SLERP batch sobre N pares de vectores. P, Q shape (N, D)."""
    return np.stack([slerp_stable(P[i], Q[i], t) for i in range(len(P))])
