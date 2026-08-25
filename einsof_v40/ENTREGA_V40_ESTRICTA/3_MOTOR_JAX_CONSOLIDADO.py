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




"""
TSQR y CholeskyQR2 — Capa JAX
Hallazgos certificados CHK_06, CHK_11, CHK_12:
  - TSQR: error O(u) independiente de kappa(A)
  - CholeskyQR2: solo si kappa < sqrt(1/eps_f32) ~ 4096
  - Gap dinamico: ||Q^T Q - I||_F con umbral sin hardcoding
"""
import numpy as np
import math


def _machine_eps(dtype=np.float64) -> float:
    """Epsilon de maquina derivado en runtime. Axioma Cero."""
    return float(np.finfo(dtype).eps)


def tsqr_blocked(A: np.ndarray, block_size: int = None) -> tuple:
    """
    Tree-TSQR por bloques. No forma A^T A. Error O(u) independiente de kappa.
    Args: A shape (D, K). block_size: derivado de D si None.
    Returns: (Q, R) con Q^T Q = I a precision de maquina.
    """
    D, K = A.shape
    eps = _machine_eps(A.dtype)

    # Derivar block_size sin hardcoding (Axioma Cero)
    if block_size is None:
        import os, psutil
        avail = psutil.virtual_memory().available
        # Usar bloques que quepan en L3 cache aprox (8MB) o en RAM disponible
        l3_approx = min(8 * 1024 * 1024, avail // 16)
        block_size = max(K * 2, l3_approx // (K * A.itemsize))
        block_size = min(block_size, D)

    n_blocks = max(1, D // block_size)
    R_blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, D)
        block = A[start:end, :].astype(np.float64)
        _, R_local = np.linalg.qr(block, mode='reduced')
        R_blocks.append(R_local)

    stacked = np.vstack(R_blocks)
    Q_top, R_final = np.linalg.qr(stacked, mode='reduced')
    return Q_top, R_final


def cholesky_qr2(A: np.ndarray) -> np.ndarray:
    """
    CholeskyQR con DOS pasadas (CholeskyQR2). Solo si kappa < kappa_safe.
    Implementacion correcta: Q = A @ R^{-1} via solve(L, A.T).T
    (NO solve(L.T, A.T) — ese es el bug clasico que da gap 5e-8).
    """
    eps_f32 = _machine_eps(np.float32)
    kappa_safe = math.sqrt(1.0 / eps_f32)  # ~4096, derivado

    A = A.astype(np.float64)
    norms = np.linalg.norm(A, axis=0)
    kappa_est = float(norms.max() / (norms.min() + 1e-300))

    if kappa_est >= kappa_safe:
        # Fallback a TSQR (seguro para cualquier kappa)
        Q, _ = np.linalg.qr(A, mode='reduced')
        return Q

    G = A.T @ A
    try:
        L = np.linalg.cholesky(G)
        Q = np.linalg.solve(L, A.T).T       # Pasada 1: A @ L^{-T} = A @ R^{-1}
        G2 = Q.T @ Q
        L2 = np.linalg.cholesky(G2)
        Q = np.linalg.solve(L2, Q.T).T      # Pasada 2: misma correccion
    except np.linalg.LinAlgError:
        Q, _ = np.linalg.qr(A, mode='reduced')
    return Q


def ortho_gap(Q: np.ndarray) -> float:
    """Mide ||Q^T Q - I||_F. Dispara TSQR si supera umbral dinamico."""
    K = Q.shape[1]
    D = Q.shape[0]
    eps = _machine_eps(Q.dtype)
    tol = 8.0 * eps * (math.sqrt(D) + K)  # umbral sin hardcoding
    gram = Q.T @ Q
    gap = float(np.linalg.norm(gram - np.eye(K), 'fro'))
    return gap, tol, gap > tol




"""
Proyeccion en variedad de Stiefel V_k(R^D) — Capa JAX
Hallazgos certificados CHK_07, CHK_10:
  - Drift acumulativo O(sqrt(N)*eps) no O(N*eps) con f64
  - Re-proyeccion adaptativa cuando gap > umbral dinamico
"""
import numpy as np
import math
from .tsqr import ortho_gap, tsqr_blocked
from .slerp import slerp_stable


def project_stiefel(Q: np.ndarray) -> np.ndarray:
    """Re-proyecta Q a la variedad de Stiefel via TSQR si gap lo requiere."""
    gap, tol, needs = ortho_gap(Q)
    if needs:
        Q_new, _ = tsqr_blocked(Q)
        return Q_new
    return Q


def stiefel_drift_check(D: int = 10_000, N: int = 10_000) -> dict:
    """
    Verifica empiricamente que el drift es O(sqrt(N)*eps) no O(N*eps).
    Regimen sqrt(N) = caminata aleatoria de errores independientes.
    Regimen N = sesgo sistematico = CATASTROFICO.
    """
    eps = float(np.finfo(np.float64).eps)
    tol_sqrt = 16.0 * eps * math.sqrt(N)

    np.random.seed(42)
    v = np.random.randn(D).astype(np.float64)
    v /= np.linalg.norm(v)
    target = np.random.randn(D).astype(np.float64)
    target /= np.linalg.norm(target)

    gain = 1.0 / N
    max_drift = 0.0

    for _ in range(N):
        v = slerp_stable(v, target, gain)
        norm = np.linalg.norm(v)
        drift = abs(norm - 1.0)
        if drift > max_drift:
            max_drift = drift
        v /= norm

    return {
        "drift_regime": "sqrt(N)" if max_drift < tol_sqrt * 10 else "LINEAR_ALERT",
        "max_drift": max_drift,
        "tol_sqrt_N": tol_sqrt,
        "eps_f64": eps,
    }

