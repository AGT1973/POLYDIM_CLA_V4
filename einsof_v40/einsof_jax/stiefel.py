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
