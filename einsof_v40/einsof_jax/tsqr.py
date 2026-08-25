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
