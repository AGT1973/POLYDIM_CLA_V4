from __future__ import annotations
import numpy as np


def controlled_matrix(m, n, cond, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    U0 = rng.standard_normal((m, n)).astype(dtype)
    V0 = rng.standard_normal((m, n)).astype(dtype)
    U, _ = np.linalg.qr(U0, mode="reduced")
    V, _ = np.linalg.qr(V0, mode="reduced")
    exponents = np.linspace(0.0, -np.log10(cond), n)
    s = 10.0 ** exponents
    return U @ np.diag(s) @ V.T


def orth_err(q):
    k = q.shape[-1]
    return float(np.linalg.norm(q.T @ q - np.eye(k), ord=2))


def qr_residual(a, q):
    r = q.T @ a
    denom = np.linalg.norm(a, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(a - q @ r, ord="fro"))
    return float(np.linalg.norm(a - q @ r, ord="fro") / denom)
