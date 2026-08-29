"""Independent NumPy reference for S^(D-1). No JAX/POLYDIM dependency."""
from __future__ import annotations
import math
import numpy as np


def norm2(x):
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x))
    if scale == 0.0:
        return 0.0
    y = x / scale
    return float(scale * np.sqrt(np.dot(y, y)))


def normalize(x, tol=1e-14):
    n = norm2(x)
    if not np.isfinite(n) or n <= tol:
        raise ValueError("zero/non-finite vector is not a sphere point")
    return np.asarray(x, dtype=np.float64) / n


def sphere_point(x, tol=1e-12):
    x = normalize(x)
    if abs(norm2(x) - 1.0) > tol:
        raise ValueError("not on sphere")
    return x


def tangent_project(x, v):
    x = sphere_point(x)
    v = np.asarray(v, dtype=np.float64)
    return v - np.dot(x, v) * x


def exp_map(x, v, tol=1e-14):
    x = sphere_point(x)
    v = tangent_project(x, v)
    r = norm2(v)
    if r <= tol:
        return x.copy()
    return math.cos(r) * x + (math.sin(r) / r) * v


def log_map(x, y, antipodal_tol=1e-12):
    x = sphere_point(x)
    y = sphere_point(y)
    # Stable scalar angle. Inputs are unit vectors.
    theta = 2.0 * math.atan2(norm2(x - y), norm2(x + y))
    p = y - np.dot(x, y) * x
    pn = norm2(p)
    if pn <= antipodal_tol:
        if np.dot(x, y) > 0:
            return np.zeros_like(x)
        raise ValueError("log map is multi-valued at the antipode")
    return (theta / pn) * p


def householder(x, v, zero_tol=0.0):
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    vv = float(np.dot(v, v))
    if vv <= zero_tol:
        raise ValueError("zero reflector vector")
    return x - (2.0 * float(np.dot(x, v)) / vv) * v


def stiefel_full_cayley(X, G, alpha):
    """Cayley action using the explicit D x D skew generator.

    Uses the common Wen-Yin sign convention:
        W = G X^T - X G^T
        Y = (I + a/2 W)^(-1) (I - a/2 W) X
    """
    X = np.asarray(X, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    d = X.shape[0]
    W = G @ X.T - X @ G.T
    I = np.eye(d)
    A = I + 0.5 * alpha * W
    B = (I - 0.5 * alpha * W) @ X
    return np.linalg.solve(A, B)


def stiefel_smw_wenyin(X, G, alpha):
    """Independent low-rank Cayley formula for X^T X = I, k arbitrary.

    W = U V^T with
      U = [G, X], V = [X, -G]
    and C = I + (a/2) V^T U.
    The Woodbury identity is applied to A = I + (a/2) U V^T.
    """
    X = np.asarray(X, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    d, k = X.shape
    I_d = np.eye(d)
    I_2k = np.eye(2 * k)
    U = np.concatenate([G, X], axis=1)
    V = np.concatenate([X, -G], axis=1)
    a2 = 0.5 * alpha
    # A = I + a2 U V^T
    # A^{-1} = I - a2 U (I + a2 V^T U)^{-1} V^T
    C = I_2k + a2 * (V.T @ U)
    rhs = (I_d - a2 * U @ np.linalg.solve(C, V.T))
    # right multiplication by (I - a2 W) on X; then A^{-1}
    return rhs @ ((I_d - a2 * (U @ V.T)) @ X)
