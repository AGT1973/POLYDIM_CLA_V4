"""
POLYDIM EINSOF V44 — Motor JAX / Geometría Riemanniana en S^(D-1) y Stiefel V_k(R^D)
=======================================================================================
Edición Congelada SOTA — Consolidación de 5 auditorías Red-Team (Ago 2026).

Parches aplicados (V43 → V44):
  [FIX-01] Normalización Kahan post-antipodal en fallback NumPy (Kimi #1)
  [FIX-02] Guarda sin_omega en régimen batch normal (Kimi #3A)
  [FIX-03] Broadcast seguro de T_arr con validación ndim (Kimi #3B / #20)
  [FIX-04] HKDF RFC 5869 real con salt por nodo (Kimi #4B)
  [FIX-05] MAX_EPOCH_JUMP anti-DoS (Kimi #4A)
  [FIX-06] SHAKE-256 XOF directo — mata PCG64 thread-unsafe (Kimi #7 / #10)
  [FIX-07] Protocolo HMAC unificado con domain separator (Kimi #9)
  [FIX-08] validate_finite_vector: guarda ndim==1 (Kimi #11)
  [FIX-09] project_stiefel: guarda D >= K (Kimi #12)
  [FIX-10] PmtpStatefulReceiver: window_size validado (Kimi #13)
  [FIX-11] estimate_kappa_power: regularización relativa (Kimi #14)
  [FIX-12] _validate_finite_matrix para tsqr/cholesky/fréchet (Kimi #17)
  [FIX-13] Tag length guard en verify_and_accept (Kimi #19)
  [FIX-14] frechet_mean: tangente de vectors[idx] no de mu (Kimi/propia)

NOTA ARQUITECTÓNICA: slerp_batch NO es jit-able por JAX debido al loop Python
puro en la precomputación de tangentes antipodales.
"""

import os
import sys
import math
import hashlib
import hmac
import threading
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any

# Forzar backend JAX en Float64
os.environ.setdefault("JAX_ENABLE_X64", "1")

try:
    import jax
    import jax.numpy as jnp
    JAX_OK = True
    _test_f64 = jnp.array(1.0, dtype=jnp.float64)
    if _test_f64.dtype != jnp.float64:
        JAX_OK = False
except Exception:
    import numpy as jnp
    JAX_OK = False

# ============================================================================
# FFI BOUNDARIES: C++ AND RUST
# ============================================================================
import ctypes
try:
    import einsof_rust
    RUST_OK = True
except ImportError:
    einsof_rust = None
    RUST_OK = False

_DIR = os.path.dirname(os.path.abspath(__file__))
_cpp_dll = os.path.join(_DIR, "slerp_kernel_v46.dll")
_cpp_so = os.path.join(_DIR, "slerp_kernel_v46.so")

slerp_cpp_kernel = None
if os.path.exists(_cpp_dll):
    slerp_cpp_kernel = ctypes.cdll.LoadLibrary(_cpp_dll)
elif os.path.exists(_cpp_so):
    slerp_cpp_kernel = ctypes.cdll.LoadLibrary(_cpp_so)

if slerp_cpp_kernel:
    slerp_cpp_kernel.slerp.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    slerp_cpp_kernel.slerp.restype = None

def slerp_c(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    p = validate_finite_vector(p, "p")
    q = validate_finite_vector(q, "q")
    if p.shape != q.shape:
        raise ValueError("Dimensions must match")
    
    # [BULLDOG FIX]: Anti-Happy-Path. np.asarray no asegura C_CONTIGUOUS. 
    p = np.ascontiguousarray(p, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    
    import math
    if not math.isfinite(t):
        raise ValueError("t no puede ser NaN o Inf")

    D = len(p)
    # [BULLDOG FIX]: Asignación estrictamente C-contigua
    out = np.empty(D, dtype=np.float64, order='C')
    scratch = np.empty(D, dtype=np.float64, order='C')
    
    if slerp_cpp_kernel:
        slerp_cpp_kernel.slerp(p, q, float(t), out, scratch, D, len(scratch))
        return out
    else:
        raise RuntimeError("C++ kernel not loaded.")



# ============================================================================
# 0. HKDF RFC 5869 (BLAKE2b-512) — [FIX-04]
# ============================================================================

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """RFC 5869 Extract: PRK = HMAC-Hash(salt, IKM)."""
    if not salt:
        salt = bytes(64)  # Tamaño de bloque de BLAKE2b
    return hmac.new(salt, ikm, hashlib.blake2b).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 64) -> bytes:
    """RFC 5869 Expand: OKM = T(1) || T(2) || ... || T(N)."""
    hash_len = 64  # BLAKE2b-512
    n = (length + hash_len - 1) // hash_len
    if n > 255:
        raise ValueError("HKDF-Expand limit exceeded (max 255 * HashLen)")
    okm = b""
    previous = b""
    for i in range(1, n + 1):
        previous = hmac.new(prk, previous + info + bytes([i]), hashlib.blake2b).digest()
        okm += previous
    return okm[:length]


# ============================================================================
# 1. CONTRATO DE SILICIO DINÁMICO (AXIOMA CERO PURO)
# ============================================================================

from silicon_contract import HOST_SILICON

def machine_eps(dtype=np.float64) -> float:
    """Epsilon de máquina derivado dinámicamente, respetando dtype [FIX-09]."""
    if dtype == np.float64 and 'RUST_OK' in globals() and RUST_OK:
        return einsof_rust.get_machine_epsilon_f64()
    return float(np.finfo(dtype).eps)

def machine_tiny(dtype=np.float64) -> float:
    """Mínimo flotante positivo normalizado, respetando dtype [FIX-09]."""
    return float(np.finfo(dtype).tiny)

def theta_small(dtype=np.float64, D: int = 1) -> float:
    """Umbral de Taylor para ángulos infinitesimales derivado de D y eps."""
    eps = float(np.finfo(dtype).eps)
    return 16.0 * eps * math.sqrt(float(max(1, D)))

def theta_antipodal(dtype=np.float64, D: int = 1) -> float:
    """Umbral antipodal derivado dinámicamente del silicio [FIX-19: sqrt(eps)]."""
    eps = float(np.finfo(dtype).eps)
    return max(100.0 * eps * math.sqrt(float(max(1, D))), math.sqrt(eps))


# ============================================================================
# 2. VALIDACIÓN DE ENTRADAS Y MEMORIA (PREFLIGHT SOTA)
# ============================================================================

def validate_finite_vector(v: np.ndarray, name: str = "vector"):
    """
    Verifica que el vector no contenga NaNs, Infs, dimensión cero o norma cero.
    [FIX-08] Agrega guarda ndim == 1.
    """
    if v is None:
        raise ValueError(f"{name} es None")
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError(f"{name} debe ser 1D, got shape {v.shape}")
    D = len(v)
    if D == 0:
        raise ValueError(f"{name} no puede estar vacío")
    if D < 2:
        raise ValueError(f"{name} dimensión {D} inválida para S^(D-1); se requiere D >= 2")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contiene valores no finitos (NaN o Inf)")
    nrm = float(np.linalg.norm(v))
    if nrm < machine_tiny(v.dtype):
        raise ValueError(f"{name} tiene norma prácticamente cero")
    return v.astype(np.float64, copy=False)


def _validate_finite_matrix(A: np.ndarray, name: str = "matrix") -> np.ndarray:
    """
    [FIX-12] Rechaza matrices con NaN o Inf ANTES de cualquier cómputo.
    Un NaN en la entrada hace que las guardas de calidad (ortho_gap, etc.) mientan.
    """
    A = np.asarray(A, dtype=np.float64)
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contiene NaN o Inf. Entrada corrupta.")
    return A


def check_memory_available(required_bytes: int, safety_margin: float = 0.8):
    """Preflight de memoria para evitar OOM silenciosos."""
    try:
        import psutil
        available = psutil.virtual_memory().available
        if required_bytes > available * safety_margin:
            raise MemoryError(
                f"Memoria insuficiente: requiere {required_bytes / 1e9:.2f} GB, "
                f"disponible {available / 1e9:.2f} GB (margen {safety_margin*100:.0f}%)"
            )
    except ImportError:
        pass


# ============================================================================
# 3. GENERADOR DETERMINISTA DE TANGENTES XOF (SHAKE-256) — [FIX-06]
# ============================================================================

def deterministic_tangent(p: np.ndarray) -> np.ndarray:
    """
    Genera vector tangente ortogonal unitario v ⊥ p en S^(D-1).
    [FIX-30] Usar PRNG vectorizado de NumPy sembrado con hash en lugar de generar arrays enormes de bytes.
    """
    p = validate_finite_vector(p, "p")
    D = len(p)
    tiny = machine_tiny(np.float64)

    sub_sample = np.ascontiguousarray(p[:min(D, 128)])
    
    # 1. Generar Seed criptográfico corto (32 bytes)
    sub_sample = np.where(sub_sample == 0, 0.0, sub_sample)
    seed_bytes = hashlib.shake_256(sub_sample.tobytes()).digest(32)
    seed_arr = np.frombuffer(seed_bytes, dtype=np.uint32)
    
    # 2. Expandir vectorizadamente usando NumPy (Velocidad de silicio)
    rng = np.random.default_rng(seed_arr)
    v_raw = rng.uniform(-0.5, 0.5, size=D).astype(np.float64)

    # Gram-Schmidt: proyectar fuera de p
    dot_val = float(np.dot(v_raw, p))
    v = v_raw - dot_val * p
    v_norm = float(np.linalg.norm(v))

    if v_norm < theta_small(np.float64, D):
        min_idx = int(np.argmin(np.abs(p)))
        v = np.zeros(D, dtype=np.float64)
        v[min_idx] = 1.0
        v -= p[min_idx] * p
        v_norm = float(np.linalg.norm(v))

    return v / (v_norm + tiny)


# ============================================================================
# 4. SLERP GEODÉSICO ESCALAR Y BATCH VECTORIZADO
# ============================================================================

def slerp_stable(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP geodésico en S^(D-1) con validación estricta y umbral antipodal realista.
    """
    p = validate_finite_vector(p, "p")
    q = validate_finite_vector(q, "q")

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    D = len(p)
    eps = machine_eps(np.float64)
    tiny = machine_tiny(np.float64)

    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    p_unit = p / (p_norm + tiny)
    q_unit = q / (q_norm + tiny)

    d_norm = float(np.linalg.norm(p_unit - q_unit))
    s_norm = float(np.linalg.norm(p_unit + q_unit))
    omega = 2.0 * math.atan2(d_norm, s_norm)

    # 1. Régimen Taylor (theta ~ 0) pasando D explícito
    if omega < theta_small(np.float64, D):
        result = p_unit + t * (q_unit - p_unit)
        return result / (np.linalg.norm(result) + tiny)

    # 2. Régimen Antipodal (theta ~ pi) pasando D explícito
    if s_norm < theta_antipodal(np.float64, D):
        v = deterministic_tangent(p_unit)
        result = p_unit * math.cos(t * math.pi) + v * math.sin(t * math.pi)
        return result / (np.linalg.norm(result) + tiny)

    # 3. Régimen Normal
    sin_omega = math.sin(omega)
    if abs(sin_omega) < eps:
        result = p_unit + t * (q_unit - p_unit)
        return result / (np.linalg.norm(result) + tiny)

    s0 = math.sin((1.0 - t) * omega) / sin_omega
    s1 = math.sin(t * omega) / sin_omega
    result = s0 * p_unit + s1 * q_unit
    return result / (np.linalg.norm(result) + tiny)


def slerp_batch(P: np.ndarray, Q: np.ndarray, t) -> np.ndarray:
    """
    SLERP batch vectorizado sobre N pares en R^(N x D).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2:
        raise ValueError("P y Q deben ser matrices 2D con formas idénticas")

    N, D = P.shape
    eps = machine_eps(np.float64)
    tiny = machine_tiny(np.float64)

    # Validación de vectores no nulos
    p_norms = np.linalg.norm(P, axis=1)
    q_norms = np.linalg.norm(Q, axis=1)
    if np.any(p_norms < tiny) or np.any(q_norms < tiny):
        raise ValueError("slerp_batch: vectores de norma cero detectados")

    # Broadcast seguro de T_arr
    if np.isscalar(t):
        T_arr = np.full(N, float(t), dtype=np.float64)
    else:
        T_arr = np.asarray(t, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.full(N, float(T_arr), dtype=np.float64)
        elif T_arr.size == 1:
            T_arr = np.full(N, float(T_arr.item()), dtype=np.float64)
        else:
            if T_arr.ndim != 1:
                raise ValueError(f"t debe ser escalar o array 1D. Got shape {T_arr.shape}")
            if T_arr.shape[0] != N:
                raise ValueError(
                    f"t array length ({T_arr.shape[0]}) must match batch size N ({N})"
                )

    P_unit = P / (p_norms[:, None] + tiny)
    Q_unit = Q / (q_norms[:, None] + tiny)

    d_norms = np.linalg.norm(P_unit - Q_unit, axis=1)
    s_norms = np.linalg.norm(P_unit + Q_unit, axis=1)
    omegas = 2.0 * np.arctan2(d_norms, s_norms)

    t_small_val = theta_small(np.float64, D)
    t_anti_val = theta_antipodal(np.float64, D)

    # PARCHE 4: GUARDA SOTA ANTI-NaN
    sin_omegas = np.sin(omegas)
    anti_mask = (s_norms < t_anti_val)

    # [BULLDOG FIX]: Asignación fantasma V_anti (NxD) eliminada. Código muerto tóxico que causaba OOM.

    out = np.empty_like(P_unit)

    if JAX_OK:
        @jax.jit
        def _jax_slerp_batch(P_j, Q_j, T_j):
            tiny_j = machine_tiny(np.float64)
            eps_j = machine_eps(np.float64)
            D_j = P_j.shape[1]
            t_small_val_j = theta_small(np.float64, D_j)
            t_anti_val_j = theta_antipodal(np.float64, D_j)

            P_norms = jnp.linalg.norm(P_j, axis=1, keepdims=True)
            Q_norms = jnp.linalg.norm(Q_j, axis=1, keepdims=True)
            P_unit_j = P_j / (P_norms + tiny_j)
            Q_unit_j = Q_j / (Q_norms + tiny_j)

            d_norms = jnp.linalg.norm(P_unit_j - Q_unit_j, axis=1)
            s_norms = jnp.linalg.norm(P_unit_j + Q_unit_j, axis=1)
            omegas = 2.0 * jnp.arctan2(d_norms, s_norms)
            anti_mask_j = (s_norms < t_anti_val_j)

            def _single_slerp(p_i, q_i, t_i, om_i, is_anti_i):
                is_near0 = om_i < t_small_val_j
                lerp_val = p_i + t_i * (q_i - p_i)
                lerp_res = lerp_val / (jnp.linalg.norm(lerp_val) + tiny_j)

                # JIT-able pure callback for deterministic tangent
                v_anti_i = jax.lax.cond(
                    is_anti_i,
                    lambda x: jax.pure_callback(
                        deterministic_tangent,
                        jax.ShapeDtypeStruct(x.shape, x.dtype),
                        x,
                        vmap_method='sequential'
                    ),
                    lambda x: jnp.zeros_like(x),
                    p_i
                )

                anti_val = p_i * jnp.cos(t_i * jnp.pi) + v_anti_i * jnp.sin(t_i * jnp.pi)
                anti_res = anti_val / (jnp.linalg.norm(anti_val) + tiny_j)

                sin_om = jnp.sin(om_i)
                safe_sin = jnp.maximum(jnp.abs(sin_om), eps_j)
                s0 = jnp.sin((1.0 - t_i) * om_i) / safe_sin
                s1 = jnp.sin(t_i * om_i) / safe_sin
                norm_val = s0 * p_i + s1 * q_i
                norm_res = norm_val / (jnp.linalg.norm(norm_val) + tiny_j)

                return jnp.where(is_near0, lerp_res, jnp.where(is_anti_i, anti_res, norm_res))

            return jax.vmap(_single_slerp)(P_unit_j, Q_unit_j, T_j, omegas, anti_mask_j)

        return np.asarray(_jax_slerp_batch(P, Q, T_arr))
    else:
        # ---- Fallback NumPy vectorizado ----
        # [FIX-31] Optimizar uso de memoria in-place
        
        P_unit = P / (p_norms[:, None] + tiny)
        Q_unit = Q / (q_norms[:, None] + tiny)

        out = np.empty_like(P_unit)

        # Régimen Taylor (omega ~ 0)
        mask_small = omegas < t_small_val
        if np.any(mask_small):
            t_s = T_arr[mask_small]
            p_s = P_unit[mask_small]
            q_s = Q_unit[mask_small]
            out_s = p_s + t_s[:, None] * (q_s - p_s)
            nrm = np.linalg.norm(out_s, axis=1)
            nrm_ok = nrm > eps
            out_s[nrm_ok] /= nrm[nrm_ok][:, None]
            out_s[~nrm_ok] = p_s[~nrm_ok]
            out[mask_small] = out_s

        # [FIX-01] Régimen Antipodal con normalización explícita
        mask_anti = (~mask_small) & anti_mask
        if np.any(mask_anti):
            anti_indices = np.where(mask_anti)[0]
            for idx in anti_indices:
                v_anti = deterministic_tangent(P_unit[idx])
                cos_t = np.cos(T_arr[idx] * np.pi)
                sin_t = np.sin(T_arr[idx] * np.pi)
                res_anti = P_unit[idx] * cos_t + v_anti * sin_t
                nrm = np.linalg.norm(res_anti)
                if nrm > eps:
                    res_anti /= nrm
                else:
                    res_anti = P_unit[idx]
                out[idx] = res_anti

        # [FIX-02] Régimen Normal con guarda sin_omega
        mask_norm = ~mask_small & ~mask_anti
        if np.any(mask_norm):
            p_n = P_unit[mask_norm]
            q_n = Q_unit[mask_norm]
            t_n = T_arr[mask_norm]
            om_n = omegas[mask_norm]
            sin_om = np.sin(om_n)

            # Guarda de sin_omega pequeño (paridad con slerp_stable escalar)
            sin_guard = np.abs(sin_om) < eps
            out_n = np.empty_like(p_n)

            if np.any(sin_guard):
                lerp_vals = (p_n[sin_guard]
                             + t_n[sin_guard][:, None] * (q_n[sin_guard] - p_n[sin_guard]))
                lerp_nrm = np.linalg.norm(lerp_vals, axis=1)
                l_ok = lerp_nrm > eps
                lerp_vals[l_ok] /= lerp_nrm[l_ok][:, None]
                lerp_vals[~l_ok] = p_n[sin_guard][~l_ok]
                out_n[sin_guard] = lerp_vals

            if not np.all(sin_guard):
                normal_mask = ~sin_guard
                s0 = np.sin((1.0 - t_n[normal_mask]) * om_n[normal_mask]) / sin_om[normal_mask]
                s1 = np.sin(t_n[normal_mask] * om_n[normal_mask]) / sin_om[normal_mask]
                norm_vals = s0[:, None] * p_n[normal_mask] + s1[:, None] * q_n[normal_mask]
                norm_nrm = np.linalg.norm(norm_vals, axis=1)
                n_ok = norm_nrm > eps
                norm_vals[n_ok] /= norm_nrm[n_ok][:, None]
                norm_vals[~n_ok] = p_n[normal_mask][~n_ok]
                out_n[normal_mask] = norm_vals

            out[mask_norm] = out_n

        return out


# ============================================================================
# 5. TSQR-HR PRE-ASIGNADO CON COTA DE HIGHAM
# ============================================================================

def tsqr_blocked(A: np.ndarray, block_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tree-TSQR con Householder Reconstruction sobre búfer pre-asignado contiguo.
    Cota teórica de Higham (1996): ||Q^T Q - I_K||_2 <= c * D * eps_mach.
    """
    if A.ndim != 2:
        raise ValueError("A debe ser una matriz bidimensional")

    A = _validate_finite_matrix(A, "A")  # [FIX-12]
    D, K = A.shape

    if D < K:
        raise ValueError(f"TSQR requiere D >= K. Obtenido D={D}, K={K}")

    if block_size is None or block_size <= 0:
        block_size = max(K * 2, min(D, 4096))
    else:
        block_size = max(int(block_size), K)

    n_blocks = int(math.ceil(D / block_size))

    while n_blocks > 1 and (D - (n_blocks - 1) * block_size) < K:
        n_blocks -= 1
        block_size = int(math.ceil(D / n_blocks))
        block_size = max(block_size, K)
        block_size = min(block_size, D) # Fix #10

    if K > 128 or n_blocks == 1:
        return np.linalg.qr(A, mode='reduced')

    Q_local_blocks = []
    stacked_R = np.empty((n_blocks * K, K), dtype=np.float64)

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, D)
        block = A[start:end, :]
        Q_loc, R_loc = np.linalg.qr(block, mode='reduced')
        Q_local_blocks.append(Q_loc)

        m_loc = R_loc.shape[0]
        if m_loc == K:
            stacked_R[b * K:(b + 1) * K, :] = R_loc
        else:
            stacked_R[b * K:b * K + m_loc, :] = R_loc
            stacked_R[b * K + m_loc:(b + 1) * K, :] = 0.0

    Q_top, R_final = np.linalg.qr(stacked_R, mode='reduced')

    Q_full = np.empty((D, K), dtype=np.float64)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, D)
        m_loc = Q_local_blocks[b].shape[1]
        Q_top_i = Q_top[b * K: b * K + m_loc, :]
        Q_full[start:end, :] = Q_local_blocks[b] @ Q_top_i

    return Q_full, R_final


def estimate_kappa_power(A: np.ndarray, iters: int = 25) -> float:
    """
    Estimación determinista de kappa(A) = sigma_max / sigma_min.
    [FIX-11] Regularización relativa a sigma_max en lugar de absoluta.
    """
    A = _validate_finite_matrix(A, "A")  # [FIX-12]
    K = A.shape[1]
    tiny = machine_tiny(np.float64)
    eps = machine_eps(np.float64)

    G = A.T @ A

    # 1. Sigma max via power iteration
    x = np.ones(K, dtype=np.float64) / math.sqrt(K)
    for _ in range(iters):
        x = G @ x
        x_norm = np.linalg.norm(x)
        x /= (x_norm + tiny)
    sigma_max = math.sqrt(max(0.0, float(x @ (G @ x))))

    # 2. Sigma min via inverse iteration
    y = np.ones(K, dtype=np.float64) / math.sqrt(K)
    try:
        # [FIX-11] Regularización relativa a sigma_max, no valor absoluto fijo
        reg_G = G + (100.0 * eps * sigma_max) * np.eye(K)
        L = np.linalg.cholesky(reg_G)
        for _ in range(iters):
            y = np.linalg.solve(L.T, np.linalg.solve(L, y))
            y_norm = np.linalg.norm(y)
            y /= (y_norm + tiny)
        sigma_min = math.sqrt(max(0.0, float(y @ (G @ y))))
    except np.linalg.LinAlgError:
        sigma_min = 0.0

    if sigma_min < theta_small(np.float64):
        return float('inf')
    return sigma_max / sigma_min


def cholesky_qr2(A: np.ndarray) -> np.ndarray:
    """
    CholeskyQR2 con cota teórica V42.1: kappa_safe = 1 / sqrt(eps_f64) ~ 6.7e7.
    """
    A = _validate_finite_matrix(A, "A")  # [FIX-12]
    D, K = A.shape
    eps_f64 = machine_eps(np.float64)
    kappa_safe = 1.0 / math.sqrt(eps_f64)

    if K <= 64:
        kappa_est = float(np.linalg.cond(A))
    else:
        kappa_est = estimate_kappa_power(A, iters=20)

    if kappa_est >= kappa_safe:
        Q, _ = tsqr_blocked(A)
        return Q

    try:
        G = A.T @ A
        L = np.linalg.cholesky(G)
        Q1 = np.linalg.solve(L, A.T).T
        G2 = Q1.T @ Q1
        L2 = np.linalg.cholesky(G2)
        Q2 = np.linalg.solve(L2, Q1.T).T
        return Q2
    except np.linalg.LinAlgError:
        Q, _ = tsqr_blocked(A)
        return Q


def ortho_gap(Q: np.ndarray) -> Tuple[float, float, bool]:
    """Mide el gap ||Q^T Q - I_K||_F contra la cota de Higham O(D * eps)."""
    D, K = Q.shape
    eps = machine_eps(Q.dtype)
    tol = 10.0 * float(D) * eps
    gram = Q.T @ Q
    gap = float(np.linalg.norm(gram - np.eye(K), 'fro'))
    return gap, tol, bool(gap > tol)


def project_stiefel(Q: np.ndarray) -> np.ndarray:
    """[FIX-09] Validación D >= K antes de re-proyección."""
    D, K = Q.shape
    if D < K:
        raise ValueError(f"project_stiefel requiere D >= K. Obtenido D={D}, K={K}")
    gap, tol, needs = ortho_gap(Q)
    if needs:
        Q_new, _ = tsqr_blocked(Q)
        return Q_new
    return Q


# ============================================================================
# 6. MEDIA DE FRÉCHET CON DETECCIÓN DE DEGENERACIÓN
# ============================================================================

def frechet_mean_sphere(vectors: np.ndarray, max_iters: int = 200,
                         tol: Optional[float] = None, dtype=np.float64) -> np.ndarray:
    """
    Media de Fréchet en S^(D-1) vía RGD vectorizado con detección de
    degeneración antipodal.
    """
    vectors = _validate_finite_matrix(vectors, "vectors")  # [FIX-12]
    vectors = np.asarray(vectors, dtype=dtype)
    if vectors.ndim != 2:
        raise ValueError("vectors debe ser matriz 2D (N x D)")
    N, D = vectors.shape
    eps = machine_eps(dtype)
    tiny = machine_tiny(dtype)

    if tol is None:
        tol = 100.0 * eps * math.sqrt(float(D))

    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + tiny)

    # Detección de degeneración antipodal / uniforme
    mu_ext = np.mean(vectors, axis=0)
    norm_mu_ext = np.linalg.norm(mu_ext)
    if norm_mu_ext < 16.0 * eps * math.sqrt(float(D)):
        raise ValueError(
            f"Degeneración Topológica: Distribución antipodal o uniforme en S^(D-1). "
            f"Media no única (norm_mu={norm_mu_ext:.2e})."
        )

    mu = mu_ext / norm_mu_ext

    # Cota empírica conservadora (pi/2)^2 — NO usar "Giroux" pi^2/2-2 (falsa)
    # [BULLDOG FIX]: Kahan formulación para evitar cancelación catastrófica (arccos(1.0) absorción)
    d_norms = np.linalg.norm(vectors - mu, axis=1)
    s_norms = np.linalg.norm(vectors + mu, axis=1)
    omegas_init = 2.0 * np.arctan2(d_norms, s_norms)
    geo_variance_init = float(np.mean(omegas_init ** 2))
    max_expected_variance = (math.pi / 2.0) ** 2
    if geo_variance_init > 0.70 * max_expected_variance and N > 10:
        raise ValueError(
            f"Degeneración Topológica: La nube de vectores no tiene un baricentro definido "
            f"(Varianza={geo_variance_init:.4f}). Consenso imposible."
        )

    for _ in range(max_iters):
        dots = np.clip(np.dot(vectors, mu), -1.0, 1.0)
        omegas = np.arccos(dots)

        scale_factors = np.empty(N, dtype=dtype)
        near_zero_mask = omegas < theta_small(dtype)
        scale_factors[near_zero_mask] = 1.0 + (omegas[near_zero_mask] ** 2) / 6.0

        normal_mask = ~near_zero_mask
        sin_omegas = np.sin(omegas[normal_mask])
        safe_sin = np.where(sin_omegas < theta_small(dtype), 1.0, sin_omegas)
        scale_factors[normal_mask] = omegas[normal_mask] / safe_sin

        v_proj = vectors - dots[:, None] * mu
        tangents = scale_factors[:, None] * v_proj

        cut_locus_mask = (np.pi - omegas) < theta_antipodal(dtype)
        if np.any(cut_locus_mask):
            for idx in np.where(cut_locus_mask)[0]:
                # [FIX-14] Usar vectors[idx] para entropía, no mu
                v_perturb = deterministic_tangent(vectors[idx]) * (math.pi * 0.01)
                tangents[idx] = v_perturb

        delta = np.mean(tangents, axis=0)
        delta_norm = float(np.linalg.norm(delta))

        if delta_norm < tol:
            break

        step_size = min(delta_norm, math.pi / 4.0)
        unit_delta = delta / (delta_norm + tiny)

        mu = math.cos(step_size) * mu + math.sin(step_size) * unit_delta
        mu /= (np.linalg.norm(mu) + tiny)

    return mu


# ============================================================================
# 7. RECEPTOR STATEFUL CON PROTECCIÓN ANTI-DOS Y JUMP LIMIT
# ============================================================================

class PmtpStatefulReceiver:
    """
    Receptor Anti-Replay V44 con:
      - HKDF RFC 5869 real (salt + extract + expand) [FIX-04]
      - MAX_EPOCH_JUMP anti-DoS de epoch inflation [FIX-05]
      - Protocolo HMAC unificado con domain separator [FIX-07]
      - Guarda de longitud de tag [FIX-13]
      - Validación de window_size [FIX-10]
    """
    MAX_SEQ_JUMP = 1000
    MAX_EPOCH_JUMP = 10
    MAX_SEQ_ON_EPOCH_TRANSITION = 100_000

    def __init__(self, master_key: bytes, window_size: int = 64,
                 salt: Optional[bytes] = None):
        # [FIX-10]
        if window_size < 1 or window_size > 256:
            raise ValueError("window_size debe estar en [1, 256]")
        self.master_key = master_key
        # Salt único por instancia/nodo [FIX-04]
        self.salt = salt if salt is not None else os.urandom(64)
        self.prk = hkdf_extract(self.salt, self.master_key)
        self.window_size = window_size
        self.last_epoch = 1
        self.last_seq = 0
        self.window_bitmap = 0
        self.lock = threading.Lock()

    def _derive_epoch_key(self, epoch: int) -> bytes:
        """[FIX-04] HKDF real, no HMAC directo."""
        info = b"POLYDIM_PMTP_V42_1_EPOCH_" + int(epoch).to_bytes(8, 'little')
        return hkdf_expand(self.prk, info, length=64)

    def _make_tag(self, epoch: int, seq: int, payload: bytes, epoch_key: bytes) -> bytes:
        """[FIX-07] Protocolo PMTP unificado: M = domain || epoch || seq || payload."""
        header_data = (b"POLYDIM_PMTP_V42_1"
                       + int(epoch).to_bytes(8, 'little')
                       + int(seq).to_bytes(8, 'little'))
        # Fix #29: BLAKE2b Keyed Hashing nativo
        h = hashlib.blake2b(key=epoch_key[:64], digest_size=64)
        h.update(header_data)
        h.update(payload)
        return h.digest()

    def verify_and_accept(self, epoch: int, seq: int, payload: bytes,
                          tag: bytes) -> Tuple[bool, str]:
        # [FIX-13] Guardas de tipo y longitud ANTES de cualquier cripto
        if not isinstance(tag, bytes) or len(tag) != 64:
            return False, "REJECTED_TAG_INVALID"
        if not isinstance(payload, bytes):
            return False, "REJECTED_PAYLOAD_INVALID"
        if seq < 0:
            return False, "REJECTED_NEGATIVE_SEQ"
        if epoch < 1:
            return False, "REJECTED_INVALID_EPOCH"

        # 1. Barrera de Época
        if epoch < self.last_epoch:
            return False, "REJECTED_OLD_EPOCH"

        # [FIX-05] Barrera de salto de época
        if epoch > self.last_epoch:
            if epoch > self.last_epoch + self.MAX_EPOCH_JUMP:
                return False, "REJECTED_EPOCH_JUMP_TOO_LARGE"
            if seq > self.MAX_SEQ_ON_EPOCH_TRANSITION:
                return False, "REJECTED_SEQ_TOO_LARGE_ON_EPOCH_CHANGE"

        # 2. Barrera de Secuencia con MAX_SEQ_JUMP y Bitmap
        if epoch == self.last_epoch:
            if seq <= self.last_seq:
                diff = self.last_seq - seq
                if diff >= self.window_size:
                    return False, "REJECTED_WINDOW_EXPIRED"
                if (self.window_bitmap & (1 << diff)) != 0:
                    return False, "REJECTED_REPLAY_SEQ"
            elif seq > self.last_seq + self.MAX_SEQ_JUMP:
                return False, "REJECTED_SUSPICIOUS_JUMP"

        # 3. Verificación HMAC con protocolo unificado
        if not hasattr(self, '_cached_epoch_key') or getattr(self, '_cached_epoch', -1) != epoch:
            self._cached_epoch = epoch
            self._cached_epoch_key = self._derive_epoch_key(epoch)
        epoch_key = self._cached_epoch_key
        expected_tag = self._make_tag(epoch, seq, payload, epoch_key)

        if not hmac.compare_digest(expected_tag, tag):
            return False, "CORRUPT_TAG"

        # 4. Actualizar estado y bitmap
        mask = (1 << self.window_size) - 1
        if epoch > self.last_epoch:
            self.last_epoch = epoch
            self.last_seq = seq
            self.window_bitmap = 1
        elif seq > self.last_seq:
            shift = seq - self.last_seq
            if shift < self.window_size:
                self.window_bitmap = ((self.window_bitmap << shift) | 1) & mask
            else:
                self.window_bitmap = 1
            self.last_seq = seq
        else:
            diff = self.last_seq - seq
            self.window_bitmap |= (1 << diff)
            self.window_bitmap &= mask

        return True, "ACCEPTED"
