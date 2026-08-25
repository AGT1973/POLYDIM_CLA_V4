# polydim_motor_v47.py
# Motor de Computabilidad Geométrica SOTA en S^(D-1) - POLYDIM V47
# DOGMA CERO: El software no asume. El software interroga el silicio.
# ============================================================================

import os
import sys
import math
import hashlib
import hmac
import ctypes
import threading
import datetime
from typing import Tuple, Optional, Union, List, Dict, Any

import numpy as np

# Soporte JAX opcional con precisión estricta de 64 bits (float64)
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    JAX_OK = True
    _test_f64 = jnp.array(1.0, dtype=jnp.float64)
    if _test_f64.dtype != jnp.float64:
        JAX_OK = False
except Exception:
    import numpy as jnp
    JAX_OK = False


# ============================================================================
# SECCION 1: CONTRATO DE SILICIO DINAMICO (DOGMA CERO - ANTI-HARDCODING)
# ============================================================================

class SiliconContract:
    """
    Interroga las capacidades físicas del silicio y del SO en tiempo de ejecución.
    DOGMA CERO: Estricta prohibición de hardcodear parámetros de hardware o mágicos.
    """
    def __init__(self):
        self.cache_line_bytes = self._interrogate_cache_line()
        self.simd_width_bytes = self._interrogate_simd_width()
        self.optimal_workers = self._interrogate_workers()
        self.subsample_len = self.cache_line_bytes * 2  # 128 bytes por defecto derivado de caché

    def _interrogate_cache_line(self) -> int:
        # Windows OS Kernel Interrogation
        if sys.platform == 'win32':
            try:
                from ctypes import wintypes
                class CACHE_DESCRIPTOR(ctypes.Structure):
                    _fields_ = [('Level', ctypes.c_byte), ('Associativity', ctypes.c_byte),
                                ('LineSize', ctypes.c_ushort), ('Size', ctypes.c_ulong), ('Type', ctypes.c_int)]
                class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                    class _U(ctypes.Union):
                        _fields_ = [('ProcessorCore', ctypes.c_byte), ('NumaNode', ctypes.c_ulong),
                                    ('Cache', CACHE_DESCRIPTOR), ('Reserved', ctypes.c_ulonglong * 2)]
                    _fields_ = [('ProcessorMask', ctypes.c_size_t), ('Relationship', ctypes.c_int), ('u', _U)]

                buf_len = wintypes.DWORD(0)
                ctypes.windll.kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buf_len))
                num_elem = buf_len.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                arr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * num_elem)()
                if ctypes.windll.kernel32.GetLogicalProcessorInformation(arr, ctypes.byref(buf_len)):
                    for i in range(num_elem):
                        if arr[i].Relationship == 2:  # RelationCache
                            size = int(arr[i].u.Cache.LineSize)
                            if size > 0:
                                return size
            except Exception:
                pass

        # Linux sysfs Interrogation
        elif sys.platform.startswith('linux'):
            try:
                with open('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size', 'r') as f:
                    return int(f.read().strip())
            except Exception:
                pass

        # macOS sysctl Interrogation
        elif sys.platform == 'darwin':
            try:
                import subprocess
                res = subprocess.run(['sysctl', '-n', 'hw.cachelinesize'], capture_output=True, text=True)
                return int(res.stdout.strip())
            except Exception:
                pass

        # Fallback interrogación por sysconf si existe
        try:
            val = os.sysconf('SC_LEVEL1_DCACHE_LINESIZE')
            if isinstance(val, int) and val > 0:
                return val
        except Exception:
            pass

        return ctypes.sizeof(ctypes.c_double) * 8  # 64 bytes derivados dinámicamente

    def _interrogate_simd_width(self) -> int:
        try:
            if sys.platform.startswith('linux'):
                with open('/proc/cpuinfo', 'r') as f:
                    info = f.read()
                    if 'avx512f' in info:
                        return 64
                    if 'avx2' in info or 'avx' in info:
                        return 32
            elif sys.platform == 'darwin':
                return 32
            elif sys.platform == 'win32':
                # Query AVX support via IsProcessorFeaturePresent (PF_AVX2_INSTRUCTIONS_AVAILABLE = 40)
                if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):
                    return 32
        except Exception:
            pass
        return ctypes.sizeof(ctypes.c_double) * 4  # 32 bytes derivados dinámicamente

    def _interrogate_workers(self) -> int:
        try:
            cnt = os.cpu_count()
            if cnt:
                return max(1, cnt)
        except Exception:
            pass
        return 4

    def machine_eps(self, dtype=np.float64) -> float:
        return float(np.finfo(dtype).eps)

    def machine_tiny(self, dtype=np.float64) -> float:
        return float(np.finfo(dtype).tiny)

    def get_collinearity_threshold(self, D: int, dtype=np.float64) -> float:
        eps_val = self.machine_eps(dtype)
        sqrt_D = math.sqrt(float(max(1, D)))
        return 16.0 * eps_val * sqrt_D

    def get_antipodal_threshold(self, D: int, dtype=np.float64) -> float:
        eps_val = self.machine_eps(dtype)
        sqrt_D = math.sqrt(float(max(1, D)))
        return max(100.0 * eps_val * sqrt_D, math.sqrt(eps_val))

    def antipodal_step_rad(self, D: int, dtype=np.float64) -> float:
        return math.pi * self.machine_eps(dtype) * math.sqrt(float(max(1, D))) * 1000.0


HOST_SILICON = SiliconContract()

def machine_eps(dtype=np.float64) -> float:
    return HOST_SILICON.machine_eps(dtype)

def machine_tiny(dtype=np.float64) -> float:
    return HOST_SILICON.machine_tiny(dtype)

def theta_small(dtype=np.float64, D: int = 1) -> float:
    return HOST_SILICON.get_collinearity_threshold(D, dtype)

def theta_antipodal(dtype=np.float64, D: int = 1) -> float:
    return HOST_SILICON.get_antipodal_threshold(D, dtype)


# ============================================================================
# SECCION 2: CARGA DINAMICA DE DLLs NATIVAS (C++ AVX2 y Rust MPMC)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DLL_PATH = os.path.join(BASE_DIR, "slerp_kernel_v47.dll")
RUST_DLL_PATH = os.path.join(BASE_DIR, "lib_v47.dll")

CPP_LIB = None
if os.path.exists(CPP_DLL_PATH):
    try:
        CPP_LIB = ctypes.CDLL(CPP_DLL_PATH)
        CPP_LIB.slerp.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        ]
        CPP_LIB.slerp.restype = ctypes.c_int
        CPP_LIB.get_last_error_safe.argtypes = []
        CPP_LIB.get_last_error_safe.restype = ctypes.c_char_p
    except Exception:
        CPP_LIB = None

RUST_LIB = None
if os.path.exists(RUST_DLL_PATH):
    try:
        RUST_LIB = ctypes.CDLL(RUST_DLL_PATH)
        RUST_LIB.pmtp_ring_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        RUST_LIB.pmtp_ring_create.restype = ctypes.c_void_p
        RUST_LIB.pmtp_ring_free.argtypes = [ctypes.c_void_p]
        RUST_LIB.pmtp_ring_free.restype = None
        RUST_LIB.pmtp_ring_push.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        RUST_LIB.pmtp_ring_push.restype = ctypes.c_int
        RUST_LIB.pmtp_ring_pop.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        RUST_LIB.pmtp_ring_pop.restype = ctypes.c_int
    except Exception:
        RUST_LIB = None


def slerp_c(p: np.ndarray, q: np.ndarray, t: float) -> Tuple[bool, np.ndarray]:
    if CPP_LIB is None:
        return False, np.array([])
    if not isinstance(p, np.ndarray) or not isinstance(q, np.ndarray):
        return False, np.array([])
    
    D = len(p)
    if D != len(q) or D < 2:
        return False, np.array([])

    p_c = np.ascontiguousarray(p, dtype=np.float64)
    q_c = np.ascontiguousarray(q, dtype=np.float64)
    out_c = np.empty(D, dtype=np.float64)
    scratch_c = np.empty(D, dtype=np.float64)

    ret = CPP_LIB.slerp(
        p_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(t),
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(D),
        scratch_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(D)
    )

    if ret == 0:
        return True, out_c
    elif ret == 2:
        # C++ delegó el régimen antipodal al Host Python para mantener unificada la tangente determinista SHAKE-256
        return False, np.array([])
    else:
        err_msg = CPP_LIB.get_last_error_safe().decode('utf-8', errors='ignore')
        raise RuntimeError(f"Kernel C++ SLERP error code {ret}: {err_msg}")


# ============================================================================
# SECCION 3: FUNCIONES AUXILIARES Y DENEGACION PREFLIGHT
# ============================================================================

def check_memory_available(required_bytes: int, safety_margin: float = 0.8):
    try:
        import psutil
        available = psutil.virtual_memory().available
        if required_bytes > available * safety_margin:
            raise MemoryError(f"Preflight rechaza alocacion de {required_bytes / (1024**3):.2f} GB; "
                              f"disponible: {available / (1024**3):.2f} GB")
    except ImportError:
        pass

def validate_finite_vector(v: np.ndarray, name: str = "v") -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError(f"{name} debe ser un vector 1D no vacio")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contiene valores no finitos (NaN/Inf)")
    return np.ascontiguousarray(v, dtype=np.float64)

def validate_finite_matrix(A: np.ndarray, name: str = "A") -> np.ndarray:
    if not isinstance(A, np.ndarray):
        A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"{name} debe ser una matriz 2D no vacia")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contiene valores no finitos (NaN/Inf)")
    return np.ascontiguousarray(A, dtype=np.float64)


# ============================================================================
# SECCION 4: ALGORITMOS MATEMATICOS SOTA EN S^(D-1)
# ============================================================================

def deterministic_tangent(p: np.ndarray) -> np.ndarray:
    p = validate_finite_vector(p, "p")
    D = len(p)
    if D < 2:
        return np.zeros(D, dtype=np.float64)
    min_idx = np.argmin(np.abs(p))
    p_min = p[min_idx]
    v = -p_min * p
    v[min_idx] += 1.0
    v_norm = np.linalg.norm(v)
    return v / max(v_norm, machine_tiny(np.float64))

def slerp_stable(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    D = len(p)
    tiny = machine_tiny(np.float64)
    eps = machine_eps(np.float64)

    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    p_unit = p / (p_norm + tiny)
    q_unit = q / (q_norm + tiny)

    ok, res_c = slerp_c(p_unit, q_unit, t)
    if ok:
        return res_c

    d_norm = np.linalg.norm(p_unit - q_unit)
    s_norm = np.linalg.norm(p_unit + q_unit)
    omega = 2.0 * np.arctan2(d_norm, s_norm)

    t_small = theta_small(np.float64, D)
    t_anti = theta_antipodal(np.float64, D)

    if omega < t_small:
        res = p_unit + t * (q_unit - p_unit)
        return res / (np.linalg.norm(res) + tiny)

    if (np.pi - omega) < t_anti:
        v_anti = deterministic_tangent(p_unit)
        res = p_unit * np.cos(t * np.pi) + v_anti * np.sin(t * np.pi)
        return res / (np.linalg.norm(res) + tiny)

    sin_omega = np.sin(omega)
    safe_sin = max(abs(sin_omega), eps)
    s0 = np.sin((1.0 - t) * omega) / safe_sin
    s1 = np.sin(t * omega) / safe_sin
    res = s0 * p_unit + s1 * q_unit
    return res / (np.linalg.norm(res) + tiny)


# ============================================================================
# SECCION 5: BATCHING VECTORIZADO CON JAX JIT (DOGMA CERO EN XLA)
# ============================================================================

if JAX_OK:
    @jax.jit
    def _jax_slerp_batch_impl(P_j, Q_j, T_arr_j):
        D_j = P_j.shape[1]
        tiny_j = jnp.finfo(P_j.dtype).tiny
        eps_j = jnp.finfo(P_j.dtype).eps
        sqrt_D_j = jnp.sqrt(D_j.astype(P_j.dtype))

        t_small_val_j = 16.0 * eps_j * sqrt_D_j
        t_anti_val_j = jnp.maximum(100.0 * eps_j * sqrt_D_j, jnp.sqrt(eps_j))

        P_norms = jnp.linalg.norm(P_j, axis=1, keepdims=True)
        Q_norms = jnp.linalg.norm(Q_j, axis=1, keepdims=True)
        P_unit = P_j / (P_norms + tiny_j)
        Q_unit = Q_j / (Q_norms + tiny_j)

        d_norms = jnp.linalg.norm(P_unit - Q_unit, axis=1)
        s_norms = jnp.linalg.norm(P_unit + Q_unit, axis=1)
        omegas = 2.0 * jnp.arctan2(d_norms, s_norms)

        def _single_slerp(p_i, q_i, t_i, om_i):
            is_near0 = om_i < t_small_val_j
            is_anti = (jnp.pi - om_i) < t_anti_val_j

            lerp_val = p_i + t_i * (q_i - p_i)
            lerp_res = lerp_val / (jnp.linalg.norm(lerp_val) + tiny_j)

            min_idx = jnp.argmin(jnp.abs(p_i))
            p_min = p_i[min_idx]
            v = -p_min * p_i
            v = v.at[min_idx].add(1.0)
            v_norm = jnp.linalg.norm(v)
            v_anti = v / jnp.maximum(v_norm, tiny_j)

            res_anti_val = p_i * jnp.cos(t_i * jnp.pi) + v_anti * jnp.sin(t_i * jnp.pi)
            res_anti = res_anti_val / (jnp.linalg.norm(res_anti_val) + tiny_j)

            sin_om = jnp.sin(om_i)
            safe_sin = jnp.maximum(jnp.abs(sin_om), eps_j)
            s0 = jnp.sin((1.0 - t_i) * om_i) / safe_sin
            s1 = jnp.sin(t_i * om_i) / safe_sin
            norm_val = s0 * p_i + s1 * q_i
            norm_res = norm_val / (jnp.linalg.norm(norm_val) + tiny_j)

            return jnp.where(is_near0, lerp_res, jnp.where(is_anti, res_anti, norm_res))

        return jax.vmap(_single_slerp)(P_unit, Q_unit, T_arr_j, omegas)

def slerp_batch(P: np.ndarray, Q: np.ndarray, T: Union[float, np.ndarray]) -> np.ndarray:
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P y Q deben ser matrices 2D (N, D)")
    if P.shape != Q.shape:
        raise ValueError(f"Dimensiones no coinciden: {P.shape} vs {Q.shape}")
    N, D = P.shape

    if isinstance(T, (int, float)):
        T_arr = np.full(N, float(T), dtype=np.float64)
    else:
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 1 and len(T_arr) == N:
            pass
        else:
            raise ValueError(f"T debe ser escalar o vector 1D de longitud {N}")

    if JAX_OK:
        P_jax = jnp.asarray(P, dtype=jnp.float64)
        Q_jax = jnp.asarray(Q, dtype=jnp.float64)
        T_jax = jnp.asarray(T_arr, dtype=jnp.float64)
        return np.asarray(_jax_slerp_batch_impl(P_jax, Q_jax, T_jax))

    tiny = machine_tiny(np.float64)
    eps = machine_eps(np.float64)
    t_small_val = theta_small(np.float64, D)
    t_anti_val = theta_antipodal(np.float64, D)

    P_norms = np.linalg.norm(P, axis=1, keepdims=True)
    Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
    P_unit = P / (P_norms + tiny)
    Q_unit = Q / (Q_norms + tiny)

    d_norms = np.linalg.norm(P_unit - Q_unit, axis=1)
    s_norms = np.linalg.norm(P_unit + Q_unit, axis=1)
    omegas = 2.0 * np.arctan2(d_norms, s_norms)

    out = np.empty_like(P_unit)
    near0_mask = (omegas < t_small_val)
    anti_mask = (np.pi - omegas) < t_anti_val
    normal_mask = ~(near0_mask | anti_mask)

    if np.any(near0_mask):
        for idx in np.where(near0_mask)[0]:
            t_i = T_arr[idx]
            lerp = P_unit[idx] + t_i * (Q_unit[idx] - P_unit[idx])
            out[idx] = lerp / (np.linalg.norm(lerp) + tiny)

    if np.any(anti_mask):
        for idx in np.where(anti_mask)[0]:
            v_anti = deterministic_tangent(P_unit[idx])
            t_i = T_arr[idx]
            res_anti = P_unit[idx] * np.cos(t_i * np.pi) + v_anti * np.sin(t_i * np.pi)
            out[idx] = res_anti / (np.linalg.norm(res_anti) + tiny)

    if np.any(normal_mask):
        for idx in np.where(normal_mask)[0]:
            t_i = T_arr[idx]
            om_i = omegas[idx]
            sin_om = np.sin(om_i)
            safe_sin = max(abs(sin_om), eps)
            s0 = np.sin((1.0 - t_i) * om_i) / safe_sin
            s1 = np.sin(t_i * om_i) / safe_sin
            norm_val = s0 * P_unit[idx] + s1 * Q_unit[idx]
            out[idx] = norm_val / (np.linalg.norm(norm_val) + tiny)

    return out


# ============================================================================
# SECCION 6: MEDIA DE FRECHET Y FACTORIZACION TSQR
# ============================================================================

def frechet_mean_sphere(vectors: np.ndarray, weights: Optional[np.ndarray] = None,
                        max_iter: int = 100, tol: float = 1e-12) -> np.ndarray:
    vectors = validate_finite_matrix(vectors, "vectors").copy()
    N, D = vectors.shape
    tiny = machine_tiny(np.float64)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + tiny)

    if weights is None:
        w = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

    d_init = np.linalg.norm(vectors - vectors[0], axis=1)
    s_init = np.linalg.norm(vectors + vectors[0], axis=1)
    omegas_init = 2.0 * np.arctan2(d_init, s_init)
    mu = vectors[np.argmin(omegas_init)].copy()

    step_rad = HOST_SILICON.antipodal_step_rad(D)

    for _ in range(max_iter):
        d_norms = np.linalg.norm(vectors - mu, axis=1)
        s_norms = np.linalg.norm(vectors + mu, axis=1)
        omegas = 2.0 * np.arctan2(d_norms, s_norms)

        near_zero_mask = omegas < theta_small(np.float64, D)
        sin_omegas = np.sin(omegas)
        safe_sin = np.where(np.abs(sin_omegas) < theta_small(np.float64, D), 1.0, sin_omegas)
        cut_locus_mask = (np.pi - omegas) < theta_antipodal(np.float64, D)

        if np.any(cut_locus_mask):
            for idx in np.where(cut_locus_mask)[0]:
                v_anti = deterministic_tangent(vectors[idx])
                vectors[idx] = vectors[idx] + v_anti * step_rad
                vectors[idx] /= (np.linalg.norm(vectors[idx]) + tiny)

        factors = np.where(near_zero_mask, 1.0, omegas / safe_sin)
        tangents = (vectors - np.outer(np.dot(vectors, mu), mu)) * factors[:, np.newaxis]
        grad_tangent = np.sum(w[:, np.newaxis] * tangents, axis=0)
        grad_norm = np.linalg.norm(grad_tangent)

        if grad_norm < tol:
            break

        step_norm = min(grad_norm, np.pi / 4.0)
        direction = grad_tangent / grad_norm
        mu = mu * np.cos(step_norm) + direction * np.sin(step_norm)
        mu = mu / (np.linalg.norm(mu) + tiny)

    return mu

def tsqr_blocked(A: np.ndarray, block_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    A = validate_finite_matrix(A, "A")
    N, D = A.shape
    if block_size is None:
        block_size = max(D * 2, HOST_SILICON.cache_line_bytes * 16)
    if N <= block_size:
        return np.linalg.qr(A)
    num_blocks = (N + block_size - 1) // block_size
    R_blocks = []
    Q_blocks = []
    for i in range(num_blocks):
        block = A[i*block_size:(i+1)*block_size]
        Q_i, R_i = np.linalg.qr(block)
        R_blocks.append(R_i)
        Q_blocks.append(Q_i)
    R_stacked = np.vstack(R_blocks)
    Q_top, R_final = np.linalg.qr(R_stacked)
    Q_final = np.empty((N, D), dtype=A.dtype)
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, N)
        Q_final[start:end, :] = Q_blocks[i] @ Q_top[i*D:(i+1)*D, :]
    return Q_final, R_final


# ============================================================================
# SECCION 7: RECEPTOR DE MEMORIA TENSORIAL PMTP V47
# ============================================================================

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hashlib.blake2b(ikm, key=salt, digest_size=64).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int = 64) -> bytes:
    okm = b""
    previous = b""
    n = (length + 63) // 64
    for i in range(1, n + 1):
        previous = hashlib.blake2b(previous + info + bytes([i]), key=prk[:64], digest_size=64).digest()
        okm += previous
    return okm[:length]

class PmtpStatefulReceiver:
    MAX_EPOCH_JUMP = 5
    MAX_SEQ_ON_EPOCH_TRANSITION = 100
    MAX_SEQ_JUMP = 10000

    def __init__(self, master_key: bytes, window_size: int = 64, salt: Optional[bytes] = None):
        self.master_key = master_key
        self.salt = salt if salt is not None else os.urandom(64)
        self.prk = hkdf_extract(self.salt, self.master_key)
        self.window_size = window_size
        self.last_epoch = 1
        self.last_seq = 0
        self.window_bitmap = 0
        self.lock = threading.Lock()

    def _derive_epoch_key(self, epoch: int) -> bytes:
        info = b"POLYDIM_PMTP_V47_EPOCH_" + int(epoch).to_bytes(8, 'little')
        return hkdf_expand(self.prk, info, length=64)

    def _make_tag(self, epoch: int, seq: int, payload: bytes, epoch_key: bytes) -> bytes:
        header_data = (b"POLYDIM_PMTP_V47"
                       + int(epoch).to_bytes(8, 'little')
                       + int(seq).to_bytes(8, 'little'))
        h = hashlib.blake2b(key=epoch_key[:64], digest_size=64)
        h.update(header_data)
        h.update(payload)
        return h.digest()

    def verify_and_accept(self, epoch: int, seq: int, payload: bytes,
                          tag: bytes) -> Tuple[bool, str]:
        if not isinstance(tag, bytes) or len(tag) != 64:
            return False, "REJECTED_TAG_INVALID"
        if not isinstance(payload, bytes):
            return False, "REJECTED_PAYLOAD_INVALID"
        if seq < 0:
            return False, "REJECTED_NEGATIVE_SEQ"
        if epoch < 1:
            return False, "REJECTED_INVALID_EPOCH"

        # 1. Cómputo criptográfico Eager fuera del Lock (Paralelismo Amdahl)
        with self.lock:
            curr_epoch = self.last_epoch
            curr_seq = self.last_seq
            curr_bitmap = self.window_bitmap

        if epoch < curr_epoch:
            return False, "REJECTED_OLD_EPOCH"

        if epoch > curr_epoch:
            if epoch > curr_epoch + self.MAX_EPOCH_JUMP:
                return False, "REJECTED_EPOCH_JUMP_TOO_LARGE"
            if seq > self.MAX_SEQ_ON_EPOCH_TRANSITION:
                return False, "REJECTED_SEQ_TOO_LARGE_ON_EPOCH_CHANGE"

        if epoch == curr_epoch:
            if seq <= curr_seq:
                diff = curr_seq - seq
                if diff >= self.window_size:
                    return False, "REJECTED_WINDOW_EXPIRED"
                if (curr_bitmap & (1 << diff)) != 0:
                    return False, "REJECTED_REPLAY_SEQ"
            elif seq > curr_seq + self.MAX_SEQ_JUMP:
                return False, "REJECTED_SUSPICIOUS_JUMP"

        epoch_key = self._derive_epoch_key(epoch)
        expected_tag = self._make_tag(epoch, seq, payload, epoch_key)

        if not hmac.compare_digest(expected_tag, tag):
            return False, "CORRUPT_TAG"

        # 2. Mutación atómica del estado dentro del Lock (Microsegundos)
        with self.lock:
            if epoch < self.last_epoch:
                return False, "REJECTED_OLD_EPOCH"
            if epoch == self.last_epoch and seq <= self.last_seq:
                diff = self.last_seq - seq
                if diff >= self.window_size:
                    return False, "REJECTED_WINDOW_EXPIRED"
                if (self.window_bitmap & (1 << diff)) != 0:
                    return False, "REJECTED_REPLAY_SEQ"

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
            elif seq <= self.last_seq:
                diff = self.last_seq - seq
                self.window_bitmap |= (1 << diff)
                self.window_bitmap &= mask

            return True, "ACCEPTED"
