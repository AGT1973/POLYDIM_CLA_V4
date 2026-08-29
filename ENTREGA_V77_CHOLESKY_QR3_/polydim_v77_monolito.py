"""
POLYDIM V75 MONOLITH - BULLDOG REDTEAM EDITION
Protocolo de Comunicación Nativa Tensorial (PMTP) & Geometría Diferencial en JAX.
Arquitectura de Enjambre (Swarm Architecture) + XLA FFI Zero-Copy Puro (Anti-GIL).

Este archivo consolida las soluciones a los 34 vectores asintóticos (Capa Micro)
y la infraestructura Gossip/SHM (Capa Macro).
"""

import os
import sys
import ctypes
import hashlib
import hmac
import socket
import struct
import tempfile
import threading
import time
import uuid
import warnings
import random
from queue import Queue
from typing import Tuple, Dict, Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
import ml_dtypes
from jax.ffi import ffi_call

# Fuerza X64 para evitar errores de precisión en S^(D-1)
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. FUENTES NATIVOS (C++ y RUST) - XLA CUSTOM CALL SIGNATURE
# ==============================================================================

CPP_SOURCE = """
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>

#ifdef _WIN32
#define EXPORT_SYM __declspec(dllexport)
#else
#define EXPORT_SYM __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// XLA Legacy Custom Call Signature (CPU)
EXPORT_SYM void polydim_cpp_householder_xla(double* out, const void** in) {
    // Escudo 1: Null-Pointer Guards
    if (!out || !in || !in[0] || !in[1] || !in[2]) return;

    const double* x = reinterpret_cast<const double*>(in[0]);
    const double* v = reinterpret_cast<const double*>(in[1]);
    const uint64_t* dim_ptr = reinterpret_cast<const uint64_t*>(in[2]);
    
    // Escudo 2: Verificación de Alineación SIMD
    if (reinterpret_cast<uintptr_t>(out) % alignof(double) != 0 ||
        reinterpret_cast<uintptr_t>(x) % alignof(double) != 0 ||
        reinterpret_cast<uintptr_t>(v) % alignof(double) != 0 ||
        reinterpret_cast<uintptr_t>(dim_ptr) % alignof(uint64_t) != 0) {
        return;
    }

    size_t dim = static_cast<size_t>(*dim_ptr);
    if (dim == 0) return;
    
    // Escudo 3: Rechazo absoluto de aliasing parcial (Capa 4 - Vectorización SIMD)
    if (x != out && ((x < out + dim) && (out < x + dim))) return;
    
    double v_max = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double abs_v = std::abs(v[i]);
        if (abs_v > v_max) v_max = abs_v;
    }
    
    if (v_max < 1e-30) {
        if (x != out) {
            for (size_t i = 0; i < dim; ++i) out[i] = x[i];
        }
        return;
    }
    
    double v_norm_sq = 0.0;
    double dot_xv = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double v_scaled = v[i] / v_max;
        v_norm_sq += v_scaled * v_scaled;
        dot_xv += x[i] * v_scaled;
    }
    
    double scale = 2.0 * dot_xv / v_norm_sq;
    for (size_t i = 0; i < dim; ++i) {
        out[i] = x[i] - scale * (v[i] / v_max);
    }
}

#ifdef __cplusplus
}
#endif
"""

RUST_SOURCE = """
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[no_mangle]
pub extern "C" fn polydim_rust_householder_xla(
    out_ptr: *mut f64,
    in_ptrs: *const *const c_void,
) {
    // Escudo 1: Null-Pointer Guards
    if out_ptr.is_null() || in_ptrs.is_null() {
        return;
    }

    // Escudo 2: Atrapado de pánicos en frontera FFI (Cero UB)
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let ins = unsafe { std::slice::from_raw_parts(in_ptrs, 3) };
        if ins[0].is_null() || ins[1].is_null() || ins[2].is_null() {
            return;
        }

        let x_ptr = ins[0] as *const f64;
        let v_ptr = ins[1] as *const f64;
        let dim_ptr = ins[2] as *const u64;

        // Escudo 3: Verificación de Alineación SIMD (8 bytes para f64)
        let align_f64 = std::mem::align_of::<f64>();
        if (out_ptr as usize) % align_f64 != 0
            || (x_ptr as usize) % align_f64 != 0
            || (v_ptr as usize) % align_f64 != 0
            || (dim_ptr as usize) % std::mem::align_of::<u64>() != 0
        {
            return;
        }

        let dim = unsafe { *dim_ptr } as usize;
        if dim == 0 {
            return;
        }

        let x_addr = x_ptr as usize;
        let out_addr = out_ptr as usize;
        let byte_len = match dim.checked_mul(std::mem::size_of::<f64>()) {
            Some(len) => len,
            None => return,
        };

        // Escudo 4: Rechazo absoluto de aliasing parcial
        if x_addr != out_addr {
            let x_end = x_addr.checked_add(byte_len).unwrap_or(usize::MAX);
            let out_end = out_addr.checked_add(byte_len).unwrap_or(usize::MAX);
            if x_addr < out_end && out_addr < x_end {
                return;
            }
        }

        let x = unsafe { std::slice::from_raw_parts(x_ptr, dim) };
        let v = unsafe { std::slice::from_raw_parts(v_ptr, dim) };
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, dim) };

        let mut v_max: f64 = 0.0;
        for i in 0..dim {
            let abs_v = v[i].abs();
            if abs_v > v_max {
                v_max = abs_v;
            }
        }

        if v_max < 1e-30 {
            if x_addr != out_addr {
                unsafe { std::ptr::copy_nonoverlapping(x_ptr, out_ptr, dim) };
            }
            return;
        }

        let mut v_norm_sq = 0.0;
        let mut dot_xv = 0.0;
        for i in 0..dim {
            let v_s = v[i] / v_max;
            v_norm_sq += v_s * v_s;
            dot_xv += x[i] * v_s;
        }

        let scale = 2.0 * dot_xv / v_norm_sq;
        for i in 0..dim {
            out[i] = x[i] - scale * (v[i] / v_max);
        }
    }));
}
"""

# ==============================================================================
# 2. BRIDGE FFI (ZERO-COPY JAX.FFI & TOCTOU-SAFE)
# ==============================================================================
class NativeFFIBridge:
    _rust_dll = None
    _cpp_dll = None
    _init_lock = threading.Lock()
    _temp_files = []
    _xla_registered = False

    @classmethod
    def initialize(cls):
        with cls._init_lock:
            if cls._xla_registered:
                return

            cache_dir = os.path.expanduser(f"~/.cache/polydim_ffi/worker_{os.getpid()}")
            os.makedirs(cache_dir, exist_ok=True)

            import shutil
            import subprocess

            # RUST COMPILATION
            rust_uuid = uuid.uuid4().hex
            rust_src_path = os.path.join(cache_dir, f"kernel_rust_{rust_uuid}.rs")
            rust_dll_temp = os.path.join(cache_dir, f"rust_lib_{rust_uuid}.dll")
            rust_dll_final = os.path.join(cache_dir, "rust_lib_v75_xla.dll")
            
            with open(rust_src_path, "w") as f:
                f.write(RUST_SOURCE)
            
            rustc = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
            try:
                subprocess.run([rustc, "--crate-type", "cdylib", rust_src_path, "-o", rust_dll_temp], check=True, capture_output=True)
                os.replace(rust_dll_temp, rust_dll_final)
                cls._rust_dll = ctypes.CDLL(rust_dll_final)
            except Exception as e:
                warnings.warn(f"Rust FFI no disponible. Fallback a C++. Error: {e}")

            # C++ COMPILATION
            cpp_uuid = uuid.uuid4().hex
            cpp_src_path = os.path.join(cache_dir, f"kernel_cpp_{cpp_uuid}.cpp")
            cpp_dll_temp = os.path.join(cache_dir, f"cpp_lib_{cpp_uuid}.dll")
            cpp_dll_final = os.path.join(cache_dir, "cpp_lib_v75_xla.dll")

            with open(cpp_src_path, "w") as f:
                f.write(CPP_SOURCE)
            
            if not cls._rust_dll:
                try:
                    if sys.platform == "win32":
                        subprocess.run(["cl.exe", "/LD", "/O2", cpp_src_path, f"/Fe{cpp_dll_temp}"], check=True, capture_output=True)
                    else:
                        subprocess.run(["g++", "-shared", "-O3", "-fPIC", cpp_src_path, "-o", cpp_dll_temp], check=True, capture_output=True)
                    
                    os.replace(cpp_dll_temp, cpp_dll_final)
                    cls._cpp_dll = ctypes.CDLL(cpp_dll_final)
                except Exception as e:
                    warnings.warn(f"C++ FFI no disponible. Fallback a Python JAX puro. Error: {e}")

            cls._temp_files.extend([rust_src_path, cpp_src_path])

            # -----------------------------------------------------------------
            # XLA PYCAPSULE INJECTION (KILL THE GIL)
            # -----------------------------------------------------------------
            if cls._rust_dll or cls._cpp_dll:
                ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
                ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
                
                if cls._rust_dll:
                    ptr = ctypes.cast(cls._rust_dll.polydim_rust_householder_xla, ctypes.c_void_p)
                else:
                    ptr = ctypes.cast(cls._cpp_dll.polydim_cpp_householder_xla, ctypes.c_void_p)
                    
                capsule = ctypes.pythonapi.PyCapsule_New(ptr, b"xla._CUSTOM_CALL_TARGET", None)
                jax.ffi.register_ffi_target("householder_xla", capsule, api_version=1)
                cls._xla_registered = True

    @classmethod
    def cleanup(cls):
        # FIX V74.1: NUNCA LLAMAR A dlclose() o FreeLibrary().
        for path in cls._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

    @classmethod
    @jax.jit
    def householder_reflect(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Se asume que initialize() fue llamado externamente antes del primer JIT,
        # o que el fallback puro se usa si no hay DLLs disponibles.
        if not cls._xla_registered:
            return cls._jax_fallback(x, v)
            
        dim = jnp.array([x.shape[-1]], dtype=jnp.uint64)
        out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        
        # El XLA Custom Call inyecta el Kernel nativo puro en el Grafo (Zero-Copy)
        return jax.lax.custom_call(
            "householder_xla",
            out_shape,
            operands=(x, v, dim)
        )

    @classmethod
    def _jax_fallback(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Fallback diferenciable puro en JAX
        v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)
        dot_xv = jnp.sum(x * v, axis=-1, keepdims=True)
        safe_v_norm_sq = jnp.where(v_norm_sq < 1e-30, 1.0, v_norm_sq)
        scale = 2.0 * dot_xv / safe_v_norm_sq
        reflection = x - scale * v
        return jnp.where(v_norm_sq < 1e-30, x, reflection)

# ==============================================================================
# 3. KERNELS GEOMÉTRICOS Y MATEMÁTICOS (SD-1)
# ==============================================================================
def safe_norm(x: jnp.ndarray, axis=-1, keepdims: bool = True) -> jnp.ndarray:
    eps = jnp.finfo(x.dtype).eps
    axis_t = (axis,) if isinstance(axis, int) else tuple(axis)
    scale = jnp.max(jnp.abs(x), axis=axis_t, keepdims=True)
    
    safe_scale = jnp.where(scale == 0.0, 1.0, scale)
    scaled_x = x / safe_scale
    
    # FIX V74.1: safe_norm maneja álgebra hermitiana estrictamente
    if x.dtype.kind == 'c':
        sq_sum = jnp.sum((scaled_x * jnp.conj(scaled_x)).real, axis=axis_t, keepdims=keepdims)
    else:
        sq_sum = jnp.sum(scaled_x * scaled_x, axis=axis_t, keepdims=keepdims)
        
    norm = scale * jnp.sqrt(jnp.where(scale == 0.0, 1.0, sq_sum))
    
    if not keepdims:
        norm = jnp.squeeze(norm, axis=axis_t)
    
    # La norma de un vector (real o complejo) SIEMPRE es un escalar real.
    real_dtype = jnp.finfo(x.dtype).dtype if x.dtype.kind == 'c' else x.dtype
    return norm.astype(real_dtype)


def safe_dot(a: jnp.ndarray, b: jnp.ndarray, keepdims: bool = False) -> jnp.ndarray:
    # Asegurar precisión (convertir enteros a float64)
    a = a.astype(jnp.float64) if not jnp.issubdtype(a.dtype, jnp.inexact) else a
    b = b.astype(jnp.float64) if not jnp.issubdtype(b.dtype, jnp.inexact) else b
    
    res = jnp.sum(a * b, axis=-1, keepdims=keepdims)
    return res


class GeodesicKernels:
    @staticmethod
    @jax.jit
    def exp_map(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        v_norm = safe_norm(v, keepdims=True)
        is_zero = v_norm < eps
        
        safe_v_norm = jnp.where(is_zero, 1.0, v_norm)
        v_tangent = v / safe_v_norm
        
        cos_t = jnp.cos(v_norm)
        sin_t = jnp.sin(v_norm)
        
        result = cos_t * x + sin_t * v_tangent
        result = jnp.where(is_zero, x, result)
        
        return result / jnp.maximum(safe_norm(result, keepdims=True), eps)

    @staticmethod
    @jax.jit
    def _log_map_unit(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        dot = jnp.clip(safe_dot(x, y, keepdims=True), -1.0, 1.0)
        
        # FIX V77: SOTA Asintótico - Distancia cordal con arcsin para evitar
        # singularidades del gradiente de arccos en x ~ y (El fin de acos(1.0))
        dist = safe_norm(x - y, keepdims=True)
        # Escudo Epsilon (Anti-NaN en colisión exacta)
        safe_dist = jnp.maximum(dist, eps)
        theta = 2.0 * jnp.arcsin(jnp.clip(safe_dist / 2.0, 0.0, 1.0))
        
        proj = y - dot * x
        proj_norm = safe_norm(proj, keepdims=True)
        safe_proj_norm = jnp.where(proj_norm < eps, 1.0, proj_norm)
        
        return theta * (proj / safe_proj_norm)

    @staticmethod
    @jax.jit
    def log_map(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        xu = x / jnp.maximum(safe_norm(x, keepdims=True), eps)
        yu = y / jnp.maximum(safe_norm(y, keepdims=True), eps)
        dim = x.shape[-1]
        
        # FIX V74.1: Guarda de identidad por distancia Euclidiana
        dist_sq = jnp.sum((xu - yu)**2, axis=-1, keepdims=True)
        is_identity = dist_sq < (eps * dim) ** 2
        is_antipodal = dist_sq > (2.0 - eps * dim) ** 2
        
        log_normal = GeodesicKernels._log_map_unit(xu, yu)
        
        e0 = jnp.zeros_like(xu).at[..., 0].set(1.0)
        e1 = jnp.zeros_like(xu).at[..., -1].set(1.0)
        use_e1 = jnp.abs(xu[..., 0:1]) > 0.9
        e_base = jnp.where(use_e1, e1, e0)
        
        proj_e = e_base - safe_dot(e_base, xu, keepdims=True) * xu
        u_fallback = proj_e / jnp.maximum(safe_norm(proj_e, keepdims=True), eps)
        log_antipodal = jnp.pi * u_fallback
        
        result = jnp.where(is_antipodal, jax.lax.stop_gradient(log_antipodal), log_normal)
        result = jnp.where(is_identity, 0.0, result)
        
        return result

    @staticmethod
    @jax.jit
    def log_map_newton(x: jnp.ndarray, y: jnp.ndarray, max_iter: int = 10, tol: float = 1e-6) -> jnp.ndarray:
        # FIX V74.1: Convergencia adaptativa con while_loop
        eps = jnp.finfo(x.dtype).eps
        xu = x / jnp.maximum(safe_norm(x, keepdims=True), eps)
        yu = y / jnp.maximum(safe_norm(y, keepdims=True), eps)
        
        v0 = GeodesicKernels.log_map(xu, yu) # Bootstrapping
        
        def cond_fn(state):
            v, residual_norm, i = state
            return (residual_norm > tol) & (i < max_iter)
            
        def body_fn(state):
            v, _, i = state
            y_approx = GeodesicKernels.exp_map(xu, v)
            
            # Transporte inverso
            residual = GeodesicKernels._log_map_unit(y_approx, yu)
            c = safe_dot(y_approx, xu, keepdims=True)
            denom = jnp.maximum(1.0 + c, 1e-12)
            trans_res = residual - (safe_dot(residual, y_approx + xu, keepdims=True) / denom) * (y_approx + xu)
            
            v_new = v + trans_res
            y_check = GeodesicKernels.exp_map(xu, v_new)
            err = jnp.max(safe_norm(y_check - yu, keepdims=True))
            return (v_new, err, i + 1)
            
        init_err = jnp.max(safe_norm(GeodesicKernels.exp_map(xu, v0) - yu, keepdims=True))
        v_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (v0, init_err, jnp.array(0)))
        
        return v_final

class CliffordRotors:
    @staticmethod
    @jax.jit
    def cholesky_qr3(W: jnp.ndarray) -> jnp.ndarray:
        """[SOTA] Cholesky-QR3 Iterado (El asesino de Gram-Schmidt)
        Ortogonalización masivamente paralela y asintóticamente estable (O(K^3) con K=2)."""
        eps = jnp.finfo(W.dtype).eps
        I = jnp.eye(W.shape[-1], dtype=W.dtype)
        Q = W
        for _ in range(3):
            # Producto interno Gramiano O(D * K^2)
            G = jnp.einsum('...ji,...jk->...ik', Q, Q) + eps * I
            # Factorización de Cholesky O(K^3), como K=2 es instantáneo en hardware
            L = jnp.linalg.cholesky(G)
            # Inversa triangular y actualización
            L_invT = jnp.linalg.inv(L.swapaxes(-1, -2))
            Q = jnp.einsum('...ij,...jk->...ik', Q, L_invT)
        return Q

    @staticmethod
    @jax.jit
    def apply_spherical_rotor(x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray, theta: jnp.ndarray = jnp.array(0.1)) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        U = U[..., None] if U.ndim == 1 else U
        V = V[..., None] if V.ndim == 1 else V
        W = jnp.concatenate([U, V], axis=-1)
        
        # FIX V74.1: Ruido proporcional a la norma, sin PRNGKey estático
        w_norm = safe_norm(W, axis=-2, keepdims=True)
        W_reg = W + 1e-6 * w_norm * jnp.ones_like(W)
        
        # FIX V77: Cholesky-QR3 en reemplazo absoluto de Gram-Schmidt (jnp.linalg.qr)
        Q = CliffordRotors.cholesky_qr3(W_reg)
        U_orth = Q[..., :U.shape[-1]]
        V_orth = Q[..., U.shape[-1]:]
        
        # FIX V74.1: Alinear dimensiones batch explícitamente para einsum
        batch_ndim = x.ndim - 1
        if batch_ndim > 0 and U_orth.ndim == 2:
            U_orth = jnp.expand_dims(U_orth, axis=tuple(range(batch_ndim)))
            V_orth = jnp.expand_dims(V_orth, axis=tuple(range(batch_ndim)))
            
        dot_U = jnp.einsum('...dr,...d->...r', U_orth, x)
        dot_V = jnp.einsum('...dr,...d->...r', V_orth, x)
        
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot_U = c * dot_U - s * dot_V
        rot_V = s * dot_U + c * dot_V
        
        delta_U = (rot_U - dot_U)[..., None, :] * U_orth
        delta_V = (rot_V - dot_V)[..., None, :] * V_orth
        delta = jnp.sum(delta_U, axis=-1) + jnp.sum(delta_V, axis=-1)
        
        result = x + delta
        return result / jnp.maximum(safe_norm(result, keepdims=True), eps)


# ==============================================================================
# 4. SWARM ARCHITECTURE (STRUCTURED ROUTING & INT8 PMTP)
# ==============================================================================
PMTP_MAGIC = b'PMTP'
PMTP_VERSION = 3 # V75.2 - INT8 Quantization + Epoch Clocks (Empirical SOTA)

# Header: Magic, Version, Shape_len, Dtype_code, Payload_bytes, Nonce, Agent_ID, Seq_Num, Epoch, Scale_Factor
PMTP_HEADER_FMT = "<4s B B B Q 32s 32s Q Q d " + "Q" * 8 
PMTP_HEADER_SIZE = struct.calcsize(PMTP_HEADER_FMT)

# Requerimiento estricto de clave
PMTP_NET_KEY = os.environ.get("POLYDIM_PMTP_KEY", "").encode()
if not PMTP_NET_KEY or len(PMTP_NET_KEY) != 32:
    if "pytest" not in sys.modules:
        warnings.warn("POLYDIM_PMTP_KEY no está definida o no es de 32 bytes. Se usará modo INSEGURO.")
        PMTP_NET_KEY = b'0' * 32

def pmtp_mac_chunks(sender_id: bytes, receiver_id: bytes, header: bytes, payload: memoryview) -> bytes:
    h = hmac.new(PMTP_NET_KEY, digestmod=hashlib.sha256)
    h.update(sender_id)
    h.update(receiver_id)
    h.update(header)
    h.update(payload)
    return h.digest()[:32]

class EpochClock:
    def __init__(self):
        self.epoch = 0
        self._lock = threading.Lock()
        
    def increment(self):
        with self._lock:
            self.epoch += 1
            return self.epoch
            
    def sync(self, remote_epoch: int):
        with self._lock:
            self.epoch = max(self.epoch, remote_epoch) + 1

class PMTPTokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
        
    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + self.rate * (now - self.last_update))
            self.last_update = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class PMTPAgentBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, agent_id: bytes = None, peers: list = None):
        self.host = host
        self.port = port
        self.agent_id = agent_id or os.urandom(32)
        self.peers = peers or [] # lista de tuples (host, port, peer_id)
        
        self.inbox = Queue(maxsize=10) # FIX: Límite de OOM
        self.token_bucket = PMTPTokenBucket(rate=1024*1024*10, capacity=1024*1024*50) # 10MB/s, burst 50MB
        self.epoch_clock = EpochClock() # FIX: Sustituye VectorClocks O(N)
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.port = self.server_socket.getsockname()[1]
        self.running = False
        
        self._max_concurrent = threading.Semaphore(16)
        
        self.seq_num = 0
        self.last_seen_seq = {} # peer_id -> seq

    def start_server(self):
        self.server_socket.listen(128)
        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop_server(self):
        self.running = False
        self.server_socket.close()

    def _listen_loop(self):
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                if self._max_concurrent.acquire(blocking=False):
                    threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
                else:
                    conn.close() # DoS protection
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                time.sleep(0.1)

    def _recv_exact(self, conn: socket.socket, num_bytes: int, timeout: float = 10.0) -> bytearray:
        buf = bytearray(num_bytes)
        view = memoryview(buf)
        pos = 0
        
        # FIX V74.1: Slowloris fix (Absolute deadline)
        start_time = time.monotonic()
        
        while pos < num_bytes:
            if time.monotonic() - start_time > timeout:
                raise TimeoutError("Deadline expirado")
            
            chunk = conn.recv_into(view[pos:])
            if chunk == 0:
                raise ConnectionError("Conexión cerrada prematuramente")
            pos += chunk
        return buf

    def _handle_connection(self, conn: socket.socket):
        try:
            conn.settimeout(10.0)
            
            header = self._recv_exact(conn, PMTP_HEADER_SIZE)
            fields = struct.unpack(PMTP_HEADER_FMT, header)
            magic, version, shape_len, dtype_code, payload_bytes, nonce, sender_id, seq_num, epoch, scale = fields[:10]
            shape = tuple(fields[10:10+shape_len])
            
            if magic != PMTP_MAGIC or version != PMTP_VERSION:
                conn.sendall(b'\\x02') # NACK
                return
                
            # Anti-replay
            if sender_id in self.last_seen_seq and seq_num <= self.last_seen_seq[sender_id]:
                conn.sendall(b'\\x02') # NACK repetido
                return
            
            # Backpressure
            if not self.token_bucket.consume(payload_bytes):
                conn.sendall(b'\\x02') # NACK por rate limit
                return
                
            payload = self._recv_exact(conn, payload_bytes)
            
            mac_expected = conn.recv(32)
            mac_calc = pmtp_mac_chunks(sender_id, self.agent_id, header, memoryview(payload))
            if not hmac.compare_digest(mac_expected, mac_calc):
                conn.sendall(b'\\x02')
                return
                
            self.last_seen_seq[sender_id] = seq_num
            self.epoch_clock.sync(epoch)
            
            # FIX: INT8 Dequantization
            quantized_arr = np.frombuffer(payload, dtype=np.int8).reshape(shape)
            arr = (quantized_arr.astype(np.float32) * scale).copy()
                
            try:
                self.inbox.put_nowait((sender_id, arr))
                conn.sendall(b'\\x01') # ACK positivo
            except:
                conn.sendall(b'\\x02') # NACK Queue Full
                
        except Exception as e:
            pass
        finally:
            conn.close()
            self._max_concurrent.release()

    def send_tensor(self, host: str, port: int, tensor: jnp.ndarray, receiver_id: bytes) -> bool:
        self.seq_num += 1
        epoch = self.epoch_clock.increment()
        
        # FIX: INT8 Quantization en origen (Reduce Payload a 25%)
        tensor_np = np.asarray(tensor, dtype=np.float32)
        abs_max = float(np.max(np.abs(tensor_np)))
        scale = 1.0 if abs_max == 0 else abs_max / 127.0
        
        quantized_np = np.clip(np.round(tensor_np / scale), -127, 127).astype(np.int8)
        payload = quantized_np.tobytes()
        payload_bytes = len(payload)
        shape = quantized_np.shape
        
        nonce = os.urandom(32)
        
        header = struct.pack(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, PMTP_VERSION, len(shape), 1, payload_bytes,
            nonce, self.agent_id, self.seq_num, epoch, scale,
            *(shape + (0,) * (8 - len(shape))) # Ajustar relleno
        )
        
        mac = pmtp_mac_chunks(self.agent_id, receiver_id, header, memoryview(payload))
        
        try:
            s = socket.create_connection((host, port), timeout=5.0)
            s.sendall(header)
            s.sendall(payload)
            s.sendall(mac)
            
            ack = s.recv(1)
            return ack == b'\\x01'
        except Exception:
            return False
        finally:
            try:
                s.close()
            except:
                pass

class XLAQuantizer:
    @staticmethod
    @jax.jit
    def quantize_int8_tree_reduce(tensor: jnp.ndarray):
        '''[SOTA] Reduccion Jerarquica O(log N) nativa en TPU/GPU'''
        abs_max = jnp.max(jnp.abs(tensor))
        safe_max = jnp.where(abs_max == 0, 1.0, abs_max)
        scale = safe_max / 127.0
        quantized = jnp.clip(jnp.round(tensor / scale), -127, 127).astype(jnp.int8)
        return quantized, scale

if __name__ == "__main__":
    print("POLYDIM V75 MONOLITH - Arquitectura Swarm (Epoch/INT8) SOTA Lista.")
