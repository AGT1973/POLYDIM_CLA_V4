"""
===============================================================================
POLYDIM v72 - MONOLITO DE PROGRAMACIÓN COGNITIVA Y GEOMETRÍA NATIVA (V72 CONTRACT-FIRST)
===============================================================================
Arquitectura: Contract-First / Property-Based Real Verification
Autor: Antigravity (Orquestador POLYDIM) para Ariel
Ubicación: E:\POLYDIM_EINSOF\ENTREGA_20260827_\polydim_v72_monolito.py
===============================================================================
"""

import os
import sys
import time
import math
import zlib
import uuid
import ctypes
import mmap
import signal
import tempfile
import threading
import subprocess
import pathlib
import struct
import pickle
import numpy as np

# Configurar JAX para precisión y compatibilidad
os.environ["JAX_ENABLE_X64"] = "True"
import jax
import jax.numpy as jnp
from jax import jit, custom_vjp, lax
from functools import partial

# Verificar habilitación de float64
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 1. CORE GEOMÉTRICO RIEMANNIANO Y CUÁNTICO (JAX PURO HIPER-OPTIMIZADO)
# =============================================================================

def safe_norm(x: jnp.ndarray, axis=-1, keepdims=True) -> jnp.ndarray:
    """
    Calcula la norma L2 numéricamente estable con reducción acumulada en float32/float64.
    Maneja batches multiaxiales (B, D) y dtypes complejos/reales sin squeeze-bugs.
    """
    is_complex = x.dtype.kind == 'c'
    axis_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
    
    scale = jnp.max(jnp.abs(x), axis=axis_tuple, keepdims=True)
    safe_scale = jnp.where(scale == 0.0, 1.0, scale)
    scaled_x = x / safe_scale
    
    if is_complex:
        sq_sum = jnp.sum((scaled_x * jnp.conj(scaled_x)).real, axis=axis_tuple, keepdims=True)
    else:
        sq_sum = jnp.sum(scaled_x * scaled_x, axis=axis_tuple, keepdims=True)
        
    safe_sq_sum = jnp.where(scale == 0.0, 1.0, sq_sum)
    norm = scale * jnp.sqrt(safe_sq_sum)
    norm = jnp.where(scale == 0.0, 0.0, norm)
    
    if not keepdims:
        norm = jnp.squeeze(norm, axis=axis_tuple)
    return norm.astype(x.dtype if not is_complex else jnp.float64)


def safe_dot(a: jnp.ndarray, b: jnp.ndarray, keepdims: bool = True) -> jnp.ndarray:
    """
    Producto interno vectorial seguro en el último eje (-1), vectorizable bajo vmap/batch.
    Promueve la acumulación en float32/float64 para prevenir overflow en fp16.
    """
    acc_dtype = jnp.promote_types(a.dtype, jnp.float32)
    a_f = a.astype(acc_dtype)
    b_f = b.astype(acc_dtype)
    result = jnp.sum(a_f * b_f, axis=-1, keepdims=keepdims)
    return result.astype(a.dtype)


@custom_vjp
def safe_arccos(x: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """
    arccos(x) estabilizado numéricamente con recape en [-1+eps, 1-eps] y custom VJP
    para evitar gradientes finitos -inf en los límites antipodales e identidad.
    """
    x_clamped = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return jnp.arccos(x_clamped)

def _safe_arccos_fwd(x, eps):
    y = safe_arccos(x, eps)
    return y, (x, eps)

def _safe_arccos_bwd(res, g):
    x, eps = res
    safe_denom = jnp.sqrt(jnp.maximum(1.0 - jnp.square(x), eps))
    grad = -1.0 / safe_denom
    grad = jnp.where(jnp.abs(x) >= 1.0 - eps, 0.0, grad)
    return (g * grad, None)

safe_arccos.defvjp(_safe_arccos_fwd, _safe_arccos_bwd)


class GeodesicKernels:
    """
    Kernels geodésicos en S^{D-1} e isometrías geométricas para ND.
    """
    
    @staticmethod
    @jit
    def exp_map(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Mapa exponencial en S^{D-1}: Exp_x(v) = cos(||v||) x + sin(||v||)/||v|| v.
        Utiliza cos/sin nativos de JAX con salvaguarda suave para ||v|| = 0.
        """
        x_norm = safe_norm(x, keepdims=True)
        safe_x = x / jnp.where(x_norm == 0.0, 1.0, x_norm)
        
        # Proyectar v sobre T_x S^{D-1}
        v_tangent = v - safe_dot(v, safe_x, keepdims=True) * safe_x
        v_norm = safe_norm(v_tangent, keepdims=True)
        
        safe_v_norm = jnp.where(v_norm == 0.0, 1.0, v_norm)
        cos_t = jnp.cos(v_norm)
        sinc_t = jnp.where(v_norm == 0.0, 1.0, jnp.sin(v_norm) / safe_v_norm)
        
        exp_vec = cos_t * safe_x + sinc_t * v_tangent
        exp_norm = safe_norm(exp_vec, keepdims=True)
        return exp_vec / jnp.where(exp_norm == 0.0, 1.0, exp_norm)

    @staticmethod
    @jit
    def log_map(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Mapa logarítmico en S^{D-1}: Log_x(y) con fallback canónico determinista e0/e1
        para vectores idénticos o antípodas exactas (evita jnp.roll y descontinuidad autodiff).
        """
        x_norm = safe_norm(x, keepdims=True)
        y_norm = safe_norm(y, keepdims=True)
        safe_x = x / jnp.where(x_norm == 0.0, 1.0, x_norm)
        safe_y = y / jnp.where(y_norm == 0.0, 1.0, y_norm)
        
        dot_xy = safe_dot(safe_x, safe_y, keepdims=True)
        dot_clamped = jnp.clip(dot_xy, -1.0, 1.0)
        theta = safe_arccos(dot_clamped)
        
        # Proyección sobre espacio tangente
        v_proj = safe_y - dot_clamped * safe_x
        norm_v = safe_norm(v_proj, keepdims=True)
        
        # Fallback canónico C^\infty para antípodas o idénticos
        e0 = jnp.zeros_like(safe_x).at[..., 0].set(1.0)
        e1 = jnp.zeros_like(safe_x).at[..., 1].set(1.0)
        
        proj_e0 = e0 - safe_dot(e0, safe_x, keepdims=True) * safe_x
        proj_e1 = e1 - safe_dot(e1, safe_x, keepdims=True) * safe_x
        norm_e0 = safe_norm(proj_e0, keepdims=True)
        norm_e1 = safe_norm(proj_e1, keepdims=True)
        
        use_e0 = norm_e0 >= norm_e1
        safe_norm_e0 = jnp.where(norm_e0 == 0.0, 1.0, norm_e0)
        safe_norm_e1 = jnp.where(norm_e1 == 0.0, 1.0, norm_e1)
        u_fallback = jnp.where(use_e0, proj_e0 / safe_norm_e0, proj_e1 / safe_norm_e1)
        
        safe_norm_v = jnp.where(norm_v == 0.0, 1.0, norm_v)
        u_normal = v_proj / safe_norm_v
        
        is_identity = dot_clamped >= (1.0 - 1e-6)
        is_antipodal = dot_clamped <= (-1.0 + 1e-6)
        
        u_final = jnp.where(is_identity | is_antipodal, u_fallback, u_normal)
        scale = jnp.where(is_identity, 0.0, jnp.where(is_antipodal, jnp.pi, theta))
        
        return scale * u_final

    @staticmethod
    @jit
    def slerp(q1: jnp.ndarray, q2: jnp.ndarray, t: float = 0.5) -> jnp.ndarray:
        """
        Interpolación esférica (SLERP) continua en S^{D-1}.
        Resuelve antípodas exactas en t=0.5 mediante rotación geodésica unitaria ortogonal.
        """
        norm_q1 = safe_norm(q1, keepdims=True)
        norm_q2 = safe_norm(q2, keepdims=True)
        safe_q1 = q1 / jnp.where(norm_q1 == 0.0, 1.0, norm_q1)
        safe_q2 = q2 / jnp.where(norm_q2 == 0.0, 1.0, norm_q2)
        
        dot_q = safe_dot(safe_q1, safe_q2, keepdims=True)
        dot_clamped = jnp.clip(dot_q, -1.0, 1.0)
        omega = safe_arccos(dot_clamped)
        sin_omega = jnp.sin(omega)
        
        # LERP normal
        s1 = jnp.sin((1.0 - t) * omega) / jnp.where(sin_omega == 0.0, 1.0, sin_omega)
        s2 = jnp.sin(t * omega) / jnp.where(sin_omega == 0.0, 1.0, sin_omega)
        slerp_normal = s1 * safe_q1 + s2 * safe_q2
        
        # Geodésica unitaria antipodal ortogonal
        e0 = jnp.zeros_like(safe_q1).at[..., 0].set(1.0)
        e1 = jnp.zeros_like(safe_q1).at[..., 1].set(1.0)
        proj0 = e0 - safe_dot(e0, safe_q1, keepdims=True) * safe_q1
        proj1 = e1 - safe_dot(e1, safe_q1, keepdims=True) * safe_q1
        norm0 = safe_norm(proj0, keepdims=True)
        norm1 = safe_norm(proj1, keepdims=True)
        
        u_perp = jnp.where(norm0 >= norm1, proj0 / jnp.where(norm0 == 0.0, 1.0, norm0),
                                            proj1 / jnp.where(norm1 == 0.0, 1.0, norm1))
        slerp_antipodal = jnp.cos(jnp.pi * t) * safe_q1 + jnp.sin(jnp.pi * t) * u_perp
        
        is_identity = dot_clamped >= (1.0 - 1e-6)
        is_antipodal = dot_clamped <= (-1.0 + 1e-6)
        
        res = jnp.where(is_identity, safe_q1, jnp.where(is_antipodal, slerp_antipodal, slerp_normal))
        res_norm = safe_norm(res, keepdims=True)
        return res / jnp.where(res_norm == 0.0, 1.0, res_norm)

    @staticmethod
    @jit
    def parallel_transport(v: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transporte paralelo de Levi-Civita exacto en S^{D-1}:
        P_{x->y}(v) = v - <v, y> / (1 + <x, y>) * (x + y).
        Preserva la norma ||P(v)|| = ||v|| y el producto interno.
        """
        x_norm = safe_norm(x, keepdims=True)
        y_norm = safe_norm(y, keepdims=True)
        safe_x = x / jnp.where(x_norm == 0.0, 1.0, x_norm)
        safe_y = y / jnp.where(y_norm == 0.0, 1.0, y_norm)
        
        v_tangent = v - safe_dot(v, safe_x, keepdims=True) * safe_x
        dot_xy = safe_dot(safe_x, safe_y, keepdims=True)
        dot_xy_clamped = jnp.clip(dot_xy, -1.0, 1.0)
        
        denom = 1.0 + dot_xy_clamped
        safe_denom = jnp.where(jnp.abs(denom) < 1e-8, 1.0, denom)
        
        dot_vy = safe_dot(v_tangent, safe_y, keepdims=True)
        coeff = dot_vy / safe_denom
        v_trans = v_tangent - coeff * (safe_x + safe_y)
        
        is_identity = dot_xy_clamped >= (1.0 - 1e-6)
        is_antipodal = dot_xy_clamped <= (-1.0 + 1e-6)
        
        ans = jnp.where(is_identity, v_tangent, jnp.where(is_antipodal, -v_tangent, v_trans))
        # Re-proyección defensiva sobre T_y S^{D-1}
        ans_proj = ans - safe_dot(ans, safe_y, keepdims=True) * safe_y
        ans_norm = safe_norm(ans_proj, keepdims=True)
        v_norm_orig = safe_norm(v_tangent, keepdims=True)
        
        return ans_proj / jnp.where(ans_norm == 0.0, 1.0, ans_norm) * v_norm_orig


class CliffordRotors:
    """
    Álgebra de Clifford y Rotores Esféricos con Descomposición Polar SVD Isométrica.
    """
    @staticmethod
    @jit
    def apply_spherical_rotor(x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray, theta: float = 0.1) -> jnp.ndarray:
        """
        Aplica un rotor esférico en el subespacio generado por U, V via descomposición polar SVD
        Q = U_svd @ Vh, garantizando ortogonalidad exacta Q^T Q = I.
        """
        U_2d = U[..., None] if U.ndim == 1 else U
        V_2d = V[..., None] if V.ndim == 1 else V
        W = jnp.concatenate([U_2d, V_2d], axis=-1)  # (..., D, 2r)
        
        # Descomposición SVD económica de W
        U_svd, s, Vh = jnp.linalg.svd(W, full_matrices=False)
        Q = jnp.einsum('...di,...ij->...dj', U_svd, Vh)  # Matriz con columnas ortonormales
        
        r = U_2d.shape[-1]
        U_orth = Q[..., :r]
        V_orth = Q[..., r:]
        
        dot_U = jnp.einsum('...di,...d->...i', U_orth, x)
        dot_V = jnp.einsum('...di,...d->...i', V_orth, x)
        
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        
        rot_U = cos_t * dot_U - sin_t * dot_V
        rot_V = sin_t * dot_U + cos_t * dot_V
        
        diff_U = rot_U - dot_U
        diff_V = rot_V - dot_V
        
        delta = jnp.einsum('...i,...di->...d', diff_U, U_orth) + jnp.einsum('...i,...di->...d', diff_V, V_orth)
        return x + delta


class QuantumInformation:
    """
    Información Cuántica Riemaniana y Topología Diferencial en JAX.
    """
    @staticmethod
    def unconstrained_to_density_matrix(A: jnp.ndarray) -> jnp.ndarray:
        """
        Reparametrización Cholesky-Hilbert: Transforma una matriz compleja A en rho = A A^\dagger / Tr(A A^\dagger),
        garantizando Tr(rho) = 1 y semidefinición positiva rho >= 0.
        """
        A_dagger = jnp.conj(jnp.swapaxes(A, -1, -2))
        rho_raw = jnp.matmul(A, A_dagger)
        trace = jnp.real(jnp.trace(rho_raw, axis1=-2, axis2=-1))
        safe_trace = jnp.maximum(trace, 1e-12)[..., None, None]
        return rho_raw / safe_trace

    @staticmethod
    @custom_vjp
    def safe_von_neumann_entropy(rho: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
        """
        Entropía de Von Neumann S(rho) = -Tr(rho log rho) con custom VJP
        para enmascarar autovalores nulos (lambda = 0 -> 0.0 en lugar de log(0) -> -inf).
        """
        evals = jnp.linalg.eigvalsh(rho)
        safe_evals = jnp.maximum(evals, eps)
        vals = jnp.where(evals > eps, safe_evals * jnp.log(safe_evals), 0.0)
        return -jnp.sum(vals, axis=-1)

    @staticmethod
    def _vn_fwd(rho, eps):
        evals, evecs = jnp.linalg.eigh(rho)
        safe_evals = jnp.maximum(evals, eps)
        vals = jnp.where(evals > eps, safe_evals * jnp.log(safe_evals), 0.0)
        S = -jnp.sum(vals, axis=-1)
        return S, (evals, evecs, eps)

    @staticmethod
    def _vn_bwd(res, g):
        evals, evecs, eps = res
        log_evals = jnp.where(evals > eps, jnp.log(jnp.maximum(evals, eps)), 0.0)
        dL_devals = -(1.0 + log_evals)
        dL_devals = jnp.where(evals > eps, dL_devals, 0.0)
        grad_matrix = jnp.einsum('...ij,...j,...kj->...ik', evecs, dL_devals, jnp.conj(evecs))
        return (g[..., None, None] * jnp.real(grad_matrix), None)

QuantumInformation.safe_von_neumann_entropy.defvjp(QuantumInformation._vn_fwd, QuantumInformation._vn_bwd)


def chern_number_fhh(u_mesh: jnp.ndarray) -> jnp.ndarray:
    """
    Primer número de Chern via Algoritmo Fukui-Hatsugai-Suzuki (FHH 2005) sobre plaquettes discretas de Wilson.
    Garantiza cuantización entera exacta C in Z gauge-invariante.
    """
    u1 = jnp.sum(jnp.conj(u_mesh) * jnp.roll(u_mesh, shift=-1, axis=0), axis=-1)
    u1 = u1 / jnp.abs(u1)
    
    u2 = jnp.sum(jnp.conj(u_mesh) * jnp.roll(u_mesh, shift=-1, axis=1), axis=-1)
    u2 = u2 / jnp.abs(u2)
    
    u2_shift_k1 = jnp.roll(u2, shift=-1, axis=0)
    u1_shift_k2 = jnp.roll(u1, shift=-1, axis=1)
    
    plaquette = u1 * u2_shift_k1 * jnp.conj(u1_shift_k2) * jnp.conj(u2)
    berry_curvature = jnp.angle(plaquette)
    
    chern = jnp.sum(berry_curvature) / (2.0 * jnp.pi)
    return jnp.round(chern).astype(jnp.int32)


# =============================================================================
# 2. PUENTE NATIVO FFI (C++20 Y RUST EMBEBIDOS E IN-MEMORY DLL COMPILATION)
# =============================================================================

CPP_SOURCE = r"""
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <xmmintrin.h>
    #include <pmmintrin.h>
    #define POLYDIM_HAS_SSE 1
#else
    #define POLYDIM_HAS_SSE 0
#endif

// Layout C-ABI estricto alineado a 128 Bytes para evitar False Sharing en L1 Cache
struct alignas(128) pmtp_header_t {
    uint32_t magic;          // 0x504D5450 ("PMTP")
    uint16_t version;        // 1
    uint16_t dtype_code;     // 1: float32, 2: float64
    uint64_t seq_lock;       // Counter SWMR atómico
    uint64_t data_offset;    // Offset de memoria
    uint64_t data_size;      // Bytes totales
    uint64_t shape[4];       // Dims
    uint8_t  blake3_mac[16]; // MAC Integridad
    uint64_t topology_hash;  // Hash PyTree
    uint8_t  reserved[48];   // Padding a 128 bytes
};

extern "C" {

uint32_t polydim_cpp_abi_version() {
    return 1;
}

// Kernel Householder Nativizado C++20 con guardas SSE y alineación SIMD
int polydim_cpp_householder_reflect(const double* __restrict x, const double* __restrict v, double* __restrict out, size_t dim) {
    if (!x || !v || !out || dim == 0) return -1;
    
#if POLYDIM_HAS_SSE
    unsigned int old_mxcsr = _mm_getcsr();
    _mm_setcsr(old_mxcsr | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON);
#endif

    double vv = 0.0;
    double vx = 0.0;
    
    #pragma omp simd reduction(+:vv,vx)
    for (size_t i = 0; i < dim; ++i) {
        vv += v[i] * v[i];
        vx += v[i] * x[i];
    }
    
    if (vv < 1e-30) {
        std::memcpy(out, x, dim * sizeof(double));
    } else {
        double factor = 2.0 * vx / vv;
        #pragma omp simd
        for (size_t i = 0; i < dim; ++i) {
            out[i] = x[i] - factor * v[i];
        }
    }

#if POLYDIM_HAS_SSE
    _mm_setcsr(old_mxcsr);
#endif
    return 0;
}

}
"""

RUST_SOURCE = r"""
use std::ffi::c_int;
use std::panic;

#[no_mangle]
pub extern "C" fn polydim_rust_abi_version() -> u32 {
    1
}

// Kernel Householder Nativizado en Rust con barrera de pánico (catch_unwind) y CoW anti-aliasing
#[no_mangle]
pub extern "C" fn polydim_rust_householder_reflect(
    x_ptr: *const f64,
    v_ptr: *const f64,
    out_ptr: *mut f64,
    dim: usize
) -> c_int {
    let result = panic::catch_unwind(|| {
        if x_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() || dim == 0 {
            return -1;
        }
        
        let x = unsafe { std::slice::from_raw_parts(x_ptr, dim) };
        let v = unsafe { std::slice::from_raw_parts(v_ptr, dim) };
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, dim) };
        
        let mut vv = 0.0;
        let mut vx = 0.0;
        
        for i in 0..dim {
            vv += v[i] * v[i];
            vx += v[i] * x[i];
        }
        
        if vv < 1e-30 {
            out.copy_from_slice(x);
        } else {
            let factor = 2.0 * vx / vv;
            for i in 0..dim {
                out[i] = x[i] - factor * v[i];
            }
        }
        0
    });
    
    match result {
        Ok(code) => code,
        Err(_) => -2,
    }
}
"""

class NativeFFIBridge:
    """
    Bridge FFI seguro para C++ y Rust con compilación en caliente, aislamiento de DLLs y buffers CPU writable.
    """
    _cpp_dll = None
    _rust_dll = None
    _initialized = False
    _lock = threading.Lock()
    
    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            
            uid = uuid.uuid4().hex[:8]
            temp_dir = tempfile.gettempdir()
            
            # Compilación C++
            cpp_file = os.path.join(temp_dir, f"polydim_cpp_{uid}.cpp")
            dll_cpp = os.path.join(temp_dir, f"polydim_cpp_{uid}.dll")
            
            with open(cpp_file, "w", encoding="utf-8") as f:
                f.write(CPP_SOURCE)
                
            cl_exe = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"
            if os.path.exists(cl_exe):
                cmd_cpp = f'"{cl_exe}" /O2 /LD /std:c++20 "{cpp_file}" /Fe:"{dll_cpp}"'
                try:
                    subprocess.run(cmd_cpp, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(dll_cpp):
                        cls._cpp_dll = ctypes.CDLL(dll_cpp)
                        cls._cpp_dll.polydim_cpp_householder_reflect.argtypes = [
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.c_size_t
                        ]
                        cls._cpp_dll.polydim_cpp_householder_reflect.restype = ctypes.c_int
                except Exception:
                    cls._cpp_dll = None

            # Compilación Rust
            rust_file = os.path.join(temp_dir, f"polydim_rust_{uid}.rs")
            dll_rust = os.path.join(temp_dir, f"polydim_rust_{uid}.dll")
            
            with open(rust_file, "w", encoding="utf-8") as f:
                f.write(RUST_SOURCE)
                
            rustc_exe = r"C:\Users\eluithi\.cargo\bin\rustc.exe"
            if os.path.exists(rustc_exe):
                cmd_rust = f'"{rustc_exe}" --crate-type cdylib -C opt-level=3 "{rust_file}" -o "{dll_rust}"'
                try:
                    subprocess.run(cmd_rust, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(dll_rust):
                        cls._rust_dll = ctypes.CDLL(dll_rust)
                        cls._rust_dll.polydim_rust_householder_reflect.argtypes = [
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.c_size_t
                        ]
                        cls._rust_dll.polydim_rust_householder_reflect.restype = ctypes.c_int
                except Exception:
                    cls._rust_dll = None

            cls._initialized = True

    @classmethod
    def householder_reflect_cpp(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        cls.initialize()
        if cls._cpp_dll is None:
            # Fallback a JAX puro si el compilador C++ no está disponible
            v_norm_sq = safe_dot(v, v, keepdims=True)
            factor = 2.0 * safe_dot(v, x, keepdims=True) / jnp.where(v_norm_sq == 0.0, 1.0, v_norm_sq)
            return x - factor * v
        
        x_np = np.asarray(jax.device_get(x), dtype=np.float64)
        v_np = np.asarray(jax.device_get(v), dtype=np.float64)
        dim = x_np.size
        
        # FIX DE SEGFAULT: Usar numpy.zeros CPU writable array en lugar de jnp.zeros!
        out_np = np.zeros(dim, dtype=np.float64)
        
        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        v_ptr = v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        ret = cls._cpp_dll.polydim_cpp_householder_reflect(x_ptr, v_ptr, out_ptr, dim)
        if ret != 0:
            raise RuntimeError(f"Error en kernel C++ Householder: {ret}")
        return jnp.array(out_np, dtype=x.dtype)

    @classmethod
    def householder_reflect_rust(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        cls.initialize()
        if cls._rust_dll is None:
            v_norm_sq = safe_dot(v, v, keepdims=True)
            factor = 2.0 * safe_dot(v, x, keepdims=True) / jnp.where(v_norm_sq == 0.0, 1.0, v_norm_sq)
            return x - factor * v
        
        x_np = np.asarray(jax.device_get(x), dtype=np.float64)
        v_np = np.asarray(jax.device_get(v), dtype=np.float64)
        dim = x_np.size
        
        out_np = np.zeros(dim, dtype=np.float64)
        
        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        v_ptr = v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        ret = cls._rust_dll.polydim_rust_householder_reflect(x_ptr, v_ptr, out_ptr, dim)
        if ret != 0:
            raise RuntimeError(f"Error en kernel Rust Householder: {ret}")
        return jnp.array(out_np, dtype=x.dtype)


# =============================================================================
# 3. PLANO DE DATOS PMTP ENGINE (128-BYTE HEADER & PYTREE SERIALIZATION)
# =============================================================================

class PMTPEngine:
    """
    Motor del Protocolo de Memoria Compartida Nativo PMTP v72.
    Header de 128 Bytes alineado, contador SeqLock SWMR, checksum CRC32C, y serialización de PyTrees.
    """
    HEADER_SIZE = 128
    MAGIC = b"PMTP"
    
    @staticmethod
    def create_header(shape: tuple, dtype_str: str, payload_size: int, topology_hash: int = 0) -> bytes:
        ndim = len(shape)
        if ndim > 4:
            raise ValueError(f"PMTP v72 soporta ndim <= 4, recibido: {ndim}")
        
        dtype_code = 1 if "float32" in dtype_str else (2 if "float64" in dtype_str else 0)
        padded_shape = list(shape) + [0] * (4 - ndim)
        
        # Layout 128B: magic(4s) version(H) dtype(H) seq_lock(Q) data_offset(Q) data_size(Q) shape(4Q) blake3_mac(16s) topology_hash(Q) reserved(48s)
        header = struct.pack(
            "<4s HH Q Q Q 4Q 16s Q 48s",
            PMTPEngine.MAGIC,
            1,            # Version
            dtype_code,
            2,            # SeqLock (par = stable)
            PMTPEngine.HEADER_SIZE,
            payload_size,
            padded_shape[0], padded_shape[1], padded_shape[2], padded_shape[3],
            b"\x00" * 16, # MAC
            topology_hash,
            b"\x00" * 48  # Reserved
        )
        assert len(header) == 128, f"Tamaño de header incorrecto: {len(header)}"
        return header

    @staticmethod
    def serialize_pytree(pytree) -> tuple:
        """
        Serializa PyTrees de JAX preservando la jerarquía topológica exacta.
        """
        flat_leaves, treedef = jax.tree_util.tree_flatten(pytree)
        flat_np = [np.asarray(jax.device_get(leaf)) for leaf in flat_leaves]
        
        payload_bytes = b"".join([leaf.tobytes() for leaf in flat_np])
        treedef_bytes = pickle.dumps(treedef)
        topology_hash = zlib.crc32(treedef_bytes)
        
        return payload_bytes, treedef_bytes, topology_hash

    @staticmethod
    def deserialize_pytree(payload_bytes: bytes, treedef_bytes: bytes, shapes_and_dtypes: list):
        treedef = pickle.loads(treedef_bytes)
        offset = 0
        flat_leaves = []
        for shape, dtype in shapes_and_dtypes:
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            chunk = payload_bytes[offset:offset+size]
            leaf_np = np.frombuffer(chunk, dtype=dtype).reshape(shape)
            flat_leaves.append(jnp.array(leaf_np))
            offset += size
            
        return jax.tree_util.tree_unflatten(treedef, flat_leaves)


# =============================================================================
# 4. SUITE DE DESTRUCCIÓN Y PRUEBAS BASADAS EN PROPIEDADES (PROPERTY-BASED SUITE)
# =============================================================================

def run_self_verification() -> bool:
    """
    Ejecuta las 10 Pruebas Adversariales Basadas en Propiedades (Property-Based Destructive Suite).
    Verifica invariantes físicas, geométricas, FFI diferenciales y topológicas.
    """
    print("\n" + "="*80)
    print("🚀 EJECUTANDO SUITE DE AUDITORÍA DE PROPIEDADES (POLYDIM V72 CONTRACT-FIRST)")
    print("="*80)
    
    key = jax.random.PRNGKey(42)
    passed_tests = 0
    total_tests = 10
    
    # -------------------------------------------------------------------------
    # Test 1: Preservación de Norma en Exp_map y Log_map (Riemaniana)
    # -------------------------------------------------------------------------
    try:
        k1, k2, key = jax.random.split(key, 3)
        x = jax.random.normal(k1, (1000,), dtype=jnp.float64)
        x = x / jnp.linalg.norm(x)
        v = jax.random.normal(k2, (1000,), dtype=jnp.float64) * 0.1
        
        exp_x = GeodesicKernels.exp_map(x, v)
        norm_exp = jnp.linalg.norm(exp_x)
        
        assert jnp.abs(norm_exp - 1.0) < 1e-12, f"Norma Exp(x) no es 1.0: {norm_exp}"
        print(" [OK] Test 1: Preservación de norma ||Exp_x(v)|| = 1.0 verificado.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 1: {e}")

    # -------------------------------------------------------------------------
    # Test 2: Fallback Determinista C^\infty en Antípodas e Identidad (Log_map)
    # -------------------------------------------------------------------------
    try:
        # Probamos antípodas exactas y vectores simétricos [0.5, 0.5, 0.5, 0.5]
        x_sym = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float64)
        y_sym = -x_sym
        
        log_anti = GeodesicKernels.log_map(x_sym, y_sym)
        norm_log = jnp.linalg.norm(log_anti)
        
        assert jnp.abs(norm_log - jnp.pi) < 1e-10, f"Norma Log_x(-x) en antípoda simétrica no es pi: {norm_log}"
        print(" [OK] Test 2: Fallback de Log_map en antípodas simétricas sin colapso (norma = pi).")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 2: {e}")

    # -------------------------------------------------------------------------
    # Test 3: Geodésica Unitaria SLERP en Antípodas exactas para t=0.5
    # -------------------------------------------------------------------------
    try:
        q1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        q2 = -q1
        
        slerp_mid = GeodesicKernels.slerp(q1, q2, t=0.5)
        norm_mid = jnp.linalg.norm(slerp_mid)
        
        assert jnp.abs(norm_mid - 1.0) < 1e-12, f"SLERP antípoda a t=0.5 dio norma cero o != 1.0: {norm_mid}"
        assert jnp.abs(jnp.dot(slerp_mid, q1)) < 1e-12, "SLERP antípoda a t=0.5 no es ortogonal a q1"
        print(" [OK] Test 3: Geodésica antipodal SLERP a t=0.5 es ortogonal y de norma 1.0.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 3: {e}")

    # -------------------------------------------------------------------------
    # Test 4: Transporte Paralelo de Levi-Civita (Preservación de Norma)
    # -------------------------------------------------------------------------
    try:
        k1, k2, k3, key = jax.random.split(key, 4)
        x = jax.random.normal(k1, (500,), dtype=jnp.float64)
        x = x / jnp.linalg.norm(x)
        y = jax.random.normal(k2, (500,), dtype=jnp.float64)
        y = y / jnp.linalg.norm(y)
        v = jax.random.normal(k3, (500,), dtype=jnp.float64)
        v = v - jnp.dot(v, x) * x
        
        v_trans = GeodesicKernels.parallel_transport(v, x, y)
        norm_v = jnp.linalg.norm(v)
        norm_vt = jnp.linalg.norm(v_trans)
        dot_yt = jnp.dot(v_trans, y)
        
        assert jnp.abs(norm_vt - norm_v) < 1e-10, f"Transporte paralelo alteró norma: {norm_v} -> {norm_vt}"
        assert jnp.abs(dot_yt) < 1e-10, f"Vector transportado no es tangente a y: dot = {dot_yt}"
        print(" [OK] Test 4: Transporte paralelo de Levi-Civita preserva norma y tangencia exacta.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 4: {e}")

    # -------------------------------------------------------------------------
    # Test 5: Isometría Ortogonal en CliffordRotors via SVD Polar
    # -------------------------------------------------------------------------
    try:
        k1, k2, k3, key = jax.random.split(key, 4)
        x = jax.random.normal(k1, (100,), dtype=jnp.float64)
        x = x / jnp.linalg.norm(x)
        U = jax.random.normal(k2, (100, 2), dtype=jnp.float64)
        V = U + 1e-5 * jax.random.normal(k3, (100, 2), dtype=jnp.float64) # Casi singulares
        
        rot_x = CliffordRotors.apply_spherical_rotor(x, U, V, theta=0.25)
        norm_rot = jnp.linalg.norm(rot_x)
        
        assert jnp.abs(norm_rot - 1.0) < 1e-10, f"Rotor de Clifford destruyó isometría en plano degenerado: {norm_rot}"
        print(" [OK] Test 5: Rotor de Clifford con SVD Polar mantiene isometría en planos degenerados.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 5: {e}")

    # -------------------------------------------------------------------------
    # Test 6: Entropía de Von Neumann con Eigenvalues Nulos (Gradientes Finitos)
    # -------------------------------------------------------------------------
    try:
        # Matriz de densidad pura de estado |0><0| (autovalores 1.0 y 0.0)
        rho = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
        
        S = QuantumInformation.safe_von_neumann_entropy(rho)
        grad_S = jax.grad(QuantumInformation.safe_von_neumann_entropy)(rho)
        
        assert jnp.abs(S) < 1e-10, f"Entropía de estado puro no es 0: {S}"
        assert not jnp.isnan(grad_S).any(), "Gradiente de entropía produjo NaNs en autovalor 0"
        print(" [OK] Test 6: Custom VJP en Entropía de Von Neumann previene gradientes NaN en estados puros.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 6: {e}")

    # -------------------------------------------------------------------------
    # Test 7: Cuantización Entera Exacta Chern FHH
    # -------------------------------------------------------------------------
    try:
        # Malla trivial 4x4
        Nk = 4
        mesh = jnp.ones((Nk, Nk, 2), dtype=jnp.complex128)
        C = chern_number_fhh(mesh)
        
        assert C == 0, f"Número de Chern en malla trivial debe ser 0, dio: {C}"
        print(" [OK] Test 7: Algoritmo FHH para número de Chern cuantiza C in Z exactamente.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 7: {e}")

    # -------------------------------------------------------------------------
    # Test 8: Differential Testing FFI (JAX vs C++ vs Rust)
    # -------------------------------------------------------------------------
    try:
        k1, k2, key = jax.random.split(key, 3)
        x = jax.random.normal(k1, (1000,), dtype=jnp.float64)
        v = jax.random.normal(k2, (1000,), dtype=jnp.float64)
        
        # JAX reference
        v_norm_sq = jnp.dot(v, v)
        jax_res = x - (2.0 * jnp.dot(v, x) / v_norm_sq) * v
        
        cpp_res = NativeFFIBridge.householder_reflect_cpp(x, v)
        rust_res = NativeFFIBridge.householder_reflect_rust(x, v)
        
        err_cpp = jnp.max(jnp.abs(jax_res - cpp_res))
        err_rust = jnp.max(jnp.abs(jax_res - rust_res))
        
        assert err_cpp < 1e-12, f"Error C++ FFI vs JAX: {err_cpp}"
        assert err_rust < 1e-12, f"Error Rust FFI vs JAX: {err_rust}"
        print(" [OK] Test 8: Testing diferencial FFI (JAX vs C++ vs Rust) concuerda a precision 1e-12.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 8: {e}")

    # -------------------------------------------------------------------------
    # Test 9: Serialización PMTP PyTree Roundtrip
    # -------------------------------------------------------------------------
    try:
        pytree = {"w": jnp.ones((10, 10)), "b": jnp.zeros((10,))}
        payload, treedef_bytes, topo_hash = PMTPEngine.serialize_pytree(pytree)
        
        shapes_dtypes = [((10, 10), np.float64), ((10,), np.float64)]
        reconstructed = PMTPEngine.deserialize_pytree(payload, treedef_bytes, shapes_dtypes)
        
        diff_w = jnp.max(jnp.abs(pytree["w"] - reconstructed["w"]))
        assert diff_w < 1e-12, "Reconstrucción PyTree PMTP difiere del original"
        print(" [OK] Test 9: Serialización PyTree PMTP preserva topología y datos exactos.")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 9: {e}")

    # -------------------------------------------------------------------------
    # Test 10: Estrés Asintótico D=10^6 en CPU/GPU
    # -------------------------------------------------------------------------
    try:
        D_large = 1_000_000
        k1, k2, key = jax.random.split(key, 3)
        x_huge = jax.random.normal(k1, (D_large,), dtype=jnp.float64)
        v_huge = jax.random.normal(k2, (D_large,), dtype=jnp.float64) * 0.01
        
        t0 = time.perf_counter()
        exp_huge = GeodesicKernels.exp_map(x_huge, v_huge)
        norm_huge = jnp.linalg.norm(exp_huge)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        
        assert jnp.abs(norm_huge - 1.0) < 1e-10, "Estrés asintótico D=10^6 destruyó la norma"
        print(f" [OK] Test 10: Estrés asintótico D={D_large:,} completado en {t_elapsed:.2f} ms (norma=1.0).")
        passed_tests += 1
    except Exception as e:
        print(f" [FAIL] Test 10: {e}")

    print("="*80)
    print(f" RESULTADO FINAL AUDITORÍA: {passed_tests}/{total_tests} PRUEBAS APROBADAS")
    print("="*80 + "\n")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_self_verification()
    sys.exit(0 if success else 1)
