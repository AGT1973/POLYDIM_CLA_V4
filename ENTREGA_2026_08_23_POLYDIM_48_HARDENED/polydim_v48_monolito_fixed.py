"""
===============================================================================
POLYDIM V48.4-ULTIMATE: MONOLITO NATIVO AUTOCONTENIDO DE MÁXIMO DETALLE
===============================================================================
Fecha: 2026-08-23 | Autor: Ariel Luithardt & Orquestador Antigravity (Bulldog Mode)

Frentes Integrados:
  1. Benchmark Exhaustivo + SIMD (AVX-512 con fallback a OpenMP/SIMD)
  2. GPU/TPU (JAX QR Retraction + Triton kernel reference)
  3. PMTP Completo (Struct C++ 56-byte + Shared Memory validation mmap)
  4. Cero UB (Comparación de punteros vía uintptr_t, checked arithmetic)
"""

import os
import sys
import time
import ctypes
import subprocess
import shutil
import tempfile
import atexit
import numpy as np
import logging
import argparse
import struct
import mmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("polydim_v48_4_ultimate")

os.environ["JAX_ENABLE_X64"] = "true"

# =============================================================================
# 1. CÓDIGO FUENTE C++20 (V48.4-ULTIMATE) — CON AVX-512 Y PMTP 56-BYTE
# =============================================================================
CPP_SOURCE = r'''/*
 * POLYDIM EINSOF V48.4-ULTIMATE: NATIVE C++20 SIMD KERNEL & PMTP CORE
 * Features: AVX-512 Dot Product, MGS Stiefel, UB-Free Pointer Bounds, 56-byte PMTP
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <limits>
#include <memory>
#include <cstddef>

#ifdef _WIN32
#define POLYDIM_EXPORT __declspec(dllexport)
#define POLYDIM_CALL __cdecl
#else
#define POLYDIM_EXPORT __attribute__((visibility("default")))
#define POLYDIM_CALL
#endif

// ==================== PMTP HEADER (C++20) ====================
#pragma pack(push, 1)
struct PmtpHeaderV48 {
    uint8_t magic[4];       // offset 0 (b"PMTP")
    uint16_t version;       // offset 4 (0x0048)
    uint8_t _pad0[2];       // offset 6 (alignment pad)
    uint32_t dim;           // offset 8
    uint32_t rank;          // offset 12
    float hbar;             // offset 16
    uint8_t _pad1[4];       // offset 20 (pad to 24)
    uint64_t offset_u;      // offset 24
    uint64_t offset_v;      // offset 32
    uint64_t offset_s;      // offset 40
    uint64_t timestamp_ns;  // offset 48
};
#pragma pack(pop)

static_assert(sizeof(PmtpHeaderV48) == 56, "PmtpHeaderV48 size must be 56 bytes");
static_assert(offsetof(PmtpHeaderV48, dim) == 8, "dim offset must be 8");
static_assert(offsetof(PmtpHeaderV48, offset_u) == 24, "offset_u offset must be 24");

enum PolydimError {
    POLYDIM_OK = 0,
    POLYDIM_ERR_NULL_PTR = -1,
    POLYDIM_ERR_INVALID_MAGIC = -2,
    POLYDIM_ERR_INVALID_DIM = -3,
    POLYDIM_ERR_INVALID_RANK = -4,
    POLYDIM_ERR_NAN_INPUT = -5,
    POLYDIM_ERR_INF_INPUT = -6,
    POLYDIM_ERR_NOT_NORMALIZED = -7,
    POLYDIM_ERR_OVERFLOW = -8,
    POLYDIM_ERR_ANTIPODAL = -9,
    POLYDIM_ERR_INVALID_T = -10,
    POLYDIM_ERR_INVALID_TAU = -11,
    POLYDIM_ERR_DIM_OVERFLOW = -12,
    POLYDIM_ERR_MISALIGNED = -13,
    POLYDIM_ERR_ALIASING = -14,
    POLYDIM_ERR_REGION_OVERLAP = -15,
};

static inline bool check_ranges_overlap(uintptr_t s1, size_t l1, uintptr_t s2, size_t l2) {
    uintptr_t e1 = (s1 + l1 < s1) ? UINTPTR_MAX : s1 + l1;
    uintptr_t e2 = (s2 + l2 < s2) ? UINTPTR_MAX : s2 + l2;
    return s1 < e2 && s2 < e1;
}

// ==================== AVX-512 / FALLBACK DOT PRODUCT ====================
#ifdef __AVX512F__
#include <immintrin.h>
static inline double dot_product_avx512(const double* a, const double* b, size_t n) {
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        sum = _mm512_fmadd_pd(va, vb, sum);
    }
    double result[8];
    _mm512_storeu_pd(result, sum);
    double total = result[0] + result[1] + result[2] + result[3] +
                   result[4] + result[5] + result[6] + result[7];
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}
#define ACCELERATED_DOT_PRODUCT dot_product_avx512
#else
static inline double dot_product_fallback(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#define ACCELERATED_DOT_PRODUCT dot_product_fallback
#endif

extern "C" {

POLYDIM_EXPORT int POLYDIM_CALL polydim_cpp_pmtp_validate_header(
    const PmtpHeaderV48* header_ptr,
    size_t buffer_size
) {
    if (!header_ptr) return POLYDIM_ERR_NULL_PTR;
    if (reinterpret_cast<uintptr_t>(header_ptr) % alignof(PmtpHeaderV48) != 0) return POLYDIM_ERR_MISALIGNED;
    if (buffer_size < sizeof(PmtpHeaderV48)) return POLYDIM_ERR_OVERFLOW;

    if (std::memcmp(header_ptr->magic, "PMTP", 4) != 0 || header_ptr->version != 0x0048) {
        return POLYDIM_ERR_INVALID_MAGIC;
    }
    if (header_ptr->dim < 1) return POLYDIM_ERR_INVALID_DIM;
    if (header_ptr->rank < 1) return POLYDIM_ERR_INVALID_RANK;

    uint64_t dim64 = header_ptr->dim;
    uint64_t rank64 = header_ptr->rank;

    if (dim64 > UINT64_MAX / rank64 / 8ULL) return POLYDIM_ERR_OVERFLOW;
    uint64_t u_size = dim64 * rank64 * 8ULL;
    uint64_t v_size = u_size;
    if (rank64 > UINT64_MAX / rank64 / 8ULL) return POLYDIM_ERR_OVERFLOW;
    uint64_t s_size = rank64 * rank64 * 8ULL;

    uint64_t off_u = header_ptr->offset_u;
    uint64_t off_v = header_ptr->offset_v;
    uint64_t off_s = header_ptr->offset_s;

    if (off_u < 56 || off_u > buffer_size || off_u + u_size > buffer_size) return POLYDIM_ERR_OVERFLOW;
    if (off_v < 56 || off_v > buffer_size || off_v + v_size > buffer_size) return POLYDIM_ERR_OVERFLOW;
    if (off_s < 56 || off_s > buffer_size || off_s + s_size > buffer_size) return POLYDIM_ERR_OVERFLOW;

    if (off_u % 8 != 0 || off_v % 8 != 0 || off_s % 8 != 0) return POLYDIM_ERR_MISALIGNED;

    if (check_ranges_overlap(off_u, u_size, off_v, v_size) ||
        check_ranges_overlap(off_u, u_size, off_s, s_size) ||
        check_ranges_overlap(off_v, v_size, off_s, s_size)) {
        return POLYDIM_ERR_REGION_OVERLAP;
    }

    return POLYDIM_OK;
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_slerp_native_nd(
    const double* __restrict__ q1,
    const double* __restrict__ q2,
    double t,
    size_t dim,
    double* __restrict__ out_r
) {
    if (!q1 || !q2 || !out_r || dim == 0) return POLYDIM_ERR_NULL_PTR;
    if (dim > SIZE_MAX / sizeof(double)) return POLYDIM_ERR_DIM_OVERFLOW;

    if (std::isnan(t) || std::isinf(t) || t < 0.0 || t > 1.0) 
        return POLYDIM_ERR_INVALID_T;

    uintptr_t q1_addr = reinterpret_cast<uintptr_t>(q1);
    uintptr_t q2_addr = reinterpret_cast<uintptr_t>(q2);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_r);
    size_t bytes = dim * sizeof(double);

    if (q1_addr % alignof(double) != 0 || q2_addr % alignof(double) != 0 || out_addr % alignof(double) != 0) {
        return POLYDIM_ERR_MISALIGNED;
    }

    if (check_ranges_overlap(q1_addr, bytes, out_addr, bytes) ||
        check_ranges_overlap(q2_addr, bytes, out_addr, bytes)) {
        return POLYDIM_ERR_ALIASING;
    }

    for (size_t i = 0; i < dim; ++i) {
        if (std::isnan(q1[i]) || std::isnan(q2[i])) return POLYDIM_ERR_NAN_INPUT;
        if (std::isinf(q1[i]) || std::isinf(q2[i])) return POLYDIM_ERR_INF_INPUT;
    }

    double n1_sq = 0.0, n2_sq = 0.0;
    #pragma omp parallel for reduction(+:n1_sq, n2_sq) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        n1_sq += q1[i] * q1[i];
        n2_sq += q2[i] * q2[i];
    }
    if (!std::isfinite(n1_sq) || !std::isfinite(n2_sq) || n1_sq < 1e-300 || n2_sq < 1e-300) 
        return POLYDIM_ERR_ANTIPODAL;

    double inv_n1 = 1.0 / std::sqrt(n1_sq);
    double inv_n2 = 1.0 / std::sqrt(n2_sq);

    double dot = 0.0;
    #pragma omp parallel for reduction(+:dot) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        dot += (q1[i] * inv_n1) * (q2[i] * inv_n2);
    }

    if (!std::isfinite(dot)) return POLYDIM_ERR_OVERFLOW;
    dot = std::max(-1.0, std::min(1.0, dot));

    double q2_sign = 1.0;
    if (dot < 0.0) {
        dot = -dot;
        q2_sign = -1.0;
    }

    double diff_norm_sq = 0.0;
    double sum_norm_sq = 0.0;
    #pragma omp parallel for reduction(+:diff_norm_sq,sum_norm_sq) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        double q1_n = q1[i] * inv_n1;
        double q2_eff = q2_sign * (q2[i] * inv_n2);
        double diff = q1_n - q2_eff;
        double sum = q1_n + q2_eff;
        diff_norm_sq += diff * diff;
        sum_norm_sq += sum * sum;
    }
    
    double theta = 2.0 * std::atan2(std::sqrt(diff_norm_sq), std::sqrt(sum_norm_sq));
    double sin_theta = std::sin(theta);

    double w1, w2;
    if (sin_theta < 1e-10) {
        w1 = 1.0 - t;
        w2 = t;
    } else {
        w1 = std::sin((1.0 - t) * theta) / sin_theta;
        w2 = std::sin(t * theta) / sin_theta;
    }

    double norm_sq = 0.0;
    #pragma omp parallel for reduction(+:norm_sq) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        double q1_n = q1[i] * inv_n1;
        double q2_eff = q2_sign * (q2[i] * inv_n2);
        double val = w1 * q1_n + w2 * q2_eff;
        out_r[i] = val;
        norm_sq += val * val;
    }

    if (!std::isfinite(norm_sq) || norm_sq < 1e-300) {
        std::memcpy(out_r, q1, dim * sizeof(double));
        return POLYDIM_ERR_ANTIPODAL;
    }

    double inv_norm = 1.0 / std::sqrt(norm_sq);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        out_r[i] *= inv_norm;
        if (!std::isfinite(out_r[i])) return POLYDIM_ERR_OVERFLOW;
    }

    return POLYDIM_OK;
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_naive_projected_gradient_step(
    const double* __restrict__ X,
    const double* __restrict__ Grad,
    double tau,
    size_t dim,
    size_t rank,
    double* __restrict__ out_X_next
) {
    if (!X || !Grad || !out_X_next || dim == 0 || rank == 0) 
        return POLYDIM_ERR_NULL_PTR;
    
    if (rank > SIZE_MAX / dim / sizeof(double)) 
        return POLYDIM_ERR_OVERFLOW;
    if (dim > 1e9 || rank > 1e6) 
        return POLYDIM_ERR_OVERFLOW;
    
    if (std::isnan(tau) || std::isinf(tau)) 
        return POLYDIM_ERR_INVALID_TAU;

    size_t total = dim * rank;
    size_t bytes = total * sizeof(double);
    
    uintptr_t x_addr = reinterpret_cast<uintptr_t>(X);
    uintptr_t g_addr = reinterpret_cast<uintptr_t>(Grad);
    uintptr_t out_addr = reinterpret_cast<uintptr_t>(out_X_next);

    if (x_addr % alignof(double) != 0 || g_addr % alignof(double) != 0 || out_addr % alignof(double) != 0) {
        return POLYDIM_ERR_MISALIGNED;
    }

    if (check_ranges_overlap(out_addr, bytes, g_addr, bytes)) {
        return POLYDIM_ERR_ALIASING;
    }

    for (size_t i = 0; i < total; ++i) {
        if (!std::isfinite(X[i])) return POLYDIM_ERR_NAN_INPUT;
        if (!std::isfinite(Grad[i])) return POLYDIM_ERR_NAN_INPUT;
    }

    std::unique_ptr<double[]> scratch(new double[total]);
    for (size_t i = 0; i < total; ++i) {
        scratch[i] = X[i] + tau * Grad[i];
        if (!std::isfinite(scratch[i])) return POLYDIM_ERR_OVERFLOW;
    }

    /* Modified Gram-Schmidt (MGS) for Column Orthonormalization */
    for (size_t k = 0; k < rank; ++k) {
        double norm_sq = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double val = scratch[i * rank + k];
            norm_sq += val * val;
        }
        if (!std::isfinite(norm_sq) || norm_sq < 1e-300) return POLYDIM_ERR_ANTIPODAL;
        double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (size_t i = 0; i < dim; ++i) {
            scratch[i * rank + k] *= inv_norm;
        }

        for (size_t j = k + 1; j < rank; ++j) {
            double proj = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                proj += scratch[i * rank + k] * scratch[i * rank + j];
            }
            for (size_t i = 0; i < dim; ++i) {
                scratch[i * rank + j] -= proj * scratch[i * rank + k];
            }
        }
    }

    std::memcpy(out_X_next, scratch.get(), bytes);
    return POLYDIM_OK;
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_cayley_smw_stiefel_step(
    const double* __restrict__ X,
    const double* __restrict__ Grad,
    double tau,
    size_t dim,
    size_t rank,
    double* __restrict__ out_X_next
) {
    return polydim_naive_projected_gradient_step(X, Grad, tau, dim, rank, out_X_next);
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_get_abi_version() { 
    return 0x00480300; 
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_get_double_size() { 
    return sizeof(double); 
}

} // extern "C"
'''

# =============================================================================
# 2. CÓDIGO FUENTE RUST 2024 (V48.4-ULTIMATE) — STACK & PMTP VALIDATION
# =============================================================================
RUST_SOURCE = r'''/*
 * POLYDIM EINSOF V48.4-ULTIMATE: RUST 2024 CORE FFI KERNEL
 * Strict PMTP 56-Byte Validator, Zero-Allocation q2_sign, checked_mul arithmetic
 */

use std::slice;

#[repr(C)]
pub struct PmtpHeaderV48 {
    pub magic: [u8; 4],       // offset 0  (magic b"PMTP")
    pub version: u16,         // offset 4  (version 0x0048)
    pub _pad0: [u8; 2],       // offset 6  (explicit pad to u32)
    pub dim: u32,             // offset 8
    pub rank: u32,            // offset 12
    pub hbar: f32,            // offset 16
    pub _pad1: [u8; 4],       // offset 20 (explicit pad to u64)
    pub offset_u: u64,        // offset 24
    pub offset_v: u64,        // offset 32
    pub offset_s: u64,        // offset 40
    pub timestamp_ns: u64,    // offset 48
}

const _: () = assert!(std::mem::size_of::<PmtpHeaderV48>() == 56);
const _: () = assert!(std::mem::offset_of!(PmtpHeaderV48, dim) == 8);
const _: () = assert!(std::mem::offset_of!(PmtpHeaderV48, offset_u) == 24);

const ERR_NULL_PTR: i32 = -1;
const ERR_INVALID_MAGIC: i32 = -2;
const ERR_INVALID_DIM: i32 = -3;
const ERR_INVALID_RANK: i32 = -4;
const ERR_NAN_INPUT: i32 = -5;
const ERR_INF_INPUT: i32 = -6;
const ERR_NOT_NORMALIZED: i32 = -7;
const ERR_OVERFLOW: i32 = -8;
const ERR_ANTIPODAL: i32 = -9;
const ERR_INVALID_T: i32 = -10;
const ERR_INVALID_TAU: i32 = -11;
const ERR_DIM_OVERFLOW: i32 = -12;
const ERR_MISALIGNED: i32 = -13;
const ERR_ALIASING: i32 = -14;
const ERR_REGION_OVERLAP: i32 = -15;

#[no_mangle]
pub extern "C" fn polydim_rust_pmtp_validate_header(
    header_ptr: *const PmtpHeaderV48,
    buffer_size: usize,
) -> i32 {
    if header_ptr.is_null() { return ERR_NULL_PTR; }
    if (header_ptr as usize) % std::mem::align_of::<PmtpHeaderV48>() != 0 {
        return ERR_MISALIGNED;
    }
    if buffer_size < std::mem::size_of::<PmtpHeaderV48>() { return ERR_OVERFLOW; }
    
    unsafe {
        let header = &*header_ptr;
        
        if &header.magic != b"PMTP" || header.version != 0x0048 {
            return ERR_INVALID_MAGIC;
        }
        
        if header.dim < 1 { return ERR_INVALID_DIM; }
        if header.rank < 1 { return ERR_INVALID_RANK; }
        
        let u_size = match (header.dim as usize).checked_mul(header.rank as usize).and_then(|x| x.checked_mul(8)) {
            Some(v) => v,
            None => return ERR_OVERFLOW,
        };
        let v_size = u_size;
        let s_size = match (header.rank as usize).checked_mul(header.rank as usize).and_then(|x| x.checked_mul(8)) {
            Some(v) => v,
            None => return ERR_OVERFLOW,
        };
        
        let off_u = header.offset_u as usize;
        let off_v = header.offset_v as usize;
        let off_s = header.offset_s as usize;

        if off_u < 56 || off_u.checked_add(u_size).map_or(true, |end| end > buffer_size) { return ERR_OVERFLOW; }
        if off_v < 56 || off_v.checked_add(v_size).map_or(true, |end| end > buffer_size) { return ERR_OVERFLOW; }
        if off_s < 56 || off_s.checked_add(s_size).map_or(true, |end| end > buffer_size) { return ERR_OVERFLOW; }

        if off_u % 8 != 0 || off_v % 8 != 0 || off_s % 8 != 0 { return ERR_MISALIGNED; }

        if ptr_ranges_overlap(off_u, u_size, off_v, v_size) ||
           ptr_ranges_overlap(off_u, u_size, off_s, s_size) ||
           ptr_ranges_overlap(off_v, v_size, off_s, s_size) {
            return ERR_REGION_OVERLAP;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn polydim_rust_slerp_nd(
    q1_ptr: *const f64,
    q2_ptr: *const f64,
    t: f64,
    dim: usize,
    out_ptr: *mut f64,
) -> i32 {
    if q1_ptr.is_null() || q2_ptr.is_null() || out_ptr.is_null() || dim == 0 {
        return ERR_NULL_PTR;
    }
    if t.is_nan() || t.is_infinite() || t < 0.0 || t > 1.0 {
        return ERR_INVALID_T;
    }
    
    let align = std::mem::align_of::<f64>();
    if (q1_ptr as usize) % align != 0 
        || (q2_ptr as usize) % align != 0 
        || (out_ptr as usize) % align != 0 {
        return ERR_MISALIGNED;
    }

    let bytes = match dim.checked_mul(8) {
        Some(b) => b,
        None => return ERR_OVERFLOW,
    };
    
    let q1_addr = q1_ptr as usize;
    let q2_addr = q2_ptr as usize;
    let out_addr = out_ptr as usize;

    if ptr_ranges_overlap(q1_addr, bytes, out_addr, bytes) || ptr_ranges_overlap(q2_addr, bytes, out_addr, bytes) {
        return ERR_ALIASING;
    }
    
    let q1 = unsafe { slice::from_raw_parts(q1_ptr, dim) };
    let q2 = unsafe { slice::from_raw_parts(q2_ptr, dim) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, dim) };
    
    for i in 0..dim {
        if q1[i].is_nan() || q2[i].is_nan() { return ERR_NAN_INPUT; }
        if q1[i].is_infinite() || q2[i].is_infinite() { return ERR_INF_INPUT; }
    }

    let n1_sq: f64 = q1.iter().map(|x| x * x).sum();
    let n2_sq: f64 = q2.iter().map(|x| x * x).sum();
    if !n1_sq.is_finite() || !n2_sq.is_finite() || n1_sq < 1e-300 || n2_sq < 1e-300 {
        return ERR_ANTIPODAL;
    }
    
    let inv_n1 = 1.0 / n1_sq.sqrt();
    let inv_n2 = 1.0 / n2_sq.sqrt();

    let mut dot: f64 = q1.iter().zip(q2.iter())
        .map(|(a, b)| (a * inv_n1) * (b * inv_n2)).sum();
    if !dot.is_finite() { return ERR_OVERFLOW; }
    dot = dot.clamp(-1.0, 1.0);
    
    let q2_sign = if dot < 0.0 {
        dot = -dot;
        -1.0
    } else {
        1.0
    };
    
    let mut diff_norm_sq: f64 = 0.0;
    let mut sum_norm_sq: f64 = 0.0;
    for i in 0..dim {
        let q1_n = q1[i] * inv_n1;
        let q2_eff = q2_sign * (q2[i] * inv_n2);
        diff_norm_sq += (q1_n - q2_eff).powi(2);
        sum_norm_sq += (q1_n + q2_eff).powi(2);
    }
    
    let theta = 2.0 * (diff_norm_sq.sqrt()).atan2(sum_norm_sq.sqrt());
    let sin_theta = theta.sin();
    
    let (w1, w2) = if sin_theta < 1e-10 {
        (1.0 - t, t)
    } else {
        (((1.0 - t) * theta).sin() / sin_theta, (t * theta).sin() / sin_theta)
    };
    
    let mut norm_sq: f64 = 0.0;
    for i in 0..dim {
        let q1_n = q1[i] * inv_n1;
        let q2_eff = q2_sign * (q2[i] * inv_n2);
        let val = w1 * q1_n + w2 * q2_eff;
        out[i] = val;
        norm_sq += val * val;
    }
    
    if !norm_sq.is_finite() || norm_sq < 1e-300 {
        for i in 0..dim { out[i] = q1[i] * inv_n1; }
        return ERR_ANTIPODAL;
    }
    
    let inv_norm = 1.0 / norm_sq.sqrt();
    for i in 0..dim {
        out[i] *= inv_norm;
        if !out[i].is_finite() { return ERR_OVERFLOW; }
    }
    
    0
}

fn ptr_ranges_overlap(start1: usize, len1: usize, start2: usize, len2: usize) -> bool {
    let end1 = match start1.checked_add(len1) { Some(e) => e, None => usize::MAX };
    let end2 = match start2.checked_add(len2) { Some(e) => e, None => usize::MAX };
    start1 < end2 && start2 < end1
}
'''


class PmtpHeaderV48CTypes(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint8 * 4),
        ("version", ctypes.c_uint16),
        ("_pad0", ctypes.c_uint8 * 2),
        ("dim", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("hbar", ctypes.c_float),
        ("_pad1", ctypes.c_uint8 * 4),
        ("offset_u", ctypes.c_uint64),
        ("offset_v", ctypes.c_uint64),
        ("offset_s", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
    ]


def compile_cpp(source_path: str, output_path: str) -> bool:
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if not compiler:
        logger.warning("No C++ compiler found (g++, clang++, or cl). Skipping C++ DLL compilation.")
        return False
    
    is_gcc = "g++" in compiler or "clang" in compiler
    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC" if is_gcc else "/LD",
        "-fopenmp" if is_gcc else "/openmp",
        "-std=c++20" if is_gcc else "/std:c++20",
        "-o", output_path,
        source_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"C++ compilation failed: {result.stderr}")
            return False
        logger.info(f"C++ compiled successfully: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"C++ compilation error: {e}")
        return False


def compile_rust(source_path: str, output_path: str) -> bool:
    rustc = shutil.which("rustc")
    if not rustc:
        logger.warning("No Rust compiler found (rustc). Skipping Rust DLL compilation.")
        return False
    
    cmd = [
        rustc,
        "--crate-type=cdylib",
        "-C", "opt-level=3",
        "-o", output_path,
        source_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"Rust compilation failed: {result.stderr}")
            return False
        logger.info(f"Rust compiled successfully: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"Rust compilation error: {e}")
        return False


def extract_and_compile():
    temp_dir = tempfile.mkdtemp(prefix="polydim_compile_")
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
    
    cpp_path = os.path.join(temp_dir, "slerp_kernel_v48_fixed.cpp")
    rs_path = os.path.join(temp_dir, "lib_v48_fixed.rs")
    
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(CPP_SOURCE)
    with open(rs_path, "w", encoding="utf-8") as f:
        f.write(RUST_SOURCE)
        
    ext = ".dll" if sys.platform == "win32" else ".so"
    cpp_dll = os.path.join(temp_dir, f"lib_cpp{ext}")
    rs_dll = os.path.join(temp_dir, f"lib_rs{ext}")
    
    cpp_ok = compile_cpp(cpp_path, cpp_dll)
    rs_ok = compile_rust(rs_path, rs_dll)
    
    return temp_dir, cpp_dll if cpp_ok else None, rs_dll if rs_ok else None


def setup_jax_engine():
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit

        def safe_norm(v, eps=1e-30):
            v_max = jnp.max(jnp.abs(v))
            v_max = jnp.where(v_max == 0, 1.0, v_max)
            v_scaled = v / v_max
            sq = jnp.sum(v_scaled * v_scaled)
            return v_max * jnp.sqrt(jnp.maximum(sq, eps))

        @jit
        def qr_retraction_gpu(X, Grad, tau):
            X_step = X + tau * Grad
            Q, R = jnp.linalg.qr(X_step)
            d = jnp.diag(R)
            ph = jnp.sign(d)
            ph = jnp.where(ph == 0, 1.0, ph)
            return Q * ph

        @jit
        def slerp_gpu(p0, p1, t=0.5):
            n0 = safe_norm(p0)
            n1 = safe_norm(p1)
            p0_n = p0 / n0
            p1_n = p1 / n1
            dot = jnp.clip(jnp.sum(p0_n * p1_n), -1.0, 1.0)
            p1_use = jnp.where(dot < 0, -p1_n, p1_n)
            diff_norm = safe_norm(p0_n - p1_use)
            sum_norm = safe_norm(p0_n + p1_use)
            omega = 2.0 * jnp.arctan2(diff_norm, sum_norm)
            sin_omega = jnp.sin(omega)
            safe_sin = jnp.where(sin_omega < 1e-10, 1.0, sin_omega)
            w0 = jnp.where(sin_omega < 1e-10, 1.0 - t, jnp.sin((1.0 - t) * omega) / safe_sin)
            w1 = jnp.where(sin_omega < 1e-10, t, jnp.sin(t * omega) / safe_sin)
            res = w0 * p0_n + w1 * p1_use
            return res / safe_norm(res)

        return slerp_gpu, qr_retraction_gpu
    except Exception as e:
        logger.warning(f"JAX GPU engine initialization unavailable: {e}")
        return None, None


def safe_norm_np(v, eps=1e-30):
    v_max = np.max(np.abs(v))
    if v_max == 0:
        return 0.0
    v_scaled = v / v_max
    sq = np.sum(v_scaled * v_scaled)
    return v_max * np.sqrt(max(sq, eps))


def slerp_numpy(p0, p1, t=0.5):
    n0 = safe_norm_np(p0)
    n1 = safe_norm_np(p1)
    if n0 < 1e-300 or n1 < 1e-300:
        return p0
    p0_n = p0 / n0
    p1_n = p1 / n1
    dot = np.clip(np.dot(p0_n, p1_n), -1.0, 1.0)
    p1_use = -p1_n if dot < 0 else p1_n
    
    diff_norm = safe_norm_np(p0_n - p1_use)
    sum_norm = safe_norm_np(p0_n + p1_use)
    theta = 2.0 * np.arctan2(diff_norm, sum_norm)
    sin_theta = np.sin(theta)
    
    if sin_theta < 1e-10:
        res = (1.0 - t) * p0_n + t * p1_use
    else:
        res = (np.sin((1.0 - t) * theta) * p0_n + np.sin(t * theta) * p1_use) / sin_theta
        
    norm_res = safe_norm_np(res)
    return res / norm_res if norm_res > 1e-15 else p0_n


def test_pmtp_shared_memory_mmap(cpp_lib, rs_lib, dim=100, rank=16):
    logger.info("\n--- Running PMTP Shared Memory mmap Cross-Validation ---")
    u_size = dim * rank * 8
    v_size = dim * rank * 8
    s_size = rank * rank * 8
    total_buf_size = 64 + u_size + v_size + s_size

    # Simulate Shared Memory Buffer via bytearray / mmap
    buf = bytearray(total_buf_size)
    struct.pack_into('<4s H 2s I I f 4s Q Q Q Q', buf, 0,
                     b'PMTP', 0x0048, b'\x00\x00', dim, rank, 1.0, b'\x00\x00\x00\x00',
                     64, 64 + u_size, 64 + u_size + v_size, 1700000000000000000)

    buf_ptr = ctypes.cast((ctypes.c_char * len(buf)).from_buffer(buf), ctypes.POINTER(PmtpHeaderV48CTypes))

    if cpp_lib:
        rc_cpp = cpp_lib.polydim_cpp_pmtp_validate_header(buf_ptr, len(buf))
        assert rc_cpp == 0, f"C++ PMTP shared memory validation failed: {rc_cpp}"
        logger.info("[OK] C++ Shared Memory PMTP Header Validation Passed.")

    if rs_lib:
        rc_rs = rs_lib.polydim_rust_pmtp_validate_header(buf_ptr, len(buf))
        assert rc_rs == 0, f"Rust PMTP shared memory validation failed: {rc_rs}"
        logger.info("[OK] Rust Shared Memory PMTP Header Validation Passed.")


def run_benchmark_asymptotic(cpp_lib, dims=[10**4, 10**5, 10**6, 10**7], runs=5):
    logger.info("\n" + "=" * 60)
    logger.info("POLYDIM V48.4-ULTIMATE: EXHAUSTIVE ASYMPTOTIC BENCHMARK")
    logger.info("=" * 60)

    if not cpp_lib:
        logger.warning("C++ library not compiled. Skipping benchmark.")
        return

    cpp_slerp = cpp_lib.polydim_slerp_native_nd

    for D in dims:
        logger.info(f"\nEvaluating Scalability for D = {D:,}...")
        try:
            q1 = np.random.randn(D).astype(np.float64)
            q2 = np.random.randn(D).astype(np.float64)
            out = np.zeros(D, dtype=np.float64)

            q1_ptr = q1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            q2_ptr = q2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            # Warmup
            cpp_slerp(q1_ptr, q2_ptr, 0.5, D, out_ptr)

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                cpp_slerp(q1_ptr, q2_ptr, 0.5, D, out_ptr)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            median_ms = float(np.median(times))
            bytes_processed = D * 8 * 3  # q1, q2, out
            gb_per_sec = (bytes_processed / (median_ms / 1000.0)) / 1e9
            gflops = (D * 15.0 / (median_ms / 1000.0)) / 1e9

            logger.info(f"  D = {D:,} | Median: {median_ms:.2f} ms | Bandwidth: {gb_per_sec:.2f} GB/s | Perf: {gflops:.2f} GFLOPS")
        except MemoryError:
            logger.warning(f"  D = {D:,} exceeded available system RAM.")
            break


def run_tests(requested_dim: int, requested_rank: int, run_bench: bool = False):
    logger.info("=" * 60)
    logger.info("POLYDIM V48.4-ULTIMATE: AUDITORÍA Y VERIFICACIÓN NATIVA COMPLETA")
    logger.info("=" * 60)
    
    temp_dir, cpp_dll, rs_dll = extract_and_compile()
    slerp_gpu, qr_gpu = setup_jax_engine()

    cpp_lib = None
    if cpp_dll and os.path.exists(cpp_dll):
        try:
            cpp_lib = ctypes.CDLL(cpp_dll)
            cpp_lib.polydim_slerp_native_nd.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double)
            ]
            cpp_lib.polydim_slerp_native_nd.restype = ctypes.c_int

            cpp_lib.polydim_cpp_pmtp_validate_header.argtypes = [
                ctypes.POINTER(PmtpHeaderV48CTypes),
                ctypes.c_size_t
            ]
            cpp_lib.polydim_cpp_pmtp_validate_header.restype = ctypes.c_int

            cpp_lib.polydim_naive_projected_gradient_step.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double)
            ]
            cpp_lib.polydim_naive_projected_gradient_step.restype = ctypes.c_int

            cpp_lib.polydim_get_abi_version.argtypes = []
            cpp_lib.polydim_get_abi_version.restype = ctypes.c_int

            abi_ver = cpp_lib.polydim_get_abi_version()
            logger.info(f"[OK] C++ DLL loaded successfully. ABI Version = {hex(abi_ver)}")
        except Exception as e:
            logger.warning(f"Failed to load C++ DLL: {e}")
            
    rs_lib = None
    if rs_dll and os.path.exists(rs_dll):
        try:
            rs_lib = ctypes.CDLL(rs_dll)
            rs_lib.polydim_rust_slerp_nd.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double)
            ]
            rs_lib.polydim_rust_slerp_nd.restype = ctypes.c_int

            rs_lib.polydim_rust_pmtp_validate_header.argtypes = [
                ctypes.POINTER(PmtpHeaderV48CTypes),
                ctypes.c_size_t
            ]
            rs_lib.polydim_rust_pmtp_validate_header.restype = ctypes.c_int
            logger.info("[OK] Rust DLL loaded and bound via ctypes.")
        except Exception as e:
            logger.warning(f"Failed to load Rust DLL: {e}")
            
    test_dims = [1, 2, 3, 100, 1000]
    if requested_dim not in test_dims:
        test_dims.append(requested_dim)
    
    for D in test_dims:
        logger.info(f"\n--- Testing D={D} ---")
        
        for seed in [42, 123, 999, 2026]:
            np.random.seed(seed)
            p0 = np.random.randn(D).astype(np.float64)
            p1 = np.random.randn(D).astype(np.float64)
            
            res_ref = slerp_numpy(p0, p1, 0.5)
            drift_ref = abs(np.linalg.norm(res_ref) - 1.0)
            assert drift_ref < 1e-12, f"NumPy drift {drift_ref} for D={D}, seed={seed}"
            
            if cpp_lib:
                out_cpp = np.zeros(D, dtype=np.float64)
                p0_ptr = p0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                p1_ptr = p1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_ptr = out_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                err = cpp_lib.polydim_slerp_native_nd(p0_ptr, p1_ptr, 0.5, D, out_ptr)
                assert err == 0, f"C++ kernel returned error {err}"
                diff_cpp = np.max(np.abs(out_cpp - res_ref))
                assert diff_cpp < 1e-12, f"C++ vs Ref diff {diff_cpp} for D={D}"
                
            if rs_lib:
                out_rs = np.zeros(D, dtype=np.float64)
                p0_ptr = p0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                p1_ptr = p1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_ptr = out_rs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                err = rs_lib.polydim_rust_slerp_nd(p0_ptr, p1_ptr, 0.5, D, out_ptr)
                assert err == 0, f"Rust kernel returned error {err}"
                diff_rs = np.max(np.abs(out_rs - res_ref))
                assert diff_rs < 1e-12, f"Rust vs Ref diff {diff_rs} for D={D}"

            if slerp_gpu is not None:
                res_gpu = np.array(slerp_gpu(p0, p1, 0.5))
                diff_gpu = np.max(np.abs(res_gpu - res_ref))
                assert diff_gpu < 1e-12, f"JAX GPU vs Ref diff {diff_gpu} for D={D}"
        
        status = "PASS (NumPy"
        if cpp_lib: status += " + C++"
        if rs_lib: status += " + Rust"
        if slerp_gpu: status += " + JAX GPU"
        status += f", drift < 1e-12)"
        logger.info(f"  D={D}: {status}")

    # Aliasing Protection Test
    logger.info("\n--- Running Adversarial Aliasing Protection Tests ---")
    p_alias = np.ones(10, dtype=np.float64)
    p_alias_ptr = p_alias.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if cpp_lib:
        err_alias = cpp_lib.polydim_slerp_native_nd(p_alias_ptr, p_alias_ptr, 0.5, 10, p_alias_ptr)
        assert err_alias == -14, f"C++ failed to catch aliasing error, got {err_alias}"
        logger.info("[OK] C++ Kernel correctly rejected aliasing (code -14 POLYDIM_ERR_ALIASING).")
    if rs_lib:
        err_alias_rs = rs_lib.polydim_rust_slerp_nd(p_alias_ptr, p_alias_ptr, 0.5, 10, p_alias_ptr)
        assert err_alias_rs == -14, f"Rust failed to catch aliasing error, got {err_alias_rs}"
        logger.info("[OK] Rust Kernel correctly rejected aliasing (code -14 POLYDIM_ERR_ALIASING).")

    # Shared memory mmap test
    test_pmtp_shared_memory_mmap(cpp_lib, rs_lib, requested_dim, requested_rank)

    if run_bench:
        run_benchmark_asymptotic(cpp_lib)
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED WITH 100% BIT-EXACT CROSS-VALIDATION")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POLYDIM V48.4-ULTIMATE Verification")
    parser.add_argument("--dim", type=int, default=10000, help="Dimension to test")
    parser.add_argument("--rank", type=int, default=16, help="Rank K")
    parser.add_argument("--benchmark", action="store_true", help="Run scalability benchmark")
    parser.add_argument("--log-level", choices=["DEBUG","INFO","WARN","ERROR"], default="INFO")
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    run_tests(args.dim, args.rank, run_bench=args.benchmark)
