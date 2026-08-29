// ============================================================================
// POLYDIM V79 BULLDOG — RUST SIMD KERNELS (PORTABLE + FEATURE FLAGS)
// ============================================================================
// Fixes:
//   - Feature flags por arquitectura (neon, avx2, avx512)
//   - portable-vectors para dispatch automático
//   - catch_unwind + alignment + overflow (heredado del fix anterior)
// ============================================================================

#![cfg_attr(not(feature = "std"), no_std)]

use std::panic::catch_unwind;

// Feature flags para SIMD
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
use std::arch::x86_64::*;

// ============================================================================
// SCALAR FALLBACK (siempre disponible)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn polydim_householder_reflect_rust(
    x: *const f64,
    v: *const f64,
    out: *mut f64,
    dim: u64,
    batch: u64,
) -> i32 {
    catch_unwind(|| {
        if x.is_null() || v.is_null() || out.is_null() || dim == 0 || batch == 0 {
            return -1i32;
        }

        let dim_us = if dim > usize::MAX as u64 { return -3i32; } else { dim as usize };
        let batch_us = if batch > usize::MAX as u64 { return -3i32; } else { batch as usize };
        let byte_len = dim_us.checked_mul(8);
        if byte_len.is_none() { return -3i32; }
        let byte_len = byte_len.unwrap();

        // Alignment
        let align = std::mem::align_of::<f64>();
        if (x as usize) % align != 0 || (v as usize) % align != 0 || (out as usize) % align != 0 {
            return -4i32;
        }

        // Global overlap
        let total = dim_us.checked_mul(batch_us).and_then(|n| n.checked_mul(8));
        if total.is_none() { return -3i32; }
        let total = total.unwrap();

        let x_addr = x as usize;
        let v_addr = v as usize;
        let out_addr = out as usize;
        let x_end = x_addr.saturating_add(total);
        let v_end = v_addr.saturating_add(total);
        let out_end = out_addr.saturating_add(total);

        if v_addr < out_end && out_addr < v_end { return 1i32; }
        if x_addr < out_end && out_addr < x_end { return 2i32; }

        // Dispatch SIMD o scalar
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            return householder_neon(x, v, out, dim_us, batch_us, byte_len);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            return householder_avx512(x, v, out, dim_us, batch_us, byte_len);
        }

        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx512"),
        )))]
        {
            return householder_scalar(x, v, out, dim_us, batch_us, byte_len);
        }
    }).unwrap_or(-2)
}

// ============================================================================
// SCALAR IMPLEMENTATION
// ============================================================================

unsafe fn householder_scalar(
    x: *const f64, v: *const f64, out: *mut f64,
    dim: usize, batch: usize, byte_len: usize,
) -> i32 {
    for b in 0..batch {
        let x_ptr = x.add(b * dim);
        let v_ptr = v.add(b * dim);
        let out_ptr = out.add(b * dim);

        // Intra-batch overlap
        let v_b_addr = v_ptr as usize;
        let out_b_addr = out_ptr as usize;
        let v_b_end = v_b_addr.saturating_add(byte_len);
        let out_b_end = out_b_addr.saturating_add(byte_len);
        if v_b_addr < out_b_end && out_b_addr < v_b_end { return 1i32; }
        let x_b_addr = x_ptr as usize;
        let x_b_end = x_b_addr.saturating_add(byte_len);
        if x_b_addr < out_b_end && out_b_addr < x_b_end { return 2i32; }

        let xb = std::slice::from_raw_parts(x_ptr, dim);
        let vb = std::slice::from_raw_parts(v_ptr, dim);
        let ob = std::slice::from_raw_parts_mut(out_ptr, dim);

        let mut v_max = 0.0_f64;
        for &val in vb.iter() {
            let av = val.abs();
            if av > v_max { v_max = av; }
        }

        if v_max < 1e-30 {
            ob.copy_from_slice(xb);
            continue;
        }

        let inv_v_max = 1.0 / v_max;
        let mut v_sq = 0.0_f64;
        let mut xv_dot = 0.0_f64;
        for i in 0..dim {
            let vn = vb[i] * inv_v_max;
            v_sq += vn * vn;
            xv_dot += xb[i] * vn;
        }

        let factor = (2.0 * xv_dot / v_sq) * inv_v_max;
        for i in 0..dim {
            ob[i] = xb[i] - factor * vb[i];
        }
    }
    0i32
}

// ============================================================================
// NEON IMPLEMENTATION (ARM64)
// ============================================================================

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
unsafe fn householder_neon(
    x: *const f64, v: *const f64, out: *mut f64,
    dim: usize, batch: usize, byte_len: usize,
) -> i32 {
    for b in 0..batch {
        let x_ptr = x.add(b * dim);
        let v_ptr = v.add(b * dim);
        let out_ptr = out.add(b * dim);

        let xb = std::slice::from_raw_parts(x_ptr, dim);
        let vb = std::slice::from_raw_parts(v_ptr, dim);
        let ob = std::slice::from_raw_parts_mut(out_ptr, dim);

        // v_max con NEON
        let mut v_max = 0.0_f64;
        let mut i = 0_usize;
        while i + 1 < dim {
            let vi = vld1q_f64(vb.as_ptr().add(i));
            let abs_vi = vabsq_f64(vi);
            let max_lane = std::cmp::max(
                vgetq_lane_f64(abs_vi, 0).abs(),
                vgetq_lane_f64(abs_vi, 1).abs(),
            );
            if max_lane > v_max { v_max = max_lane; }
            i += 2;
        }
        for j in i..dim {
            let av = vb[j].abs();
            if av > v_max { v_max = av; }
        }

        if v_max < 1e-30 {
            ob.copy_from_slice(xb);
            continue;
        }

        let inv_v_max = 1.0 / v_max;
        let inv_max_vec = vdupq_n_f64(inv_v_max);

        let mut v_sq = 0.0_f64;
        let mut xv_dot = 0.0_f64;
        i = 0;
        while i + 1 < dim {
            let vi = vld1q_f64(vb.as_ptr().add(i));
            let xi = vld1q_f64(xb.as_ptr().add(i));
            let vn = vmulq_f64(vi, inv_max_vec);
            v_sq += vgetq_lane_f64(vn, 0) * vgetq_lane_f64(vn, 0)
                  + vgetq_lane_f64(vn, 1) * vgetq_lane_f64(vn, 1);
            xv_dot += vgetq_lane_f64(xi, 0) * vgetq_lane_f64(vn, 0)
                    + vgetq_lane_f64(xi, 1) * vgetq_lane_f64(vn, 1);
            i += 2;
        }
        for j in i..dim {
            let vn = vb[j] * inv_v_max;
            v_sq += vn * vn;
            xv_dot += xb[j] * vn;
        }

        let factor = (2.0 * xv_dot / v_sq) * inv_v_max;
        let factor_vec = vdupq_n_f64(factor);

        i = 0;
        while i + 1 < dim {
            let xi = vld1q_f64(xb.as_ptr().add(i));
            let vi = vld1q_f64(vb.as_ptr().add(i));
            let result = vsubq_f64(xi, vmulq_f64(factor_vec, vi));
            vst1q_f64(ob.as_mut_ptr().add(i), result);
            i += 2;
        }
        for j in i..dim {
            ob[j] = xb[j] - factor * vb[j];
        }
    }
    0i32
}

// ============================================================================
// AVX-512 IMPLEMENTATION (x86_64)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
unsafe fn householder_avx512(
    x: *const f64, v: *const f64, out: *mut f64,
    dim: usize, batch: usize, byte_len: usize,
) -> i32 {
    for b in 0..batch {
        let x_ptr = x.add(b * dim);
        let v_ptr = v.add(b * dim);
        let out_ptr = out.add(b * dim);

        let xb = std::slice::from_raw_parts(x_ptr, dim);
        let vb = std::slice::from_raw_parts(v_ptr, dim);
        let ob = std::slice::from_raw_parts_mut(out_ptr, dim);

        // v_max con AVX-512
        let mut v_max = 0.0_f64;
        let mut i = 0_usize;
        while i + 7 < dim {
            let vi = _mm512_loadu_pd(vb.as_ptr().add(i));
            let abs_vi = _mm512_abs_pd(vi);
            let max_lane = _mm512_reduce_max_pd(abs_vi);
            if max_lane > v_max { v_max = max_lane; }
            i += 8;
        }
        for j in i..dim {
            let av = vb[j].abs();
            if av > v_max { v_max = av; }
        }

        if v_max < 1e-30 {
            ob.copy_from_slice(xb);
            continue;
        }

        let inv_v_max = 1.0 / v_max;
        let inv_max_vec = _mm512_set1_pd(inv_v_max);

        let mut v_sq = 0.0_f64;
        let mut xv_dot = 0.0_f64;
        i = 0;
        while i + 7 < dim {
            let vi = _mm512_loadu_pd(vb.as_ptr().add(i));
            let xi = _mm512_loadu_pd(xb.as_ptr().add(i));
            let vn = _mm512_mul_pd(vi, inv_max_vec);
            v_sq += _mm512_reduce_add_pd(_mm512_mul_pd(vn, vn));
            xv_dot += _mm512_reduce_add_pd(_mm512_mul_pd(xi, vn));
            i += 8;
        }
        for j in i..dim {
            let vn = vb[j] * inv_v_max;
            v_sq += vn * vn;
            xv_dot += xb[j] * vn;
        }

        let factor = (2.0 * xv_dot / v_sq) * inv_v_max;
        let factor_vec = _mm512_set1_pd(factor);

        i = 0;
        while i + 7 < dim {
            let xi = _mm512_loadu_pd(xb.as_ptr().add(i));
            let vi = _mm512_loadu_pd(vb.as_ptr().add(i));
            let result = _mm512_fnmadd_pd(factor_vec, vi, xi);
            _mm512_storeu_pd(ob.as_mut_ptr().add(i), result);
            i += 8;
        }
        for j in i..dim {
            ob[j] = xb[j] - factor * vb[j];
        }
    }
    0i32
}

// ============================================================================
// CARGO.TOML RECOMENDADO
// ============================================================================
/*
[package]
name = "polydim_kernel_v79"
version = "0.79.0"
edition = "2021"

[features]
default = ["std"]
std = []
neon = []
avx2 = []
avx512 = []

[lib]
crate-type = ["cdylib", "staticlib"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
*/
