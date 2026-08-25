/*
 * lib_v47_matrix_free.rs
 * POLYDIM EINSOF V48.0-SOTA: REWRITTEN RUST 2024 CORE KERNEL
 * PARCHADO SEGÚN AUDITORÍA RED TEAM:
 * 1. PmtpHeaderV44: Utliza repr(C) con _pad0 explícito (8-byte alignment) sin packed.
 * 2. Header Validation: Exige dim >= 1 universal.
 * 3. Antipodal SLERP: Selección determinista de eje ortogonal sin divisiones por cero.
 * 4. Aliasing defense: Rechazo de solapamiento in-place q1_ptr == out_ptr.
 */

use std::sync::atomic::{AtomicU64, Ordering};
use std::slice;

#[repr(C)]
pub struct PmtpHeaderV44 {
    pub magic: [u8; 4],
    pub version: u16,
    pub dim: u32,
    pub rank: u32,
    pub hbar: f32,
    pub _pad0: u32,
    pub offset_u: u64,
    pub offset_v: u64,
    pub offset_s: u64,
    pub timestamp_ns: AtomicU64,
}

#[no_mangle]
pub extern "C" fn polydim_rust_pmtp_validate_header(header_ptr: *const PmtpHeaderV44) -> i32 {
    if header_ptr.is_null() { return -1; }
    unsafe {
        let header = &*header_ptr;
        if &header.magic != b"PMTP" || header.version != 0x0044 { return -2; }
        if header.dim < 1 || header.rank == 0 { return -3; }
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
    if q1_ptr.is_null() || q2_ptr.is_null() || out_ptr.is_null() || dim == 0 { return -1; }
    if !t.is_finite() { return -4; }

    let q1_addr = q1_ptr as usize;
    let out_addr = out_ptr as usize;
    if q1_addr == out_addr { return -5; }

    let q1 = unsafe { slice::from_raw_parts(q1_ptr, dim) };
    let q2 = unsafe { slice::from_raw_parts(q2_ptr, dim) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, dim) };

    for i in 0..dim {
        if !q1[i].is_finite() || !q2[i].is_finite() { return -3; }
    }

    let mut dot: f64 = q1.iter().zip(q2.iter()).map(|(a, b)| a * b).sum();
    dot = dot.clamp(-1.0, 1.0);

    if dot <= -1.0 + 1e-12 {
        let mut min_idx = 0;
        let mut min_val = q1[0].abs();
        for i in 1..dim {
            let abs_val = q1[i].abs();
            if abs_val < min_val {
                min_val = abs_val;
                min_idx = i;
            }
        }

        let proj = q1[min_idx];
        let mut norm_sq: f64 = 0.0;
        for i in 0..dim {
            let e_i = if i == min_idx { 1.0 } else { 0.0 };
            let val = e_i - proj * q1[i];
            out[i] = val;
            norm_sq += val * val;
        }

        let inv_norm = if norm_sq > 1e-300 { 1.0 / norm_sq.sqrt() } else { 1.0 };
        let theta = std::f64::consts::PI * t;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        for i in 0..dim {
            out[i] = cos_t * q1[i] + sin_t * (out[i] * inv_norm);
        }
        return 0;
    }

    let theta = dot.acos();
    let sin_theta = theta.sin();
    let (w1, w2) = if sin_theta < 1e-12 {
        (1.0 - t, t)
    } else {
        (((1.0 - t) * theta).sin() / sin_theta, (t * theta).sin() / sin_theta)
    };

    let mut norm_sq: f64 = 0.0;
    for i in 0..dim {
        let val = w1 * q1[i] + w2 * q2[i];
        out[i] = val;
        norm_sq += val * val;
    }

    let inv_norm = if norm_sq > 1e-300 { 1.0 / norm_sq.sqrt() } else { 1.0 };
    for i in 0..dim { out[i] *= inv_norm; }
    0
}
