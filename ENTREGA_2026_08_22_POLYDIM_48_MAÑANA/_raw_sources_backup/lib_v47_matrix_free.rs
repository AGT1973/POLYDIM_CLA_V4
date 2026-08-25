/*
 * POLYDIM EINSOF V47.0-SOTA: REWRITTEN RUST 2024 CORE KERNEL
 * Lock-Free Atomic Shared Memory Bus, Cayley-SMW Matrix-Free Retraction & PMTP v44 Header
 */

use std::sync::atomic::{AtomicU64, Ordering};
use std::slice;

#[repr(C, packed)]
pub struct PmtpHeaderV44 {
    pub magic: [u8; 4],      // "PMTP"
    pub version: u16,        // 0x0044
    pub dim: u32,            // Dimension D (e.g. 10000)
    pub rank: u32,           // Rank K (e.g. 32)
    pub hbar: f32,           // Noncommutative scale
    pub offset_u: u64,       // Buffer offset U
    pub offset_v: u64,       // Buffer offset V
    pub offset_s: u64,       // Buffer offset S
    pub timestamp_ns: AtomicU64, // Lock-free atomic timestamp
}

#[no_mangle]
pub extern "C" fn polydim_rust_pmtp_validate_header(header_ptr: *const PmtpHeaderV44) -> i32 {
    if header_ptr.is_null() {
        return -1;
    }
    unsafe {
        let header = &*header_ptr;
        if &header.magic != b"PMTP" || header.version != 0x0044 {
            return -2; // Invalid magic or version
        }
        if header.dim < 1000 || header.rank == 0 {
            return -3; // Invalid dimension bounds
        }
    }
    0 // Valid PMTP v44 header
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
        return -1;
    }

    let q1 = unsafe { slice::from_raw_parts(q1_ptr, dim) };
    let q2 = unsafe { slice::from_raw_parts(q2_ptr, dim) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, dim) };

    // Compute dot product
    let mut dot: f64 = q1.iter().zip(q2.iter()).map(|(a, b)| a * b).sum();
    dot = dot.clamp(-1.0, 1.0);

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

    let inv_norm = 1.0 / norm_sq.sqrt();
    for i in 0..dim {
        out[i] *= inv_norm;
    }

    0
}
