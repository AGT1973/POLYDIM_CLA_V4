/*
 * POLYDIM EINSOF V48.1-FIXED: REWRITTEN RUST 2024 CORE KERNEL
 * Correcciones aplicadas:
 *   - BLOCK-03: #[repr(C, packed)] eliminado, padding explícito
 *   - BLOCK-04: SLERP antipodal manejado
 *   - ALTO-05: Verificación de alineación en from_raw_parts
 *   - ALTO-10: Verificación de aliasing
 *   - MED-02: Normalización de inputs
 *   - MED-07: Endianness documentado (little-endian)
 */

use std::slice;

#[repr(C)]
pub struct PmtpHeaderV48 {
    pub magic: [u8; 4],       // offset 0
    pub version: u16,         // offset 4
    pub dim: u32,             // offset 6
    pub rank: u32,            // offset 10
    pub hbar: f32,            // offset 14
    pub _pad1: [u8; 2],       // offset 18 → padding hasta 24
    pub offset_u: u64,        // offset 24, alineado 8
    pub offset_v: u64,        // offset 32, alineado 8
    pub offset_s: u64,        // offset 40, alineado 8
    pub timestamp_ns: u64,    // offset 48, alineado 8 (u64 plano, no AtomicU64)
}

const _: () = assert!(std::mem::size_of::<PmtpHeaderV48>() == 56);

const ERR_NULL_PTR: i32 = -1;
const ERR_INVALID_MAGIC: i32 = -2;
const ERR_INVALID_DIM: i32 = -3;
const ERR_MISALIGNED: i32 = -7;
const ERR_OVERFLOW: i32 = -8;
const ERR_ALIASING: i32 = -5;
const ERR_ANTIPODAL: i32 = -8;

#[no_mangle]
pub extern "C" fn polydim_rust_pmtp_validate_header(
    header_ptr: *const PmtpHeaderV48,
    buffer_size: usize,
) -> i32 {
    if header_ptr.is_null() { return ERR_NULL_PTR; }
    
    unsafe {
        let header = &*header_ptr;
        
        if &header.magic != b"PMTP" || header.version != 0x0048 {
            return ERR_INVALID_MAGIC;
        }
        
        if header.dim < 1 || header.rank == 0 {
            return ERR_INVALID_DIM;
        }
        
        let u_size = header.dim as usize * header.rank as usize * 8;
        let v_size = header.dim as usize * header.rank as usize * 8;
        let s_size = header.rank as usize * header.rank as usize * 8;
        
        if header.offset_u as usize + u_size > buffer_size { return -4; }
        if header.offset_v as usize + v_size > buffer_size { return -5; }
        if header.offset_s as usize + s_size > buffer_size { return -6; }
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
    
    let align = std::mem::align_of::<f64>();
    if (q1_ptr as usize) % align != 0 
        || (q2_ptr as usize) % align != 0 
        || (out_ptr as usize) % align != 0 {
        return ERR_MISALIGNED;
    }
    
    let q1_range = q1_ptr as usize..(q1_ptr as usize + dim * 8);
    let out_range = out_ptr as usize..(out_ptr as usize + dim * 8);
    let q2_range = q2_ptr as usize..(q2_ptr as usize + dim * 8);
    
    if ranges_overlap(&q1_range, &out_range) || ranges_overlap(&q2_range, &out_range) {
        return ERR_ALIASING;
    }
    
    if dim > (isize::MAX as usize / std::mem::size_of::<f64>()) {
        return ERR_OVERFLOW;
    }
    
    let q1 = unsafe { slice::from_raw_parts(q1_ptr, dim) };
    let q2 = unsafe { slice::from_raw_parts(q2_ptr, dim) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, dim) };
    
    let n1: f64 = q1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let n2: f64 = q2.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n1 < 1e-300 || n2 < 1e-300 {
        return ERR_ANTIPODAL;
    }
    
    let q1_n: Vec<f64> = q1.iter().map(|x| x / n1).collect();
    let q2_n: Vec<f64> = q2.iter().map(|x| x / n2).collect();
    
    let mut dot: f64 = q1_n.iter().zip(q2_n.iter()).map(|(a, b)| a * b).sum();
    dot = dot.clamp(-1.0, 1.0);
    
    let q2_use: Vec<f64> = if dot < 0.0 {
        dot = -dot;
        q2_n.iter().map(|x| -x).collect()
    } else {
        q2_n.clone()
    };
    
    let diff_norm_sq: f64 = q1_n.iter().zip(q2_use.iter())
        .map(|(a, b)| (a - b).powi(2)).sum();
    let sum_norm_sq: f64 = q1_n.iter().zip(q2_use.iter())
        .map(|(a, b)| (a + b).powi(2)).sum();
    let theta = 2.0 * (diff_norm_sq.sqrt()).atan2(sum_norm_sq.sqrt());
    let sin_theta = theta.sin();
    
    let (w1, w2) = if sin_theta < 1e-12 {
        (1.0 - t, t)
    } else {
        (((1.0 - t) * theta).sin() / sin_theta, (t * theta).sin() / sin_theta)
    };
    
    let mut norm_sq: f64 = 0.0;
    for i in 0..dim {
        let val = w1 * q1_n[i] + w2 * q2_use[i];
        out[i] = val;
        norm_sq += val * val;
    }
    
    if norm_sq < 1e-300 {
        return ERR_ANTIPODAL;
    }
    
    let inv_norm = 1.0 / norm_sq.sqrt();
    for i in 0..dim {
        out[i] *= inv_norm;
    }
    
    0
}

fn ranges_overlap(a: &std::ops::Range<usize>, b: &std::ops::Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}
