
// POLYDIM V66 RUST FFI C-ABI KERNEL
#[repr(C)]
pub struct PMTPHeaderC {
    pub seq_word: u64,
    pub magic: u64,
    pub version: u32,
    pub dim: u32,
    pub dtype_code: u32,
    pub payload_bytes: u32,
    pub timestamp: u64,
    pub generation: u64,
    pub _reserved: [u8; 16],
}

#[no_mangle]
pub unsafe extern "C" fn polydim_rust_householder_reflect(
    x_ptr: *const f32,
    v_ptr: *const f32,
    out_ptr: *mut f32,
    dim: usize,
) -> i32 {
    if x_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() || dim == 0 {
        return -1;
    }
    let x = std::slice::from_raw_parts(x_ptr, dim);
    let v = std::slice::from_raw_parts(v_ptr, dim);
    let out = std::slice::from_raw_parts_mut(out_ptr, dim);

    let mut scale: f32 = 0.0;
    for i in 0..dim {
        let av = v[i].abs();
        if av > scale { scale = av; }
    }
    if scale == 0.0 {
        out.copy_from_slice(x);
        return 0;
    }
    let inv_scale = 1.0 / scale;
    let mut vv_scaled: f32 = 0.0;
    for i in 0..dim { let vi = v[i] * inv_scale; vv_scaled += vi * vi; }

    let norm_v = scale * vv_scaled.sqrt();
    if norm_v < 1e-15 {
        out.copy_from_slice(x);
        return 0;
    }

    let alpha = 1.0 / norm_v;
    let mut dot: f32 = 0.0;
    for i in 0..dim { dot += (v[i] * alpha) * x[i]; }

    let two_dot = 2.0 * dot;
    for i in 0..dim { out[i] = x[i] - two_dot * (v[i] * alpha); }
    0
}
