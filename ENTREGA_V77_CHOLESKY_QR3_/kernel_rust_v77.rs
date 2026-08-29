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
