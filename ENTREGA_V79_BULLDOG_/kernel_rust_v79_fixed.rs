// ============================================================================
// POLYDIM V79 BULLDOG — NATIVE RUST KERNELS (CORREGIDO)
// ============================================================================
// Fixes:
//   - catch_unwind para contener panic en FFI boundary
//   - Verificación de x vs out (aliasing estricto de Rust)
//   - Alignment check antes de from_raw_parts
//   - Overflow-safe batch/dim casting
//   - Pre-calcular inv_v_max (sin división en hot loop)
//   - Saturating add para overlap check
// ============================================================================

use std::panic::catch_unwind;

#[no_mangle]
pub unsafe extern "C" fn polydim_householder_reflect_rust(
    x: *const f64,
    v: *const f64,
    out: *mut f64,
    dim: u64,
    batch: u64,
) -> i32 {
    let result = catch_unwind(|| {
        // Null checks
        if x.is_null() || v.is_null() || out.is_null() || dim == 0 || batch == 0 {
            return -1i32;
        }

        let dim_us = if dim > usize::MAX as u64 {
            return -3i32;
        } else {
            dim as usize
        };

        let batch_us = if batch > usize::MAX as u64 {
            return -3i32;
        } else {
            batch as usize
        };

        let byte_len = dim_us.checked_mul(std::mem::size_of::<f64>());
        if byte_len.is_none() {
            return -3i32;
        }
        let byte_len = byte_len.unwrap();

        // Alignment check
        let align = std::mem::align_of::<f64>();
        if (x as usize) % align != 0 || (v as usize) % align != 0 || (out as usize) % align != 0 {
            return -4i32;
        }

        // Global alias check
        let total_bytes = dim_us.checked_mul(batch_us).and_then(|n| n.checked_mul(std::mem::size_of::<f64>()));
        if total_bytes.is_none() {
            return -3i32;
        }
        let total_bytes = total_bytes.unwrap();

        let x_addr = x as usize;
        let v_addr = v as usize;
        let out_addr = out as usize;

        let x_end = x_addr.saturating_add(total_bytes);
        let v_end = v_addr.saturating_add(total_bytes);
        let out_end = out_addr.saturating_add(total_bytes);

        if v_addr < out_end && out_addr < v_end {
            return 1i32;
        }
        if x_addr < out_end && out_addr < x_end {
            return 2i32;
        }

        for b in 0..batch_us {
            let x_ptr = x.add(b * dim_us);
            let v_ptr = v.add(b * dim_us);
            let out_ptr = out.add(b * dim_us);

            // Intra-batch alias check
            let v_b_addr = v_ptr as usize;
            let out_b_addr = out_ptr as usize;
            let v_b_end = v_b_addr.saturating_add(byte_len);
            let out_b_end = out_b_addr.saturating_add(byte_len);

            if v_b_addr < out_b_end && out_b_addr < v_b_end {
                return 1i32;
            }

            let x_b_addr = x_ptr as usize;
            let x_b_end = x_b_addr.saturating_add(byte_len);
            if x_b_addr < out_b_end && out_b_addr < x_b_end {
                return 2i32;
            }

            // Crear slices SOLO después de todas las verificaciones
            let xb = std::slice::from_raw_parts(x_ptr, dim_us);
            let vb = std::slice::from_raw_parts(v_ptr, dim_us);
            let ob = std::slice::from_raw_parts_mut(out_ptr, dim_us);

            let mut v_max = 0.0_f64;
            for &val in vb.iter() {
                let av = val.abs();
                if av > v_max {
                    v_max = av;
                }
            }

            if v_max < 1e-30 {
                ob.copy_from_slice(xb);
                continue;
            }

            // Pre-calcular inverso
            let inv_v_max = 1.0 / v_max;

            let mut v_sq = 0.0_f64;
            let mut xv_dot = 0.0_f64;
            for i in 0..dim_us {
                let vn = vb[i] * inv_v_max;
                v_sq += vn * vn;
                xv_dot += xb[i] * vn;
            }

            let factor = (2.0 * xv_dot / v_sq) * inv_v_max;
            for i in 0..dim_us {
                ob[i] = xb[i] - factor * vb[i];
            }
        }

        0i32
    });

    match result {
        Ok(ret) => ret,
        Err(_) => -2, // Panic contenido
    }
}
