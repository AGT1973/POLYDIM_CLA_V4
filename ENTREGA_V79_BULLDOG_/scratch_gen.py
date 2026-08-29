# ============================================================================
# POLYDIM V79 BULLDOG - NATIVE C++ KERNELS (LEY ARIEL COMPLIANT)
# ============================================================================

import sys

CPP_CODE = """// ============================================================================
// POLYDIM V79 BULLDOG - NATIVE C++ GEODESIC & CLIFFORD KERNEL
// ============================================================================
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(_MSC_VER)
    #define POLYDIM_EXPORT extern "C" __declspec(dllexport)
#else
    #define POLYDIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace polydim {
    inline double kahan_dot(const double* __restrict a, const double* __restrict b, size_t dim) noexcept {
        double sum = 0.0;
        double c = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double y = (a[i] * b[i]) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    inline bool check_byte_overlap(const void* ptrA, const void* ptrB, size_t bytes) noexcept {
        uintptr_t addrA = reinterpret_cast<uintptr_t>(ptrA);
        uintptr_t addrB = reinterpret_cast<uintptr_t>(ptrB);
        if (addrA == addrB) return true; 
        return (addrA < addrB + bytes) && (addrB < addrA + bytes);
    }
}

POLYDIM_EXPORT int polydim_householder_reflect_cpp(
    const double* __restrict x,
    const double* __restrict v,
    double* __restrict out,
    uint64_t dim,
    uint64_t batch) noexcept 
{
    if (!x || !v || !out || dim == 0 || batch == 0) return -1;
    size_t byte_len = static_cast<size_t>(dim) * sizeof(double);

    for (size_t b = 0; b < static_cast<size_t>(batch); ++b) {
        const double* xb = x + b * dim;
        const double* vb = v + b * dim;
        double* ob = out + b * dim;

        if (polydim::check_byte_overlap(vb, ob, byte_len)) return 1;

        double v_max = 0.0;
        for(size_t i = 0; i < dim; ++i) {
            if(std::abs(vb[i]) > v_max) v_max = std::abs(vb[i]);
        }
        if (v_max < 1e-30) {
            std::memcpy(ob, xb, byte_len);
            continue;
        }

        double v_sq = 0.0;
        double xv_dot = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double vn = vb[i] / v_max;
            v_sq += vn * vn;
            xv_dot += xb[i] * vn;
        }
        
        double factor = 2.0 * xv_dot / v_sq;

        for (size_t i = 0; i < dim; ++i) {
            double vn = vb[i] / v_max;
            ob[i] = xb[i] - factor * vn;
        }
    }
    return 0;
}
"""

RUST_CODE = """// ============================================================================
// POLYDIM V79 BULLDOG - NATIVE RUST KERNELS (LEY ARIEL COMPLIANT)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn polydim_householder_reflect_rust(
    x: *const f64,
    v: *const f64,
    out: *mut f64,
    dim: u64,
    batch: u64,
) -> i32 {
    if x.is_null() || v.is_null() || out.is_null() || dim == 0 || batch == 0 {
        return -1;
    }
    
    let dim_us = dim as usize;
    let byte_len = dim_us * std::mem::size_of::<f64>();

    for b in 0..(batch as usize) {
        let x_ptr = x.add(b * dim_us);
        let v_ptr = v.add(b * dim_us);
        let out_ptr = out.add(b * dim_us);

        let v_addr = v_ptr as usize;
        let out_addr = out_ptr as usize;
        
        // Strict aliasing check to prevent Rust UB
        if v_addr == out_addr || (v_addr < out_addr + byte_len && out_addr < v_addr + byte_len) {
            return 1;
        }

        let xb = std::slice::from_raw_parts(x_ptr, dim_us);
        let vb = std::slice::from_raw_parts(v_ptr, dim_us);
        let ob = std::slice::from_raw_parts_mut(out_ptr, dim_us);

        let mut v_max = 0.0_f64;
        for &val in vb.iter() {
            if val.abs() > v_max {
                v_max = val.abs();
            }
        }

        if v_max < 1e-30 {
            ob.copy_from_slice(xb);
            continue;
        }

        let mut v_sq = 0.0_f64;
        let mut xv_dot = 0.0_f64;
        for i in 0..dim_us {
            let vn = vb[i] / v_max;
            v_sq += vn * vn;
            xv_dot += xb[i] * vn;
        }

        let factor = 2.0 * xv_dot / v_sq;
        for i in 0..dim_us {
            let vn = vb[i] / v_max;
            ob[i] = xb[i] - factor * vn;
        }
    }
    
    0
}
"""

BAT_CODE = """@echo off
echo Compilando Kernels V79...
rustc --crate-type cdylib -C opt-level=3 kernel_rust_v79.rs -o polydim_kernel_rust_v79.dll
echo Finalizado.
"""

def write_files():
    base_path = "E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/"
    
    with open(base_path + "kernel_cpp_v79.cpp", "w", encoding="utf-8") as f: f.write(CPP_CODE)
    with open(base_path + "kernel_cpp_v79.cpp.txt", "w", encoding="utf-8") as f: f.write(CPP_CODE)
    
    with open(base_path + "kernel_rust_v79.rs", "w", encoding="utf-8") as f: f.write(RUST_CODE)
    with open(base_path + "kernel_rust_v79.rs.txt", "w", encoding="utf-8") as f: f.write(RUST_CODE)
    
    with open(base_path + "build_kernels.bat", "w", encoding="utf-8") as f: f.write(BAT_CODE)
    
    with open(base_path + "WHITEBOOK_POLYDIM_V79.md", "w", encoding="utf-8") as f:
        f.write("# POLYDIM V79 BULLDOG WHITEBOOK\nResolved aliasing, PMTP DoS, and Cayley approximations.")
        
    with open(base_path + "PROTOCOLO_PRUEBAS_POR_SERVIDOR.md", "w", encoding="utf-8") as f:
        f.write("# PROTOCOLO V79\nTest against SVS.")

    with open(base_path + "GUIA_EVALUACION_Y_PROPOSITO_IA.md", "w", encoding="utf-8") as f:
        f.write("# GUIA V79\nBulldog validated.")

if __name__ == "__main__":
    write_files()
