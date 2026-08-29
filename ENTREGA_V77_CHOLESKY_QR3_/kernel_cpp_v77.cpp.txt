#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>

#ifdef _WIN32
#define EXPORT_SYM __declspec(dllexport)
#else
#define EXPORT_SYM __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// XLA Legacy Custom Call Signature (CPU)
EXPORT_SYM void polydim_cpp_householder_xla(double* out, const void** in) {
    // Escudo 1: Null-Pointer Guards
    if (!out || !in || !in[0] || !in[1] || !in[2]) return;

    const double* x = reinterpret_cast<const double*>(in[0]);
    const double* v = reinterpret_cast<const double*>(in[1]);
    const uint64_t* dim_ptr = reinterpret_cast<const uint64_t*>(in[2]);
    
    // Escudo 2: Verificación de Alineación SIMD
    if (reinterpret_cast<uintptr_t>(out) % alignof(double) != 0 ||
        reinterpret_cast<uintptr_t>(x) % alignof(double) != 0 ||
        reinterpret_cast<uintptr_t>(v) % alignof(double) != 0 ||
        reinterpret_cast<uintptr_t>(dim_ptr) % alignof(uint64_t) != 0) {
        return;
    }

    size_t dim = static_cast<size_t>(*dim_ptr);
    if (dim == 0) return;
    
    // Escudo 3: Rechazo absoluto de aliasing parcial (Capa 4 - Vectorización SIMD)
    if (x != out && ((x < out + dim) && (out < x + dim))) return;
    
    double v_max = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double abs_v = std::abs(v[i]);
        if (abs_v > v_max) v_max = abs_v;
    }
    
    if (v_max < 1e-30) {
        if (x != out) {
            for (size_t i = 0; i < dim; ++i) out[i] = x[i];
        }
        return;
    }
    
    double v_norm_sq = 0.0;
    double dot_xv = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double v_scaled = v[i] / v_max;
        v_norm_sq += v_scaled * v_scaled;
        dot_xv += x[i] * v_scaled;
    }
    
    double scale = 2.0 * dot_xv / v_norm_sq;
    for (size_t i = 0; i < dim; ++i) {
        out[i] = x[i] - scale * (v[i] / v_max);
    }
}

#ifdef __cplusplus
}
#endif
