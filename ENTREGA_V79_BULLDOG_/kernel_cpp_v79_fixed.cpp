// ============================================================================
// POLYDIM V79 BULLDOG — NATIVE C++ GEODESIC & CLIFFORD KERNEL (CORREGIDO)
// ============================================================================
// Fixes:
//   - check_byte_overlap verifica punteros del batch actual (xb, vb, ob)
//   - Overflow-safe arithmetic con SIZE_MAX checks
//   - Alignment verification (alignof(double) == 8)
//   - Pre-calcular inv_v_max (sin división en hot loop)
//   - memset de out al inicio (previene uninitialized memory)
//   - Quitar __restrict para evitar UB con alias real
//   - size_t robusto para Windows LLP64
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#if defined(_MSC_VER)
    #define POLYDIM_EXPORT extern "C" __declspec(dllexport)
#else
    #define POLYDIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace polydim {
    // Robust size type: uint64_t en Windows 32-bit, size_t en 64-bit
    #if defined(_WIN32) && !defined(_WIN64)
        typedef uint64_t polydim_size_t;
    #else
        typedef size_t polydim_size_t;
    #endif

    inline bool check_byte_overlap(const void* ptrA, const void* ptrB, size_t bytes) noexcept {
        if (bytes == 0) return false;
        uintptr_t addrA = reinterpret_cast<uintptr_t>(ptrA);
        uintptr_t addrB = reinterpret_cast<uintptr_t>(ptrB);
        if (addrA == addrB) return true;
        // Overflow-safe
        if (addrA > std::numeric_limits<uintptr_t>::max() - bytes) return true;
        if (addrB > std::numeric_limits<uintptr_t>::max() - bytes) return true;
        return (addrA < addrB + bytes) && (addrB < addrA + bytes);
    }

    inline bool is_aligned(const void* ptr, size_t alignment) noexcept {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
}

POLYDIM_EXPORT int polydim_householder_reflect_cpp(
    const double* x,
    const double* v,
    double* out,
    uint64_t dim_u64,
    uint64_t batch_u64) noexcept
{
    // Validación de punteros
    if (!x || !v || !out) return -1;
    if (dim_u64 == 0 || batch_u64 == 0) return -1;

    // Overflow check
    if (batch_u64 > 0 && (std::numeric_limits<polydim::polydim_size_t>::max() / batch_u64) < dim_u64)
        return -3;

    polydim::polydim_size_t dim = static_cast<polydim::polydim_size_t>(dim_u64);
    polydim::polydim_size_t batch = static_cast<polydim::polydim_size_t>(batch_u64);
    size_t byte_len = dim * sizeof(double);
    size_t total_bytes = byte_len * batch;

    // Alignment check
    const size_t ALIGN = alignof(double);
    if (!polydim::is_aligned(x, ALIGN)) return -4;
    if (!polydim::is_aligned(v, ALIGN)) return -4;
    if (!polydim::is_aligned(out, ALIGN)) return -4;

    // Inicializar out (previene uninitialized memory)
    std::memset(out, 0, total_bytes);

    // Verificar alias global ANTES del loop (sin __restrict, el compilador
    // no asume independencia, pero verificamos por seguridad)
    if (polydim::check_byte_overlap(v, out, total_bytes)) return 1;
    if (polydim::check_byte_overlap(x, out, total_bytes)) return 2;

    for (polydim::polydim_size_t b = 0; b < batch; ++b) {
        const double* xb = x + b * dim;
        const double* vb = v + b * dim;
        double* ob = out + b * dim;

        // Verificar alias intra-batch (redundante con el global, pero defensivo)
        if (polydim::check_byte_overlap(vb, ob, byte_len)) return 1;
        if (polydim::check_byte_overlap(xb, ob, byte_len)) return 2;

        // Encontrar v_max
        double v_max = 0.0;
        for (polydim::polydim_size_t i = 0; i < dim; ++i) {
            double av = std::abs(vb[i]);
            if (av > v_max) v_max = av;
        }

        if (v_max < 1e-30) {
            std::memcpy(ob, xb, byte_len);
            continue;
        }

        // Pre-calcular inverso (evita división en hot loop)
        double inv_v_max = 1.0 / v_max;

        double v_sq = 0.0;
        double xv_dot = 0.0;
        for (polydim::polydim_size_t i = 0; i < dim; ++i) {
            double vn = vb[i] * inv_v_max;
            v_sq += vn * vn;
            xv_dot += xb[i] * vn;
        }

        double factor = (2.0 * xv_dot / v_sq) * inv_v_max;

        for (polydim::polydim_size_t i = 0; i < dim; ++i) {
            ob[i] = xb[i] - factor * vb[i];
        }
    }
    return 0;
}
