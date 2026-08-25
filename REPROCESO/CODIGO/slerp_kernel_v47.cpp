// slerp_kernel_v47.cpp
// Kernel C++20/C++17 de Computabilidad Geométrica SOTA en S^(D-1) para POLYDIM V47
// Cero Alocaciones Dinámicas en el Heap (Zero Heap Hot-Loop)

#include <cmath>
#include <cstddef>
#include <limits>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

#ifdef _WIN32
  #define POLYDIM_API __declspec(dllexport)
#else
  #define POLYDIM_API __attribute__((visibility("default")))
#endif

// Static thread-local buffer para reporte de errores FFI seguro sin memory leaks
static thread_local char g_last_error[256] = {0};

extern "C" POLYDIM_API const char* get_last_error_safe() {
    return g_last_error;
}

static inline void set_last_error(const char* msg) {
    if (msg) {
        strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    } else {
        g_last_error[0] = '\0';
    }
}

// Norma L2 al cuadrado con compensación Kahan en SIMD AVX2/FMA
static inline double norm_sq_simd(const double* p, size_t D) {
    size_t i = 0;
    double sum = 0.0;
    double c = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();

    for (; i + 4 <= D; i += 4) {
        __m256d p_vec = _mm256_loadu_pd(p + i);
        __m256d prod = _mm256_mul_pd(p_vec, p_vec);

        __m256d y = _mm256_sub_pd(prod, c_vec);
        __m256d t = _mm256_add_pd(sum_vec, y);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t, sum_vec), y);
        sum_vec = t;
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif

    for (; i < D; ++i) {
        double y = (p[i] * p[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return std::max(0.0, sum);
}

// Norma L2 del vector diferencia ||p - q|| con compensación Kahan SIMD
static inline double kahan_diff_norm(const double* p, const double* q, size_t D) {
    size_t i = 0;
    double sum = 0.0;
    double c = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();

    for (; i + 4 <= D; i += 4) {
        __m256d p_vec = _mm256_loadu_pd(p + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d diff = _mm256_sub_pd(p_vec, q_vec);
        __m256d prod = _mm256_mul_pd(diff, diff);

        __m256d y = _mm256_sub_pd(prod, c_vec);
        __m256d t = _mm256_add_pd(sum_vec, y);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t, sum_vec), y);
        sum_vec = t;
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif

    for (; i < D; ++i) {
        double diff = p[i] - q[i];
        double y = (diff * diff) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return std::sqrt(std::max(0.0, sum));
}

// Norma L2 del vector suma ||p + q|| con compensación Kahan SIMD
static inline double kahan_sum_norm(const double* p, const double* q, size_t D) {
    size_t i = 0;
    double sum = 0.0;
    double c = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();

    for (; i + 4 <= D; i += 4) {
        __m256d p_vec = _mm256_loadu_pd(p + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d s_vec = _mm256_add_pd(p_vec, q_vec);
        __m256d prod = _mm256_mul_pd(s_vec, s_vec);

        __m256d y = _mm256_sub_pd(prod, c_vec);
        __m256d t = _mm256_add_pd(sum_vec, y);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t, sum_vec), y);
        sum_vec = t;
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
#endif

    for (; i < D; ++i) {
        double s_val = p[i] + q[i];
        double y = (s_val * s_val) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return std::sqrt(std::max(0.0, sum));
}

// Generación de tangente determinista ortogonal en C++ para el régimen antipodal
// Teorema: v = e_min - p_min * p siempre satisface ||v||^2 >= 1 - 1/D >= 0.5 para todo D >= 2.
static inline void fused_det_tangent(const double* p, double* v, size_t D) {
    if (D < 2) {
        if (D == 1) v[0] = 0.0;
        return;
    }

    size_t min_idx = 0;
    double min_val = std::abs(p[0]);
    for (size_t i = 1; i < D; ++i) {
        double val = std::abs(p[i]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    double p_min = p[min_idx];
    double norm_sq = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double val = (i == min_idx ? 1.0 : 0.0) - p_min * p[i];
        v[i] = val;
        norm_sq += val * val;
    }

    double inv_norm = 1.0 / std::sqrt(std::max(0.5, norm_sq));
    for (size_t i = 0; i < D; ++i) {
        v[i] *= inv_norm;
    }
}

// Función principal SLERP en C++ SOTA (Zero Heap Hot-Loop)
extern "C" POLYDIM_API int slerp(
    const double* p,
    const double* q,
    double t,
    double* out,
    size_t D,
    double* scratch,
    size_t scratch_size
) {
    if (!p || !q || !out || !scratch) {
        set_last_error("Null pointer argument in slerp");
        return -1;
    }
    if (D < 2) {
        set_last_error("Dimension D must be >= 2");
        return -2;
    }
    if (scratch_size < D) {
        set_last_error("Scratch buffer size must be >= D");
        return -3;
    }
    if (!std::isfinite(t)) {
        set_last_error("Interpolation parameter t must be finite");
        return -4;
    }

    static const double PI = 3.14159265358979323846;
    double eps = std::numeric_limits<double>::epsilon();
    double sqrt_D = std::sqrt(static_cast<double>(D));
    double small_threshold = 16.0 * eps * sqrt_D;
    double antipodal_threshold = std::max(100.0 * eps * sqrt_D, std::sqrt(eps));

    double d_norm = kahan_diff_norm(p, q, D);
    double s_norm = kahan_sum_norm(p, q, D);
    double omega = 2.0 * std::atan2(d_norm, s_norm);

    // 1. Régimen Colineal (casi idénticos)
    if (omega < small_threshold) {
        for (size_t i = 0; i < D; ++i) {
            out[i] = p[i] + t * (q[i] - p[i]);
        }
        double nrm = std::sqrt(norm_sq_simd(out, D));
        double inv_nrm = 1.0 / (nrm + std::numeric_limits<double>::epsilon());
        for (size_t i = 0; i < D; ++i) out[i] *= inv_nrm;
        return 0;
    }

    // 2. Régimen Antipodal (opuestos) -> Tangente Canónica e_min - p_min * p
    if ((PI - omega) < antipodal_threshold) {
        fused_det_tangent(p, scratch, D);
        double cos_t_pi = std::cos(t * PI);
        double sin_t_pi = std::sin(t * PI);
        for (size_t i = 0; i < D; ++i) {
            out[i] = p[i] * cos_t_pi + scratch[i] * sin_t_pi;
        }
        double nrm = std::sqrt(norm_sq_simd(out, D));
        double inv_nrm = 1.0 / (nrm + std::numeric_limits<double>::epsilon());
        for (size_t i = 0; i < D; ++i) out[i] *= inv_nrm;
        return 0;
    }

    // 3. Régimen Normal (SLERP geodésico)
    double sin_omega = std::sin(omega);
    double safe_sin = (std::abs(sin_omega) < eps) ? eps : sin_omega;
    double s0 = std::sin((1.0 - t) * omega) / safe_sin;
    double s1 = std::sin(t * omega) / safe_sin;

    for (size_t i = 0; i < D; ++i) {
        out[i] = s0 * p[i] + s1 * q[i];
    }
    double nrm = std::sqrt(norm_sq_simd(out, D));
    double inv_nrm = 1.0 / (nrm + std::numeric_limits<double>::epsilon());
    for (size_t i = 0; i < D; ++i) out[i] *= inv_nrm;

    return 0;
}
