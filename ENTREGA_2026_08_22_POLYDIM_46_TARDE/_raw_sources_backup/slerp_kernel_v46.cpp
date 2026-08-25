#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

#if defined(USE_HW_INTERFERENCE) || defined(__cpp_lib_hardware_interference_size)
    #include <new>
    constexpr size_t CACHE_LINE = std::hardware_destructive_interference_size;
#else
    constexpr size_t CACHE_LINE = 128;
#endif

// AVX2-optimized dot product with FMA and error compensation
inline double kahan_dot_simd(const double* p, const double* q, size_t D) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    // Acumulación FMA estándar (altamente estable en Float64)
    for (; i + 8 <= D; i += 8) {
        __m256d p0 = _mm256_loadu_pd(p + i);
        __m256d q0 = _mm256_loadu_pd(q + i);
        __m256d p1 = _mm256_loadu_pd(p + i + 4);
        __m256d q1 = _mm256_loadu_pd(q + i + 4);
        
        sum_vec = _mm256_fmadd_pd(p0, q0, sum_vec);
        sum_vec = _mm256_fmadd_pd(p1, q1, sum_vec);
    }
    
    // Reducción horizontal con Kahan secuencial
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    double sum = 0.0, c = 0.0;
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    // Resto del array
    for (; i < D; ++i) {
        double y = (p[i] * q[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

inline double norm_sq_simd(const double* p, size_t D) {
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d c_vec = _mm256_setzero_pd();
    
    size_t i = 0;
    for (; i + 8 <= D; i += 8) {
        __m256d p0 = _mm256_loadu_pd(p + i);
        __m256d p1 = _mm256_loadu_pd(p + i + 4);
        
        __m256d prod0 = _mm256_mul_pd(p0, p0);
        __m256d prod1 = _mm256_mul_pd(p1, p1);
        
        __m256d y0 = _mm256_sub_pd(prod0, c_vec);
        __m256d t0 = _mm256_add_pd(sum_vec, y0);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t0, sum_vec), y0);
        sum_vec = t0;
        
        __m256d y1 = _mm256_sub_pd(prod1, c_vec);
        __m256d t1 = _mm256_add_pd(sum_vec, y1);
        c_vec = _mm256_sub_pd(_mm256_sub_pd(t1, sum_vec), y1);
        sum_vec = t1;
    }
    
    double sum = 0.0, c = 0.0;
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    for (int k = 0; k < 4; ++k) {
        double y = temp[k] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    for (; i < D; ++i) {
        double y = (p[i] * p[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

inline double kahan_diff_norm(const double* p, const double* q, size_t D) {
    double norm_sq = 0.0, c = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double diff = p[i] - q[i];
        double y = (diff * diff) - c;
        double t = norm_sq + y;
        c = (t - norm_sq) - y;
        norm_sq = t;
    }
    return std::sqrt(norm_sq);
}

inline double kahan_sum_norm(const double* p, const double* q, size_t D) {
    double norm_sq = 0.0, c = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double sum = p[i] + q[i];
        double y = (sum * sum) - c;
        double t = norm_sq + y;
        c = (t - norm_sq) - y;
        norm_sq = t;
    }
    return std::sqrt(norm_sq);
}

#include <cassert>

void fused_det_tangent(const double* p, double* v, size_t D, double eps) {
    if (D == 0) return;
    
#ifndef NDEBUG
    double p_norm_sq = 0.0;
    for (size_t i = 0; i < D; ++i) p_norm_sq += p[i] * p[i];
    double p_norm = std::sqrt(std::max(0.0, p_norm_sq));
    assert(std::abs(p_norm - 1.0) < 100.0 * eps && "fused_det_tangent requiere vector unitario");
#endif

    size_t min_idx = 0;
    double min_val = std::abs(p[0]);
    for (size_t i = 1; i < D; ++i) {
        double val = std::abs(p[i]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }
    
    double dot_e_p = p[min_idx];
    double v_norm_sq = 0.0;
    double c = 0.0;
    
    for (size_t i = 0; i < D; ++i) {
        double val = (i == min_idx ? 1.0 : 0.0) - dot_e_p * p[i];
        v[i] = val;
        
        double y = (val * val) - c;
        double t = v_norm_sq + y;
        c = (t - v_norm_sq) - y;
        v_norm_sq = t;
    }
    
    if (v_norm_sq < eps * eps) {
        size_t alt_idx = (min_idx + 1) % D;
        double dot_e_alt = p[alt_idx];
        v_norm_sq = 0.0;
        c = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double val = (i == alt_idx ? 1.0 : 0.0) - dot_e_alt * p[i];
            v[i] = val;
            double y = (val * val) - c;
            double t = v_norm_sq + y;
            c = (t - v_norm_sq) - y;
            v_norm_sq = t;
        }
    }
    
    // FIX: Clamp negativo antes de sqrt
    v_norm_sq = std::max(0.0, v_norm_sq);
    double v_norm = std::sqrt(v_norm_sq);
    double inv_norm = 1.0 / (v_norm + eps);
    for (size_t i = 0; i < D; ++i) {
        v[i] *= inv_norm;
    }
}

extern "C" {
// PARCHE 3: ZERO-ALLOCATION REAL (El llamador provee 'scratch' de tamaño D * sizeof(double))
void slerp(const double* p, const double* q, double t, double* out, double* scratch, size_t D, size_t scratch_size) {
    if (D == 0) return;
    
    // Protección de memoria: Si el caller de Python mintió, abortamos.
    if (scratch_size < D) {
        // En producción, rellenar out con NaNs para que JAX lo detecte
        for (size_t i = 0; i < D; ++i) out[i] = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    double eps = std::numeric_limits<double>::epsilon();
    double small_threshold = 16.0 * eps;
    double antipodal_threshold = std::max(100.0 * eps, std::sqrt(eps));
    
    double d_norm = kahan_diff_norm(p, q, D);
    double s_norm = kahan_sum_norm(p, q, D);
    double omega = 2.0 * std::atan2(d_norm, s_norm);
    
    // 1. Small angle regime
    if (omega < small_threshold) {
        double norm_sq = 0.0, c = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double val = p[i] + t * (q[i] - p[i]);
            out[i] = val;
            double y = (val * val) - c;
            double sum_t = norm_sq + y;
            c = (sum_t - norm_sq) - y;
            norm_sq = sum_t;
        }
        double norm = std::sqrt(norm_sq);
        if (norm < eps) {
            for (size_t i = 0; i < D; ++i) out[i] = p[i];
            return;
        }
        for (size_t i = 0; i < D; ++i) out[i] /= norm;
        return;
    }
    
    // 2. Antipodal regime (ZERO ALLOCATION: usa el scratch provisto)
    if ((std::acos(-1.0) - omega) < antipodal_threshold) {
        fused_det_tangent(p, scratch, D, eps); // <-- SCRATCH EXTERNO
        double cos_t_pi = std::cos(t * std::acos(-1.0));
        double sin_t_pi = std::sin(t * std::acos(-1.0));
        
        double norm_sq = 0.0, c = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double val = p[i] * cos_t_pi + scratch[i] * sin_t_pi;
            out[i] = val;
            double y = (val * val) - c;
            double sum_t = norm_sq + y;
            c = (sum_t - norm_sq) - y;
            norm_sq = sum_t;
        }
        double norm = std::sqrt(norm_sq);
        if (norm < eps) {
            for (size_t i = 0; i < D; ++i) out[i] = p[i];
            return;
        }
        for (size_t i = 0; i < D; ++i) out[i] /= norm;
        return;
    }
    
    // 3. Normal regime
    double sin_omega = std::sin(omega);
    
    // Fix #4: Guarda contra subnormales en sin_omega
    if (std::abs(sin_omega) < eps) {
        // Fallback seguro a interpolación lineal normalizada
        double norm_sq = 0.0, c = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double val = p[i] + t * (q[i] - p[i]);
            out[i] = val;
            double y = (val * val) - c;
            double sum_t = norm_sq + y;
            c = (sum_t - norm_sq) - y;
            norm_sq = sum_t;
        }
        double norm = std::sqrt(norm_sq);
        if (norm < eps) { for (size_t i = 0; i < D; ++i) out[i] = p[i]; return; }
        for (size_t i = 0; i < D; ++i) out[i] /= norm;
        return;
    }

    double s0 = std::sin((1.0 - t) * omega) / sin_omega;
    double s1 = std::sin(t * omega) / sin_omega;
    
    double norm_sq = 0.0, c = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double val = s0 * p[i] + s1 * q[i];
        out[i] = val;
        double y = (val * val) - c;
        double sum_t = norm_sq + y;
        c = (sum_t - norm_sq) - y;
        norm_sq = sum_t;
    }
    double norm = std::sqrt(norm_sq);
    for (size_t i = 0; i < D; ++i) out[i] /= norm;
}
}