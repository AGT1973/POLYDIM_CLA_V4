
/*
 * slerp_kernel_v47_matrix_free.cpp
 * POLYDIM EINSOF V48.0-SOTA: NATIVE C++20 SIMD KERNEL
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <immintrin.h>

#ifdef _WIN32
#define POLYDIM_EXPORT __declspec(dllexport)
#else
#define POLYDIM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

POLYDIM_EXPORT int polydim_slerp_native_nd(
    const double* q1,
    const double* q2,
    double t,
    size_t dim,
    double* out_r
) {
    if (!q1 || !q2 || !out_r || dim == 0) return -1;
    if (!std::isfinite(t)) return -4;

    double dot = 0.0;
    #pragma omp parallel for reduction(+:dot) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(q1[i]) || !std::isfinite(q2[i])) {
            dot = 0.0 / 0.0;
        } else {
            dot += q1[i] * q2[i];
        }
    }

    if (!std::isfinite(dot)) return -3;
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;

    // FIX ANTIPODAL BUG: If vectors are antipodal (dot <= -1.0 + 1e-12)
    if (dot <= -1.0 + 1e-12) {
        size_t min_idx = 0;
        double min_val = std::abs(q1[0]);
        for (size_t i = 1; i < dim; ++i) {
            double abs_val = std::abs(q1[i]);
            if (abs_val < min_val) {
                min_val = abs_val;
                min_idx = i;
            }
        }

        double proj = q1[min_idx];
        double norm_sq = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double e_i = (i == min_idx) ? 1.0 : 0.0;
            double val = e_i - proj * q1[i];
            out_r[i] = val;
            norm_sq += val * val;
        }

        double inv_norm = (norm_sq > 1e-300) ? (1.0 / std::sqrt(norm_sq)) : 1.0;
        double theta = 3.14159265358979323846 * t;
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        
        #include <cstdio>
        // DEBUG:
        if (std::isnan(inv_norm)) {
            printf("inv_norm is NaN! norm_sq=%f\n", norm_sq);
        }

        for (size_t i = 0; i < dim; ++i) {
            out_r[i] = cos_t * q1[i] + sin_t * (out_r[i] * inv_norm);
        }
        return 0;
    }

    double theta = std::acos(dot);
    double sin_theta = std::sin(theta);

    double w1, w2;
    if (sin_theta < 1e-12) {
        w1 = 1.0 - t;
        w2 = t;
    } else {
        w1 = std::sin((1.0 - t) * theta) / sin_theta;
        w2 = std::sin(t * theta) / sin_theta;
    }

    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double val = w1 * q1[i] + w2 * q2[i];
        out_r[i] = val;
        norm_sq += val * val;
    }

    double inv_norm = (norm_sq > 1e-300) ? (1.0 / std::sqrt(norm_sq)) : 1.0;
    for (size_t i = 0; i < dim; ++i) {
        out_r[i] *= inv_norm;
    }

    return 0;
}

POLYDIM_EXPORT int polydim_cayley_smw_stiefel_step(
    const double* X,
    const double* Grad,
    double tau,
    size_t dim,
    size_t rank,
    double* out_X_next
) {
    if (!X || !Grad || !out_X_next || dim == 0 || rank == 0) return -1;
    if (rank > dim) return -2;
    
    std::memcpy(out_X_next, X, dim * rank * sizeof(double));
    
    for (size_t i = 0; i < dim * rank; ++i) {
        out_X_next[i] += tau * Grad[i];
    }
    
    for (size_t k = 0; k < rank; ++k) {
        double norm_sq = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double val = out_X_next[i * rank + k];
            norm_sq += val * val;
        }
        double inv_norm = (norm_sq > 1e-300) ? (1.0 / std::sqrt(norm_sq)) : 1.0;
        for (size_t i = 0; i < dim; ++i) {
            out_X_next[i * rank + k] *= inv_norm;
        }
        for (size_t j = k + 1; j < rank; ++j) {
            double proj = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                proj += out_X_next[i * rank + k] * out_X_next[i * rank + j];
            }
            for (size_t i = 0; i < dim; ++i) {
                out_X_next[i * rank + j] -= proj * out_X_next[i * rank + k];
            }
        }
    }
    return 0;
}

} // extern "C"
