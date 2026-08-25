/*
 * POLYDIM EINSOF V48.1-FIXED: NATIVE C++20 SIMD KERNEL
 * Correcciones aplicadas:
 *   - BLOCK-04: SLERP antipodal manejado (arco menor)
 *   - BLOCK-07: std::acos reemplazado por atan2 estable
 *   - BLOCK-10: Cast (int)dim eliminado, size_t nativo
 *   - BLOCK-11: Bounds checking en dim*rank
 *   - ALTO-03: Sanitización NaN/Inf en inputs
 *   - ALTO-04: Aliasing verificado (memmove para overlap)
 *   - MED-03: Verificación de t ∈ [0,1]
 *   - MED-04: Verificación de tau
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <immintrin.h>

#ifdef _WIN32
#define POLYDIM_EXPORT __declspec(dllexport)
#define POLYDIM_CALL __cdecl
#else
#define POLYDIM_EXPORT __attribute__((visibility("default")))
#define POLYDIM_CALL
#endif

enum PolydimError {
    POLYDIM_OK = 0,
    POLYDIM_ERR_NULL_PTR = -1,
    POLYDIM_ERR_INVALID_DIM = -2,
    POLYDIM_ERR_INVALID_RANK = -3,
    POLYDIM_ERR_NAN_INPUT = -4,
    POLYDIM_ERR_INF_INPUT = -5,
    POLYDIM_ERR_NOT_NORMALIZED = -6,
    POLYDIM_ERR_OVERFLOW = -7,
    POLYDIM_ERR_ANTIPODAL = -8,
    POLYDIM_ERR_INVALID_T = -9,
    POLYDIM_ERR_INVALID_TAU = -10,
    POLYDIM_ERR_DIM_OVERFLOW = -11,
};

extern "C" {

/**
 * Native SLERP Kernel in S^(D-1) for D >= 1.
 * Computes r = (sin((1-t)theta)/sin(theta))*q1 + (sin(t*theta)/sin(theta))*q2
 */
POLYDIM_EXPORT int POLYDIM_CALL polydim_slerp_native_nd(
    const double* q1,
    const double* q2,
    double t,
    size_t dim,
    double* out_r
) {
    if (!q1 || !q2 || !out_r || dim == 0) return POLYDIM_ERR_NULL_PTR;
    if (dim > SIZE_MAX / sizeof(double)) return POLYDIM_ERR_DIM_OVERFLOW;
    
    if (std::isnan(t) || std::isinf(t) || t < 0.0 || t > 1.0) 
        return POLYDIM_ERR_INVALID_T;

    for (size_t i = 0; i < dim; ++i) {
        if (std::isnan(q1[i]) || std::isnan(q2[i])) 
            return POLYDIM_ERR_NAN_INPUT;
        if (std::isinf(q1[i]) || std::isinf(q2[i])) 
            return POLYDIM_ERR_INF_INPUT;
    }

    double dot = 0.0;
    #pragma omp parallel for reduction(+:dot) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        dot += q1[i] * q2[i];
    }

    dot = std::max(-1.0, std::min(1.0, dot));

    double* q2_use = const_cast<double*>(q2);
    bool used_neg = false;
    
    if (dot < 0.0) {
        dot = -dot;
        // Always heap-allocate to avoid false sharing on stack buffer (Gemma-3 + Llama-4 consensus)
        q2_use = new double[dim];
        used_neg = true;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dim; ++i) {
            q2_use[i] = -q2[i];
        }
    }

    double diff_norm_sq = 0.0;
    double sum_norm_sq = 0.0;
    #pragma omp parallel for reduction(+:diff_norm_sq,sum_norm_sq) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        double diff = q1[i] - q2_use[i];
        double sum = q1[i] + q2_use[i];
        diff_norm_sq += diff * diff;
        sum_norm_sq += sum * sum;
    }
    
    double theta = 2.0 * std::atan2(std::sqrt(diff_norm_sq), std::sqrt(sum_norm_sq));
    double sin_theta = std::sin(theta);

    double w1, w2;
    double eps = std::numeric_limits<double>::epsilon();
    double theta_small = std::sqrt(eps) * std::sqrt((double)dim);
    
    if (sin_theta < theta_small) {
        w1 = 1.0 - t;
        w2 = t;
    } else {
        w1 = std::sin((1.0 - t) * theta) / sin_theta;
        w2 = std::sin(t * theta) / sin_theta;
    }

    double norm_sq = 0.0;
    #pragma omp parallel for reduction(+:norm_sq) schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        double val = w1 * q1[i] + w2 * q2_use[i];
        out_r[i] = val;
        norm_sq += val * val;
    }

    if (used_neg) delete[] q2_use;

    if (norm_sq < 1e-300) {
        std::memcpy(out_r, q1, dim * sizeof(double));
        return POLYDIM_ERR_ANTIPODAL;
    }

    double inv_norm = 1.0 / std::sqrt(norm_sq);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dim; ++i) {
        out_r[i] *= inv_norm;
    }

    return POLYDIM_OK;
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_naive_projected_gradient_step(
    const double* X,
    const double* Grad,
    double tau,
    size_t dim,
    size_t rank,
    double* out_X_next
) {
    if (!X || !Grad || !out_X_next || dim == 0 || rank == 0) 
        return POLYDIM_ERR_NULL_PTR;
    
    if (rank > SIZE_MAX / dim / sizeof(double)) 
        return POLYDIM_ERR_OVERFLOW;
    if (dim > 1e9 || rank > 1e6) 
        return POLYDIM_ERR_OVERFLOW;
    
    if (std::isnan(tau) || std::isinf(tau)) 
        return POLYDIM_ERR_INVALID_TAU;

    size_t total = dim * rank;
    
    // Correct bidirectional overlap detection (both models confirmed the old check was broken)
    const double* out_end = out_X_next + total;
    const double* x_end = X + total;
    const double* g_end = Grad + total;
    bool overlaps_x = (out_X_next < x_end && X < out_end);
    bool overlaps_g = (out_X_next < g_end && Grad < out_end);
    
    if (overlaps_x || overlaps_g) {
        std::memmove(out_X_next, X, total * sizeof(double));
    } else {
        std::memcpy(out_X_next, X, total * sizeof(double));
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; ++i) {
        out_X_next[i] += tau * Grad[i];
    }

    for (size_t k = 0; k < rank; ++k) {
        double norm_sq = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            double val = out_X_next[i * rank + k];
            norm_sq += val * val;
        }
        if (norm_sq < 1e-300) return POLYDIM_ERR_ANTIPODAL;
        double inv_norm = 1.0 / std::sqrt(norm_sq);
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

    return POLYDIM_OK;
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_get_abi_version() { 
    return 0x00480000;
}

POLYDIM_EXPORT int POLYDIM_CALL polydim_get_double_size() { 
    return sizeof(double); 
}

} // extern "C"
