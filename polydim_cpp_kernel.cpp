
// POLYDIM V66 NATIVE C++20 AVX-512 KERNEL
#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

extern "C" {

static double scaled_norm_sq(const double* v, size_t dim, double* out_scale) {
    double scale = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double av = std::fabs(v[i]);
        if (av > scale) scale = av;
    }
    *out_scale = scale;
    if (scale == 0.0) return 0.0;
    double inv_scale = 1.0 / scale;
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double vi = v[i] * inv_scale;
        sum += vi * vi;
    }
    return sum;
}

#if defined(__AVX512F__)
__declspec(dllexport) int polydim_cpp_householder_reflect(const double* x, const double* v, double* out, size_t dim) {
    if (!x || !v || !out || dim == 0) return -1;
    double scale = 0.0;
    double vv_scaled = scaled_norm_sq(v, dim, &scale);
    if (scale == 0.0 || vv_scaled < 1e-30) {
        for (size_t j = 0; j < dim; ++j) out[j] = x[j];
        return 0;
    }
    double norm_v = scale * std::sqrt(vv_scaled);
    double alpha = 1.0 / norm_v;
    __m512d zmm_dot = _mm512_setzero_pd();
    __m512d zmm_alpha = _mm512_set1_pd(alpha);
    size_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m512d zmm_v = _mm512_loadu_pd(&v[i]);
        __m512d zmm_u = _mm512_mul_pd(zmm_v, zmm_alpha);
        __m512d zmm_x = _mm512_loadu_pd(&x[i]);
        zmm_dot = _mm512_fmadd_pd(zmm_u, zmm_x, zmm_dot);
    }
    double dot = _mm512_reduce_add_pd(zmm_dot);
    for (; i < dim; ++i) dot += (v[i] * alpha) * x[i];
    double two_dot = 2.0 * dot;
    i = 0;
    for (; i + 7 < dim; i += 8) {
        __m512d zmm_v = _mm512_loadu_pd(&v[i]);
        __m512d zmm_u = _mm512_mul_pd(zmm_v, zmm_alpha);
        __m512d zmm_x = _mm512_loadu_pd(&x[i]);
        __m512d zmm_y = _mm512_fnmadd_pd(zmm_two_dot, zmm_u, zmm_x);
        _mm512_storeu_pd(&out[i], zmm_y);
    }
    for (; i < dim; ++i) out[i] = x[i] - two_dot * (v[i] * alpha);
    return 0;
}
#else
__declspec(dllexport) int polydim_cpp_householder_reflect(const double* x, const double* v, double* out, size_t dim) {
    if (!x || !v || !out || dim == 0) return -1;
    double scale = 0.0;
    double vv_scaled = scaled_norm_sq(v, dim, &scale);
    if (scale == 0.0 || vv_scaled < 1e-30) {
        for (size_t i = 0; i < dim; ++i) out[i] = x[i];
        return 0;
    }
    double norm_v = scale * std::sqrt(vv_scaled);
    double alpha = 1.0 / norm_v;
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) dot += (v[i] * alpha) * x[i];
    double two_dot = 2.0 * dot;
    for (size_t i = 0; i < dim; ++i) out[i] = x[i] - two_dot * (v[i] * alpha);
    return 0;
}
#endif

__declspec(dllexport) double polydim_simd_kahan_dot_aligned(const double* A, const double* B, size_t D) {
    double sum = 0.0;
    double c = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double y = (A[i] * B[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__declspec(dllexport) double polydim_log_space_overlap(const double* A, const double* B, size_t D) {
    if (D == 0) return -INFINITY;
    for (size_t i = 0; i < D; ++i) {
        if (std::isnan(A[i]) || std::isnan(B[i])) return NAN;
    }
    double max_val = A[0] + B[0];
    for (size_t i = 1; i < D; ++i) {
        double val = A[i] + B[i];
        if (val > max_val) max_val = val;
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < D; ++i) {
        sum_exp += std::exp((A[i] + B[i]) - max_val);
    }
    return max_val + std::log(sum_exp);
}
}
