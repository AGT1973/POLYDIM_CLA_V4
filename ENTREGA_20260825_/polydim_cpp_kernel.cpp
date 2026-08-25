
// POLYDIM V65 NATIVE C++20 AVX-512 KERNEL
// FIX: Norma escalada anti-overflow para v[i]^2
#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

extern "C" {

// Utilidad: norma escalada para evitar overflow en ||v||^2
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

// Parche P1: Householder Reflection u = v / ||v|| con norma escalada
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

    // Producto escalar dot = u^T x
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

    // Output y = x - 2 * dot * u
    double two_dot = 2.0 * dot;
    __m512d zmm_two_dot = _mm512_set1_pd(two_dot);
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

// SILICON CONTRACT: IEEE-754 estricto
#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#else
#pragma GCC push_options
#pragma GCC optimize ("-O3, -fno-fast-math")
#endif

__declspec(dllexport) double polydim_simd_kahan_dot_aligned(const double* __restrict A, const double* __restrict B, size_t D) {
#if defined(__AVX512F__)
    __m512d sum = _mm512_setzero_pd();
    __m512d c   = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + 7 < D; i += 8) {
        __m512d a = _mm512_load_pd(&A[i]);
        __m512d b = _mm512_load_pd(&B[i]);
        __m512d prod = _mm512_mul_pd(a, b);

        __m512d y = _mm512_sub_pd(prod, c);
        __m512d t = _mm512_add_pd(sum, y);
        __m512d temp = _mm512_sub_pd(t, sum);
        c = _mm512_sub_pd(temp, y);
        sum = t;
    }

    alignas(64) double sum_arr[8];
    alignas(64) double c_arr[8];
    _mm512_store_pd(sum_arr, sum);
    _mm512_store_pd(c_arr, c);

    double final_sum = 0.0;
    double final_c = 0.0;
    for (int j = 0; j < 8; ++j) {
        double val = sum_arr[j] - c_arr[j];
        double y = val - final_c;
        double t = final_sum + y;
        final_c = (t - final_sum) - y;
        final_sum = t;
    }

    for (; i < D; ++i) {
        double y = (A[i] * B[i]) - final_c;
        double t = final_sum + y;
        final_c = (t - final_sum) - y;
        final_sum = t;
    }
    return final_sum;
#else
    double sum = 0.0;
    double c = 0.0;
    for (size_t i = 0; i < D; ++i) {
        double y = (A[i] * B[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
#endif
}

__declspec(dllexport) double polydim_log_space_overlap(const double* A, const double* B, size_t D) {
    if (D == 0) return -INFINITY;

    // FIX: NaN check
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
