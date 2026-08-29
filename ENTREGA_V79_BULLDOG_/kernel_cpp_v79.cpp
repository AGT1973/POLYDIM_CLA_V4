// ============================================================================
// POLYDIM V79 BULLDOG - NATIVE C++ KERNELS (LEY ARIEL COMPLIANT)
// ============================================================================
// Zero-Waste: O(D) operations, full alias safety, Kahan precision, 
// no integer overflow, SIMD-aligned independent.
// ============================================================================
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <utility>

#if defined(_MSC_VER)
    #define POLYDIM_EXPORT extern "C" __declspec(dllexport)
#else
    #define POLYDIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace polydim {

inline bool check_byte_overlap(const void* ptrA, const void* ptrB, size_t bytes) noexcept {
    uintptr_t a = reinterpret_cast<uintptr_t>(ptrA);
    uintptr_t b = reinterpret_cast<uintptr_t>(ptrB);
    uintptr_t diff = (a > b) ? (a - b) : (b - a);
    return diff < bytes;
}

inline int solve_4x4_2rhs(double A[4][4], double B[4][2]) noexcept {
    for (int col = 0; col < 4; ++col) {
        int best = col;
        double best_val = std::abs(A[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            double v = std::abs(A[r][col]);
            if (v > best_val) { best_val = v; best = r; }
        }
        if (best_val < 1e-12) return -1;
        if (best != col) {
            for (int j = 0; j < 4; ++j) std::swap(A[col][j], A[best][j]);
            for (int j = 0; j < 2; ++j) std::swap(B[col][j], B[best][j]);
        }
        for (int r = col + 1; r < 4; ++r) {
            double f = A[r][col] / A[col][col];
            for (int j = col + 1; j < 4; ++j) A[r][j] -= f * A[col][j];
            A[r][col] = 0.0;
            for (int j = 0; j < 2; ++j) B[r][j] -= f * B[col][j];
        }
    }
    for (int r = 3; r >= 0; --r) {
        for (int j = 0; j < 2; ++j) {
            for (int c = r + 1; c < 4; ++c) B[r][j] -= A[r][c] * B[c][j];
            B[r][j] /= A[r][r];
        }
    }
    return 0;
}

} // namespace polydim

// ---- Householder reflection (batched, kahan-precision, alias-safe) ----

POLYDIM_EXPORT int polydim_householder_reflect_cpp(
    const double* __restrict x,
    const double* __restrict v,
    double* __restrict out,
    uint64_t dim,
    uint64_t batch) noexcept
{
    if (!x || !v || !out || dim == 0 || batch == 0) return -1;
    
    // Checked arithmetic for pointer overflow
    if (batch > SIZE_MAX / dim) return -2;
    size_t total_elements = static_cast<size_t>(batch) * static_cast<size_t>(dim);
    if (total_elements > SIZE_MAX / sizeof(double)) return -2;
    size_t total_bytes = total_elements * sizeof(double);

    // Global aliasing check. If x == out, it's a valid in-place operation.
    if (x != out && polydim::check_byte_overlap(x, out, total_bytes)) return 1;
    if (polydim::check_byte_overlap(v, out, total_bytes)) return 1;
    // We don't modify x and v, but check if they overlap with each other out of paranoia
    if (v != x && polydim::check_byte_overlap(x, v, total_bytes)) return 1;

    const double eps = std::numeric_limits<double>::epsilon();
    size_t d = static_cast<size_t>(dim);

    for (size_t b = 0; b < static_cast<size_t>(batch); ++b) {
        size_t offset = b * d;
        const double* xb = x + offset;
        const double* vb = v + offset;
        double* ob = out + offset;

        double v_sq = 0.0, c_v = 0.0;
        double xv = 0.0, c_xv = 0.0;

        #pragma clang loop vectorize(enable) interleave(enable)
        #pragma clang loop vectorize_predicate(enable)
        for (size_t i = 0; i < d; ++i) {
            double v_val = vb[i];
            double x_val = xb[i];
            
            double y_v = (v_val * v_val) - c_v;
            double t_v = v_sq + y_v;
            c_v = (t_v - v_sq) - y_v;
            v_sq = t_v;
            
            double y_xv = (x_val * v_val) - c_xv;
            double t_xv = xv + y_xv;
            c_xv = (t_xv - xv) - y_xv;
            xv = t_xv;
        }

        if (v_sq < eps * 10.0) {
            if (ob != xb) {
                std::memmove(ob, xb, d * sizeof(double));
            }
            continue;
        }
        
        double factor = 2.0 * xv / v_sq;
        
        #pragma clang loop vectorize(enable) interleave(enable)
        #pragma clang loop vectorize_predicate(enable)
        for (size_t i = 0; i < d; ++i) {
            ob[i] = xb[i] - factor * vb[i];
        }
    }
    return 0;
}

// ---- Cayley-SMW retraction on St(D, 2) ----

POLYDIM_EXPORT int polydim_cayley_retract_k2_cpp(
    const double* __restrict X, 
    const double* __restrict G, 
    double* __restrict Out,     
    uint64_t dim,
    double alpha) noexcept
{
    if (!X || !G || !Out || dim == 0) return -1;
    
    if (dim > SIZE_MAX / 2) return -2;
    size_t total_elements = static_cast<size_t>(dim) * 2;
    if (total_elements > SIZE_MAX / sizeof(double)) return -2;
    size_t byte_len = total_elements * sizeof(double);

    if (X != Out && polydim::check_byte_overlap(X, Out, byte_len)) return 1;
    if (polydim::check_byte_overlap(G, Out, byte_len)) return 1;

    if (std::abs(alpha) < 1e-30) {
        if (Out != X) {
            std::memmove(Out, X, byte_len);
        }
        return 0;
    }

    double a2 = 0.5 * alpha;
    size_t D = static_cast<size_t>(dim);

    double VtU[4][4] = {};
    double VtX[4][2] = {};

    for (size_t d = 0; d < D; ++d) {
        double x0 = X[d * 2], x1 = X[d * 2 + 1];
        double g0 = G[d * 2], g1 = G[d * 2 + 1];
        double vc[4] = {x0, x1, -g0, -g1};
        double uc[4] = {g0, g1, x0, x1};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) VtU[i][j] += vc[i] * uc[j];
            VtX[i][0] += vc[i] * x0;
            VtX[i][1] += vc[i] * x1;
        }
    }

    double C[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            C[i][j] = (i == j ? 1.0 : 0.0) + a2 * VtU[i][j];

    double RHS[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
            double s = 0.0;
            for (int m = 0; m < 4; ++m) s += VtU[i][m] * VtX[m][j];
            RHS[i][j] = VtX[i][j] - a2 * s;
        }

    if (polydim::solve_4x4_2rhs(C, RHS) != 0) return -3;

    for (size_t d = 0; d < D; ++d) {
        double x0 = X[d * 2], x1 = X[d * 2 + 1];
        double g0 = G[d * 2], g1 = G[d * 2 + 1];
        double uc[4] = {g0, g1, x0, x1};
        for (int j = 0; j < 2; ++j) {
            double corr = 0.0;
            for (int m = 0; m < 4; ++m)
                corr += uc[m] * (VtX[m][j] + RHS[m][j]);
            Out[d * 2 + j] = X[d * 2 + j] - a2 * corr;
        }
    }
    return 0;
}
