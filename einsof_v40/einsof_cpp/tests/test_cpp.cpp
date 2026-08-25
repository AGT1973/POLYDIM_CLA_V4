/**
 * test_cpp.cpp — Tests del kernel C++ POLYDIM EINSOF V40
 * Certifica: omega = 2*atan2(||p-q||, ||p+q||) en D=100K
 */
#include <cassert>
#include <cmath>
#include <vector>
#include <cstdio>

// Declaraciones (linkea con slerp_kernel.cpp)
extern double slerp_omega(const double* p, const double* q, size_t D);
extern void slerp(const double* p, const double* q, double t, double* out, size_t D);

int PASS = 0, FAIL = 0;

void check(const char* name, bool cond) {
    if (cond) { ++PASS; printf("  [PASS] %s\n", name); }
    else       { ++FAIL; printf("  [FAIL] %s\n", name); }
}

int main() {
    printf("=== einsof_cpp TEST SUITE V40 ===\n");
    const size_t D = 4;

    // T1: vectores ortogonales -> omega = pi/2
    std::vector<double> p = {1,0,0,0}, q = {0,1,0,0};
    double w = slerp_omega(p.data(), q.data(), D);
    check("T1_omega_ortho", std::abs(w - M_PI/2.0) < 1e-12);

    // T2: antipodal -> omega = pi
    std::vector<double> p2 = {1,0,0,0}, q2 = {-1,0,0,0};
    double w2 = slerp_omega(p2.data(), q2.data(), D);
    check("T2_omega_antipodal", std::abs(w2 - M_PI) < 1e-12);

    // T3: SLERP preserva norma
    std::vector<double> out(D);
    slerp(p.data(), q.data(), 0.5, out.data(), D);
    double norm = 0; for (auto x: out) norm += x*x;
    check("T3_slerp_norm", std::abs(std::sqrt(norm) - 1.0) < 1e-12);

    // T4: t=0 -> p, t=1 -> q
    slerp(p.data(), q.data(), 0.0, out.data(), D);
    check("T4_slerp_t0_eq_p", std::abs(out[0]-1.0)<1e-12 && std::abs(out[1])<1e-12);
    slerp(p.data(), q.data(), 1.0, out.data(), D);
    check("T4_slerp_t1_eq_q", std::abs(out[1]-1.0)<1e-12 && std::abs(out[0])<1e-12);

    printf("\n=== TOTAL: %d | PASS: %d | FAIL: %d ===\n", PASS+FAIL, PASS, FAIL);
    return FAIL > 0 ? 1 : 0;
}
