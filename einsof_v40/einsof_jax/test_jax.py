"""test_jax.py — Suite de pruebas einsof_jax. Certificada con los 15 CHKs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np, math
from einsof_jax.slerp import slerp_stable
from einsof_jax.tsqr import tsqr_blocked, cholesky_qr2, ortho_gap
from einsof_jax.stiefel import stiefel_drift_check

PASS = 0; FAIL = 0

def check(name, cond, info=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else:     FAIL += 1; print(f"  [FAIL] {name} {info}")

print("=== einsof_jax TEST SUITE V40 ===")

# T1: SLERP preserva norma
p = np.random.randn(50_000); p /= np.linalg.norm(p)
q = np.random.randn(50_000); q /= np.linalg.norm(q)
mid = slerp_stable(p, q, 0.5)
check("T1_slerp_norm", abs(np.linalg.norm(mid)-1.0) < 1e-10)

# T2: SLERP antipodal sin NaN
q_anti = -p
mid2 = slerp_stable(p, q_anti, 0.5)
check("T2_antipodal_no_nan", not np.isnan(mid2).any() and abs(np.linalg.norm(mid2)-1.0)<1e-10)

# T3: SLERP casi paralelos (theta ~ 0)
noise = np.random.randn(50_000)*1e-10
q_near = p + noise; q_near /= np.linalg.norm(q_near)
mid3 = slerp_stable(p, q_near, 0.5)
check("T3_slerp_near_parallel", abs(np.linalg.norm(mid3)-1.0) < 1e-10)

# T4: TSQR ortogonalidad
A = np.random.randn(100_000, 8)
Q, R = tsqr_blocked(A, block_size=10_000)
gap, tol, _ = ortho_gap(Q)
check("T4_tsqr_orthogonal", gap < tol * 10, f"gap={gap:.2e} tol={tol:.2e}")

# T5: CholeskyQR2 bien condicionado
A2 = np.random.randn(50_000, 8)
Q2 = cholesky_qr2(A2)
gap2, tol2, _ = ortho_gap(Q2)
check("T5_cholesky_qr2", gap2 < 1e-10, f"gap={gap2:.2e}")

# T6: CholeskyQR2 mal condicionado -> fallback TSQR
A3 = np.random.randn(50_000, 8); A3[:, -1] *= 1e-8
Q3 = cholesky_qr2(A3)
gap3, tol3, _ = ortho_gap(Q3)
check("T6_ill_cond_fallback", gap3 < 1e-10, f"gap={gap3:.2e}")

# T7: drift sqrt(N)
res = stiefel_drift_check(D=5_000, N=10_000)
check("T7_drift_regime", res["drift_regime"] == "sqrt(N)", str(res))

print(f"\n=== TOTAL: {PASS+FAIL} | PASS: {PASS} | FAIL: {FAIL} ===")
