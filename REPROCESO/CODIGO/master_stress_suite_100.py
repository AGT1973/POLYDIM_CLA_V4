# master_stress_suite_100.py
# SUITE DE PRUEBAS MAESTRA Y ADVERSARIAL DE 100 CONTROLES COMPLETOS (POLYDIM REPROCESO)
# PROTOCOLO BULLDOG CRITIC / LEY ARIEL (REGLA 17 - CERO AUDITORÍA PASIVA)
# ============================================================================

import os
import sys
import time
import ctypes
import threading
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)

from polydim_silicon_contract import (
    HOST_SILICON, machine_eps, machine_tiny, theta_small, theta_antipodal, check_memory_available
)
from polydim_elevator import IaSpaceElevator
from polydim_skills import SlerpBlendSkill, SoDRotationSkill, StiefelProjectionSkill, SinkhornOtSkill
from polydim_channel import PmtpLatentChannel, PmtpStatefulReceiver
from polydim_agent import IaSpaceAgent
from polydim_terminal import TerminalCollapser

def run_master_100_suite():
    print("=" * 80)
    print("  [SUITE MAESTRA ADVERSARIAL DE 100 PRUEBAS EMPÍRICAS INTEGRALES (POLYDIM V48)]")
    print("  REGLA 17: Cero Auditoría Pasiva - Verificación de 100 Checks Reales en Silicio")
    print("=" * 80)

    results = []
    
    def check(num: str, name: str, fn):
        try:
            ok, msg = fn()
            status = "PASS" if ok else "FAIL"
            results.append((num, name, status, msg))
        except Exception as ex:
            results.append((num, name, "ERROR", str(ex)))

    # -------------------------------------------------------------------------
    # BLOQUE 1: CAYLEY-SMW MATRIX-FREE Y MATEMÁTICA DENSA ORÁCULO (CHK_01..CHK_20)
    # -------------------------------------------------------------------------

    def make_cayley_test(D, K):
        def _fn():
            np.random.seed(42 + D + K)
            U = np.random.randn(D, K)
            V = np.random.randn(D, K)
            x = np.random.randn(D); x /= np.linalg.norm(x)

            # Oráculo Denso: Q = (I - 0.5 W)^(-1) (I + 0.5 W)
            W = U @ V.T - V @ U.T
            I_D = np.eye(D)
            Q_dense = np.linalg.solve(I_D - 0.5 * W, I_D + 0.5 * W)
            x_dense = Q_dense @ x

            # Cayley-SMW Exacto: x + Y @ (I_2K - 0.5 J G)^(-1) @ (J Y^T x)
            Y = np.hstack([U, V])
            G = Y.T @ Y
            I_2k = np.eye(2 * K)
            J = np.block([[np.zeros((K,K)), np.eye(K)], [-np.eye(K), np.zeros((K,K))]])

            M = I_2k - 0.5 * J @ G
            rhs = J @ (Y.T @ x)
            coeff = np.linalg.solve(M, rhs)
            x_smw_raw = x + Y @ coeff

            diff = np.linalg.norm(x_smw_raw - x_dense)
            raw_norm_err = abs(np.linalg.norm(x_smw_raw) - 1.0)
            return (diff < 1e-12) and (raw_norm_err < 1e-12), f"Cayley Oracle D={D},K={K} diff={diff:.2e}, raw_norm_err={raw_norm_err:.2e}"
        return _fn

    for idx, (d_val, k_val) in enumerate([
        (10, 2), (20, 4), (50, 5), (100, 8), (200, 10), (500, 16), (1000, 16),
        (20, 2), (40, 4), (80, 8), (100, 10), (300, 12), (600, 16), (800, 16),
        (15, 3), (25, 4), (75, 5), (150, 8), (250, 10), (400, 12)
    ], start=1):
        check(f"CHK_{idx:02d}", f"Cayley-SMW vs Oracle D={d_val},K={k_val}", make_cayley_test(d_val, k_val))

    # -------------------------------------------------------------------------
    # BLOQUE 2: GEOMETRÍA SLERP Y ANTIPODALIDAD EXACTA NO-NaN (CHK_21..CHK_40)
    # -------------------------------------------------------------------------

    def chk_21():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        res = SlerpBlendSkill().execute(p, p, 0.5)
        err = np.linalg.norm(res - p)
        return err < 1e-12, f"Idempotencia err={err:.2e}"
    check("CHK_21", "SLERP Idempotencia", chk_21)

    def chk_22():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r1 = SlerpBlendSkill().execute(p, q, 0.3)
        r2 = SlerpBlendSkill().execute(q, p, 0.7)
        err = np.linalg.norm(r1 - r2)
        return err < 1e-12, f"Simetría err={err:.2e}"
    check("CHK_22", "SLERP Simetría", chk_22)

    def chk_23():
        p = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q = -p
        res = SlerpBlendSkill().execute(p, q, 0.5)
        nrm_err = abs(np.linalg.norm(res) - 1.0)
        has_nan = np.isnan(res).any()
        dot_err = abs(np.dot(res, p))
        return (not has_nan) and nrm_err < 1e-12 and dot_err < 1e-12, f"Antipodal no-NaN norm_err={nrm_err:.2e}, dot_err={dot_err:.2e}"
    check("CHK_23", "Antipodalidad Exacta SLERP (No-NaN)", chk_23)

    for idx, d_slerp in enumerate([10, 100, 1000, 5000, 10000, 50000, 100000], start=24):
        def _slerp_check(dim=d_slerp):
            p = np.random.randn(dim); p /= np.linalg.norm(p)
            q = np.random.randn(dim); q /= np.linalg.norm(q)
            res = SlerpBlendSkill().execute(p, q, 0.5)
            err = abs(np.linalg.norm(res) - 1.0)
            return err < 1e-12, f"SLERP D={dim} norm_err={err:.2e}"
        check(f"CHK_{idx:02d}", f"SLERP Asintótico D={d_slerp}", _slerp_check)

    for idx in range(31, 41):
        def _rot_check(dim=idx*100):
            p = np.random.randn(dim); p /= np.linalg.norm(p)
            q = np.random.randn(dim); q /= np.linalg.norm(q)
            res = SoDRotationSkill().execute(p, q, theta=0.1)
            err = abs(np.linalg.norm(res) - 1.0)
            return err < 1e-12, f"SO(D) Rotación D={dim} norm_err={err:.2e}"
        check(f"CHK_{idx:02d}", f"SO(D) Rotación D={idx*100}", _rot_check)

    # -------------------------------------------------------------------------
    # BLOQUE 3: PROTOCOLO PMTP, SEQLOCK Y SEGURIDAD CRIPTOGRÁFICA (CHK_41..CHK_60)
    # -------------------------------------------------------------------------

    def chk_41():
        key = b"TEST_MASTER_KEY_32BYTES_CHK_41!"
        rec = PmtpStatefulReceiver(key)
        payload = b"TEST_PAYLOAD"
        epoch_key = rec._derive_epoch_key(1)
        tag = rec._make_tag(1, 1, payload, epoch_key)
        ok, msg = rec.verify_and_accept(1, 1, payload, tag)
        return ok, f"PMTP Paquete Válido ({msg})"
    check("CHK_41", "PMTP Aceptación Paquete Válido", chk_41)

    def chk_42():
        key = b"TEST_MASTER_KEY_32BYTES_CHK_42!"
        rec = PmtpStatefulReceiver(key)
        payload = b"TEST_PAYLOAD"
        tag_bad = b"\x00" * 64
        ok, msg = rec.verify_and_accept(1, 1, payload, tag_bad)
        return (not ok) and (msg == "CORRUPT_TAG"), f"PMTP Tag Corrupto Rechazo ({msg})"
    check("CHK_42", "PMTP Tag Corrupto Rechazo", chk_42)

    def chk_43():
        key = b"TEST_MASTER_KEY_32BYTES_CHK_43!"
        rec = PmtpStatefulReceiver(key)
        payload = b"TEST_PAYLOAD"
        epoch_key = rec._derive_epoch_key(1)
        tag1 = rec._make_tag(1, 1, payload, epoch_key)
        rec.verify_and_accept(1, 1, payload, tag1)
        ok2, msg2 = rec.verify_and_accept(1, 1, payload, tag1)
        return (not ok2) and ("REPLAY" in msg2), f"PMTP Replay Rechazo ({msg2})"
    check("CHK_43", "PMTP Replay Rechazo", chk_43)

    for idx in range(44, 61):
        def _pmtp_seq_check(seq_num=idx):
            key = f"TEST_MASTER_KEY_32BYTES_CHK_{seq_num}!".encode('utf-8')[:32]
            rec = PmtpStatefulReceiver(key)
            payload = f"PAYLOAD_{seq_num}".encode('utf-8')
            ek = rec._derive_epoch_key(1)
            tag = rec._make_tag(1, seq_num, payload, ek)
            ok, msg = rec.verify_and_accept(1, seq_num, payload, tag)
            return ok, f"PMTP Seq={seq_num} ({msg})"
        check(f"CHK_{idx:02d}", f"PMTP Transmisión Secuencial Seq={idx}", _pmtp_seq_check)

    # -------------------------------------------------------------------------
    # BLOQUE 4: VARIEDAD DE STIEFEL, SINKHORN Y CONCURRENCIA (CHK_61..CHK_80)
    # -------------------------------------------------------------------------

    def chk_61():
        p = np.random.randn(4096); p /= np.linalg.norm(p)
        skill = StiefelProjectionSkill()
        proj, r_coords = skill.execute(p, K=128)
        norm_err = abs(np.linalg.norm(proj) - 1.0)
        return norm_err < 1e-12 and r_coords.size == 128, f"Stiefel D=4096 -> K=128 norm_err={norm_err:.2e}"
    check("CHK_61", "Stiefel Projection D=4096 -> K=128", chk_61)

    def chk_62():
        p = np.random.randn(200); p /= np.linalg.norm(p)
        q = np.random.randn(200); q /= np.linalg.norm(q)
        skill = SinkhornOtSkill()
        aligned = skill.execute(p, q, reg=0.1)
        norm_err = abs(np.linalg.norm(aligned) - 1.0)
        dist_before = np.linalg.norm(p - q)
        dist_after = np.linalg.norm(aligned - q)
        return norm_err < 1e-12 and (dist_after <= dist_before + 1e-12), f"Sinkhorn OT dist_before={dist_before:.4f} -> dist_after={dist_after:.4f}"
    check("CHK_62", "Sinkhorn OT Alignment D=200", chk_62)

    def chk_63():
        key = b"TEST_MASTER_KEY_32BYTES_CHK_63!"
        rec = PmtpStatefulReceiver(key)
        errors = []
        
        def worker(thread_idx):
            for i in range(20):
                seq = thread_idx * 100 + i + 1
                payload = f"PAYLOAD_{thread_idx}_{i}".encode('utf-8')
                ek = rec._derive_epoch_key(1)
                tag = rec._make_tag(1, seq, payload, ek)
                ok, msg = rec.verify_and_accept(1, seq, payload, tag)
                if not ok:
                    errors.append((thread_idx, seq, msg))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        return len(errors) == 0, f"Thread-Safety Multi-Hilo ({len(errors)} errores)"
    check("CHK_63", "PMTP Concurrencia Multi-Hilo", chk_63)

    for idx in range(64, 81):
        def _stiefel_range_check(k_dim=idx):
            p = np.random.randn(1000); p /= np.linalg.norm(p)
            skill = StiefelProjectionSkill()
            proj, r_coords = skill.execute(p, K=k_dim)
            return abs(np.linalg.norm(proj) - 1.0) < 1e-12, f"Stiefel K={k_dim} OK"
        check(f"CHK_{idx:02d}", f"Stiefel Projection Variabilidad K={idx}", _stiefel_range_check)

    # -------------------------------------------------------------------------
    # BLOQUE 5: SILICON CONTRACT, AGENTES Y DEGENERADOS (CHK_81..CHK_100)
    # -------------------------------------------------------------------------

    def chk_81():
        eps_64 = machine_eps(np.float64)
        tiny_64 = machine_tiny(np.float64)
        th_sm = theta_small(np.float64, 10000)
        th_anti = theta_antipodal(np.float64, 10000)
        ok = eps_64 < 1e-15 and tiny_64 < 1e-300 and th_sm > 0 and th_anti > 0
        return ok, f"Silicon Contract eps={eps_64:.2e}, tiny={tiny_64:.2e}"
    check("CHK_81", "Silicon Contract Interrogación Dinámica", chk_81)

    def chk_82():
        agent_a = IaSpaceAgent("Agent_A", dim=5000)
        goal = np.random.randn(5000); goal /= np.linalg.norm(goal)
        agent_a.set_goal(goal)
        d0 = agent_a.geodesic_distance_to_goal()
        d1, status = agent_a.deliberate_step(step_size=0.3)
        return d1 < d0, f"IaSpaceAgent d0={d0:.4f} -> d1={d1:.4f}"
    check("CHK_82", "IaSpaceAgent Deliberación Geodésica", chk_82)

    def chk_83():
        collapser = TerminalCollapser("Agent_Test")
        state = np.random.randn(1000); state /= np.linalg.norm(state)
        report = collapser.collapse_all(state, goal_dist=0.05)
        ok = ("NIVEL 1" in report) and ("NIVEL 2" in report) and ("NIVEL 3" in report) and ("NIVEL 4" in report)
        return ok, "Terminal Collapser 4 Niveles Pedagógicos"
    check("CHK_83", "Terminal Collapser 4 Niveles Pedagógicos", chk_83)

    for idx in range(84, 101):
        def _deg_check(num=idx):
            elevator = IaSpaceElevator(target_dim=500)
            zero_v = np.zeros(num)
            res = elevator.elevate(zero_v)
            has_nan = np.isnan(res).any()
            err = abs(np.linalg.norm(res) - 1.0)
            return (not has_nan) and err < 1e-12, f"Degenerate zero-vector len={num} OK"
        check(f"CHK_{idx:02d}", f"Recuperación Vector Cero Dim={idx}", _deg_check)

    # -------------------------------------------------------------------------
    # RESUMEN Y IMPRESIÓN DEL VEREDICTO DE LAS 100 PRUEBAS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("               RESUMEN SUITE MAESTRA 100 PRUEBAS EMPÍRICAS")
    print("=" * 80)
    passes = sum(1 for r in results if r[2] == "PASS")
    total = len(results)
    for num, name, status, detail in results:
        print(f"  [{status}] {num} ({name}): {detail}")
    print("=" * 80)
    print(f"  VEREDICTO GLOBAL: {passes}/{total} PRUEBAS PASADAS ({passes/total*100:.1f}%)")
    print("=" * 80)
    return passes == total

if __name__ == "__main__":
    run_master_100_suite()
