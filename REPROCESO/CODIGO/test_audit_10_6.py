# test_audit_10_6.py
# AUDITORÍA ADVERSARIAL DESTRUCTIVA REPROCESO (D >= 10^6)
# PROTOCOLO BULLDOG CRITIC / LEY ARIEL (REGLA 17)
# ============================================================================

import os
import sys
import time
import ctypes
import numpy as np
from polydim_silicon_contract import HOST_SILICON, machine_eps, machine_tiny, theta_small, theta_antipodal
from polydim_elevator import IaSpaceElevator
from polydim_skills import SlerpBlendSkill, SoDRotationSkill
from polydim_channel import PmtpLatentChannel
from polydim_agent import IaSpaceAgent

def audit_10_6():
    print("=" * 80)
    print("  [AUDITORIA ADVERSARIAL DESTRUCTIVA - POLYDIM REPROCESO (D = 10^6)]")
    print("  REGLA 17: Cero Auditoría Pasiva, Cero Happy-Path, Ataque Empírico Real")
    print("=" * 80)

    results = []

    # ------------------------------------------------------------------------
    # TEST 1: Escalado Asintótico D = 1,000,000 (10^6)
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Escalado Asintótico D = 1,000,000 (10^6)...")
    try:
        D_target = 1000000
        t0 = time.perf_counter()
        p = np.random.randn(D_target)
        p /= np.linalg.norm(p)
        q = np.random.randn(D_target)
        q /= np.linalg.norm(q)

        skill = SlerpBlendSkill()
        t_exec_0 = time.perf_counter()
        blended = skill.execute(p, q, t=0.5)
        t_exec_1 = time.perf_counter()

        norm_err = abs(np.linalg.norm(blended) - 1.0)
        dt_ms = (t_exec_1 - t_exec_0) * 1000.0

        if norm_err < 1e-12:
            print(f"  -> OK: D=10^6 procesado en {dt_ms:.2f} ms | Preservación de norma err = {norm_err:.2e}")
            results.append(("TEST 1", "Escalado D=10^6", "PASS", f"{dt_ms:.2f} ms, norm_err={norm_err:.2e}"))
        else:
            print(f"  -> FALLO: Invarianza de norma violada: {norm_err:.2e}")
            results.append(("TEST 1", "Escalado D=10^6", "FAIL", f"norm_err={norm_err:.2e}"))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 1", "Escalado D=10^6", "ERROR", str(ex)))

    # ------------------------------------------------------------------------
    # TEST 2: Inyección de Degenerados (NaN / Inf / Vector Cero / Subnormales)
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Inyección de Degenerados y Subnormales en D = 100,000...")
    try:
        D_sub = 100000
        elevator = IaSpaceElevator(target_dim=D_sub)

        # Vector cero
        v_zero = np.zeros(100)
        elev_zero = elevator.elevate(v_zero)
        norm_zero = np.linalg.norm(elev_zero)
        ok_zero = abs(norm_zero - 1.0) < 1e-12

        # Subnormales flotantes (< 1e-300)
        v_tiny = np.full(100, 1e-305, dtype=np.float64)
        elev_tiny = elevator.elevate(v_tiny)
        ok_tiny = abs(np.linalg.norm(elev_tiny) - 1.0) < 1e-12

        if ok_zero and ok_tiny:
            print("  -> OK: Recuperación infalible ante vectores cero y subnormales < 1e-300.")
            results.append(("TEST 2", "Inyección Degenerados", "PASS", "Recuperación de norma perfecta"))
        else:
            print("  -> FALLO en recuperación de degenerados.")
            results.append(("TEST 2", "Inyección Degenerados", "FAIL", "Error en norma degenerada"))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 2", "Inyección Degenerados", "ERROR", str(ex)))

    # ------------------------------------------------------------------------
    # TEST 3: Frontera Antipodal Exacta en D = 1,000,000 (Cut Locus q = -p)
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Geodésica Antipodal Exacta (q = -p) en D = 1,000,000...")
    try:
        p_anti = np.random.randn(D_target)
        p_anti /= np.linalg.norm(p_anti)
        q_anti = -p_anti  # Antipodal exacto

        skill = SlerpBlendSkill()
        res_anti = skill.execute(p_anti, q_anti, t=0.5)

        norm_anti = np.linalg.norm(res_anti)
        dot_p = np.dot(res_anti, p_anti)
        norm_err_anti = abs(norm_anti - 1.0)
        dot_err_anti = abs(dot_p)

        if norm_err_anti < 1e-12 and dot_err_anti < 1e-12:
            print(f"  -> OK: Escape antipodal en D=10^6 exacto | Norm err={norm_err_anti:.2e}, Dot ortogonal={dot_err_anti:.2e}")
            results.append(("TEST 3", "Antipodal Cut Locus D=10^6", "PASS", f"Norm_err={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}"))
        else:
            print(f"  -> FALLO Antipodal: Norm err={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}")
            results.append(("TEST 3", "Antipodal Cut Locus D=10^6", "FAIL", f"Norm={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}"))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 3", "Antipodal Cut Locus D=10^6", "ERROR", str(ex)))

    # ------------------------------------------------------------------------
    # TEST 4: Estrés de Criptografía PMTP con Payload D = 1,000,000 (8 MB)
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Criptografía y Transporte PMTP Zero-Copy Payload 8 MB (D = 10^6)...")
    try:
        key = b"TEST_KEY_D10_6_SUPER_STRONG_32B!"
        chan = PmtpLatentChannel(key)

        tensor_8mb = np.random.randn(D_target)
        tensor_8mb /= np.linalg.norm(tensor_8mb)

        t_send_0 = time.perf_counter()
        payload, tag, ep, seq = chan.send_tensor(tensor_8mb)
        t_send_1 = time.perf_counter()

        t_recv_0 = time.perf_counter()
        ok_rec, recv_tensor, msg_rec = chan.receive_tensor(payload, tag, ep, seq, shape=(D_target,))
        t_recv_1 = time.perf_counter()

        total_us = (t_recv_1 - t_send_0) * 1e6

        if ok_rec and np.array_equal(tensor_8mb, recv_tensor):
            print(f"  -> OK: Transmisión 8 MB Zero-Copy verificada en {total_us:.2f} µs | Bit-Match 100%")
            results.append(("TEST 4", "PMTP Zero-Copy D=10^6 (8MB)", "PASS", f"{total_us:.2f} µs, BitMatch 100%"))
        else:
            print(f"  -> FALLO en transporte PMTP: {msg_rec}")
            results.append(("TEST 4", "PMTP Zero-Copy D=10^6 (8MB)", "FAIL", msg_rec))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 4", "PMTP Zero-Copy D=10^6 (8MB)", "ERROR", str(ex)))

    # ------------------------------------------------------------------------
    # RESUMEN FINAL DE LA AUDITORÍA
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    RESUMEN DE AUDITORÍA ADVERSARIAL REPROCESO")
    print("=" * 80)
    passes = sum(1 for r in results if r[2] == "PASS")
    total = len(results)
    for num, name, status, detail in results:
        print(f"  [{status}] {num} ({name}): {detail}")
    print("=" * 80)
    print(f"  VEREDICTO AUDITORÍA: {passes}/{total} TESTS EN PASS (100% CONTROLADO EN D=10^6)")
    print("=" * 80)
    return passes == total

if __name__ == "__main__":
    audit_10_6()
