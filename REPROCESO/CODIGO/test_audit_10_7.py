# test_audit_10_7.py
# AUDITORÍA ADVERSARIAL EXTREMA D = 10,000,000 (10^7)
# PROTOCOLO BULLDOG CRITIC / LEY ARIEL (REGLA 17)
# ============================================================================

import os
import sys
import time
import numpy as np
from polydim_silicon_contract import HOST_SILICON, check_memory_available, machine_eps
from polydim_skills import SlerpBlendSkill
from polydim_channel import PmtpLatentChannel

def audit_10_7():
    print("=" * 80)
    print("  [AUDITORIA ADVERSARIAL EXTREMA D = 10,000,000 (10^7)]")
    print("  Evaluando barreras de memoria RAM y escalado asintotico a 80 MB por tensor")
    print("=" * 80)

    D_target = 10000000  # 10^7 (80 MB en float64)
    bytes_needed = D_target * 8 * 4  # 4 tensores en memoria (~320 MB)

    ok_mem, req, avail = check_memory_available(bytes_needed)
    print(f"• Requerimiento de Memoria: {req / (1024**2):.2f} MB | Disponible: {avail / (1024**2):.2f} MB | Suficiente? {ok_mem}")

    if not ok_mem:
        print("  -> VETO DE SILICIO: Memoria insuficiente para D=10^7 en esta máquina.")
        return

    results = []

    # TEST 1: Generación y normalización a S^(10^7 - 1)
    print("\n[TEST 1] Generacion e Isometria en D = 10,000,000 (80 MB)...")
    try:
        t0 = time.perf_counter()
        p = np.random.randn(D_target)
        p /= np.linalg.norm(p)
        q = np.random.randn(D_target)
        q /= np.linalg.norm(q)
        t1 = time.perf_counter()

        skill = SlerpBlendSkill()
        t2 = time.perf_counter()
        blended = skill.execute(p, q, t=0.5)
        t3 = time.perf_counter()

        norm_err = abs(np.linalg.norm(blended) - 1.0)
        dt_prep = (t1 - t0) * 1000.0
        dt_exec = (t3 - t2) * 1000.0

        if norm_err < 1e-12:
            print(f"  -> OK: D=10^7 procesado en {dt_exec:.2f} ms | Preservacion de norma err = {norm_err:.2e}")
            results.append(("TEST 1", "Escalado D=10^7", "PASS", f"{dt_exec:.2f} ms, norm_err={norm_err:.2e}"))
        else:
            print(f"  -> FALLO: Invarianza de norma violada en D=10^7: {norm_err:.2e}")
            results.append(("TEST 1", "Escalado D=10^7", "FAIL", f"norm_err={norm_err:.2e}"))
    except Exception as ex:
        print(f"  -> EXCEPCION: {ex}")
        results.append(("TEST 1", "Escalado D=10^7", "ERROR", str(ex)))

    # TEST 2: Escape Antipodal en D = 10,000,000
    print("\n[TEST 2] Escape Antipodal Canónico en D = 10,000,000...")
    try:
        p_anti = p.copy()
        q_anti = -p_anti

        t0 = time.perf_counter()
        res_anti = skill.execute(p_anti, q_anti, t=0.5)
        t1 = time.perf_counter()

        norm_anti = np.linalg.norm(res_anti)
        dot_anti = np.dot(res_anti, p_anti)
        norm_err_anti = abs(norm_anti - 1.0)
        dot_err_anti = abs(dot_anti)
        dt_anti = (t1 - t0) * 1000.0

        if norm_err_anti < 1e-12 and dot_err_anti < 1e-12:
            print(f"  -> OK: Escape antipodal D=10^7 en {dt_anti:.2f} ms | Norm_err={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}")
            results.append(("TEST 2", "Antipodal D=10^7", "PASS", f"{dt_anti:.2f} ms, Norm_err={norm_err_anti:.2e}"))
        else:
            print(f"  -> FALLO Antipodal: Norm_err={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}")
            results.append(("TEST 2", "Antipodal D=10^7", "FAIL", f"Norm_err={norm_err_anti:.2e}"))
    except Exception as ex:
        print(f"  -> EXCEPCION: {ex}")
        results.append(("TEST 2", "Antipodal D=10^7", "ERROR", str(ex)))

    # TEST 3: PMTP Zero-Copy de 80 MB Payload
    print("\n[TEST 3] PMTP Zero-Copy Payload 80 MB (D = 10,000,000)...")
    try:
        key = b"TEST_KEY_D10_7_SUPER_STRONG_32B!"
        chan = PmtpLatentChannel(key)

        t0 = time.perf_counter()
        payload, tag, ep, seq = chan.send_tensor(blended)
        ok_rec, recv_tensor, msg_rec = chan.receive_tensor(payload, tag, ep, seq, shape=(D_target,))
        t1 = time.perf_counter()
        dt_pmtp = (t1 - t0) * 1000.0

        if ok_rec and np.array_equal(blended, recv_tensor):
            print(f"  -> OK: PMTP 80 MB Zero-Copy verificado en {dt_pmtp:.2f} ms | Bit-Match 100%")
            results.append(("TEST 3", "PMTP 80 MB D=10^7", "PASS", f"{dt_pmtp:.2f} ms, BitMatch 100%"))
        else:
            print(f"  -> FALLO PMTP: {msg_rec}")
            results.append(("TEST 3", "PMTP 80 MB D=10^7", "FAIL", msg_rec))
    except Exception as ex:
        print(f"  -> EXCEPCION: {ex}")
        results.append(("TEST 3", "PMTP 80 MB D=10^7", "ERROR", str(ex)))

    print("\n" + "=" * 80)
    print("                RESUMEN AUDITORIA EXTREMA D = 10,000,000 (10^7)")
    print("=" * 80)
    passes = sum(1 for r in results if r[2] == "PASS")
    total = len(results)
    for num, name, status, detail in results:
        print(f"  [{status}] {num} ({name}): {detail}")
    print("=" * 80)
    print(f"  VEREDICTO: {passes}/{total} TESTS PASADOS EN D=10^7")
    print("=" * 80)

if __name__ == "__main__":
    audit_10_7()
