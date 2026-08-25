# test_audit_10_8.py
# AUDITORÍA ADVERSARIAL EXTREMA D = 100,000,000 (10^8 - 100 MILLONES DE DIMENSIONES)
# PROTOCOLO BULLDOG CRITIC / LEY ARIEL (REGLA 17)
# ============================================================================

import os
import sys
import time
import numpy as np
from polydim_silicon_contract import HOST_SILICON, check_memory_available, machine_eps
from polydim_skills import SlerpBlendSkill
from polydim_channel import PmtpLatentChannel

def audit_10_8():
    print("=" * 80)
    print("  [AUDITORIA ADVERSARIAL EXTREMA D = 100,000,000 (10^8)]")
    print("  Evaluando límites físicos de silicio: 800 MB por tensor Float64 (3.2 GB Total)")
    print("=" * 80)

    D_target = 100000000  # 10^8 (100 Millones de dimensiones)
    bytes_per_tensor = D_target * 8  # 800 MB
    total_bytes_needed = bytes_per_tensor * 4  # 3.2 GB para p, q, out, scratch

    print(f"• Dimensión Target: D = {D_target:,}")
    print(f"• Huella por Tensor Float64: {bytes_per_tensor / (1024**2):.2f} MB ({bytes_per_tensor / (1024**3):.2f} GB)")
    print(f"• Huella Total Operativa Requerida: {total_bytes_needed / (1024**3):.2f} GB")

    ok_mem, req, avail = check_memory_available(total_bytes_needed)
    print(f"• Memoria RAM Disponible en Host: {avail / (1024**3):.2f} GB | Suficiente? {ok_mem}")

    if not ok_mem:
        print("  -> VETO DE SILICIO: Memoria insuficiente para D=10^8 en esta máquina.")
        return False

    results = []

    # TEST 1: Generación y SLERP en D = 100,000,000 (10^8)
    print("\n[TEST 1] Generación e Isometría SLERP en D = 100,000,000 (800 MB)...")
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
        dt_exec = (t3 - t2) * 1000.0

        if norm_err < 1e-12:
            print(f"  -> OK: D=10^8 procesado en {dt_exec:.2f} ms | Preservación de norma err = {norm_err:.2e}")
            results.append(("TEST 1", "Escalado D=10^8", "PASS", f"{dt_exec:.2f} ms, norm_err={norm_err:.2e}"))
        else:
            print(f"  -> FALLO: Invarianza de norma violada en D=10^8: {norm_err:.2e}")
            results.append(("TEST 1", "Escalado D=10^8", "FAIL", f"norm_err={norm_err:.2e}"))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 1", "Escalado D=10^8", "ERROR", str(ex)))

    # TEST 2: Escape Antipodal Canónico en D = 100,000,000
    print("\n[TEST 2] Escape Antipodal Canónico en D = 100,000,000...")
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
            print(f"  -> OK: Escape antipodal D=10^8 en {dt_anti:.2f} ms | Norm_err={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}")
            results.append(("TEST 2", "Antipodal D=10^8", "PASS", f"{dt_anti:.2f} ms, Norm_err={norm_err_anti:.2e}"))
        else:
            print(f"  -> FALLO Antipodal: Norm_err={norm_err_anti:.2e}, Dot={dot_err_anti:.2e}")
            results.append(("TEST 2", "Antipodal D=10^8", "FAIL", f"Norm_err={norm_err_anti:.2e}"))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 2", "Antipodal D=10^8", "ERROR", str(ex)))

    # TEST 3: PMTP Zero-Copy de 800 MB Payload (D = 10^8)
    print("\n[TEST 3] PMTP Zero-Copy Payload 800 MB (D = 100,000,000)...")
    try:
        key = b"TEST_KEY_D10_8_SUPER_STRONG_32B!"
        chan = PmtpLatentChannel(key)

        t0 = time.perf_counter()
        payload, tag, ep, seq = chan.send_tensor(blended)
        ok_rec, recv_tensor, msg_rec = chan.receive_tensor(payload, tag, ep, seq, shape=(D_target,))
        t1 = time.perf_counter()
        dt_pmtp = (t1 - t0) * 1000.0

        if ok_rec and np.array_equal(blended, recv_tensor):
            print(f"  -> OK: PMTP 800 MB Zero-Copy verificado en {dt_pmtp:.2f} ms | Bit-Match 100%")
            results.append(("TEST 3", "PMTP 800 MB D=10^8", "PASS", f"{dt_pmtp:.2f} ms, BitMatch 100%"))
        else:
            print(f"  -> FALLO PMTP: {msg_rec}")
            results.append(("TEST 3", "PMTP 800 MB D=10^8", "FAIL", msg_rec))
    except Exception as ex:
        print(f"  -> EXCEPCIÓN: {ex}")
        results.append(("TEST 3", "PMTP 800 MB D=10^8", "ERROR", str(ex)))

    print("\n" + "=" * 80)
    print("                RESUMEN AUDITORÍA EXTREMA D = 100,000,000 (10^8)")
    print("=" * 80)
    passes = sum(1 for r in results if r[2] == "PASS")
    total = len(results)
    for num, name, status, detail in results:
        print(f"  [{status}] {num} ({name}): {detail}")
    print("=" * 80)
    print(f"  VEREDICTO: {passes}/{total} TESTS PASADOS EN D=10^8")
    print("=" * 80)
    return passes == total

if __name__ == "__main__":
    audit_10_8()
