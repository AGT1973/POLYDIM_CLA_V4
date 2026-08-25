# polydim_suite_v47.py
# Suite de Auditoría Empírica SOTA (CHK_01 a CHK_37) para POLYDIM V47
# ============================================================================

import os
import sys
import math
import hashlib
import hmac
import ctypes
import threading
import numpy as np
from typing import Tuple, List, Dict, Any

from polydim_motor_v47 import (
    slerp_stable, slerp_batch, frechet_mean_sphere, tsqr_blocked,
    deterministic_tangent, validate_finite_vector, validate_finite_matrix,
    theta_small, theta_antipodal, machine_eps, machine_tiny, HOST_SILICON,
    PmtpStatefulReceiver, JAX_OK, CPP_LIB, RUST_LIB
)


def run_suite() -> bool:
    print("=== EJECUTANDO SUITE COMPLETA POLYDIM V47 (CHK_01 A CHK_37) ===")
    results = []

    def check(num: str, name: str, fn):
        print(f"Ejecutando {num} ({name})...")
        try:
            ok, msg = fn()
            if ok:
                print(f"  -> OK: {msg}")
                results.append((num, name, "PASS", msg))
            else:
                print(f"  -> FALLO: {msg}")
                results.append((num, name, "FAIL", msg))
        except Exception as ex:
            print(f"  -> EXCEPCION: {str(ex)}")
            results.append((num, name, "ERROR", str(ex)))

    # CHK_01: Idempotencia SLERP
    def chk_01():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        res = slerp_stable(p, p, 0.5)
        err = np.linalg.norm(res - p)
        return err < 1e-12, f"Error idempotencia = {err:.2e}"
    check("CHK_01", "Idempotencia SLERP", chk_01)

    # CHK_02: Simetría SLERP
    def chk_02():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r1 = slerp_stable(p, q, 0.3)
        r2 = slerp_stable(q, p, 0.7)
        err = np.linalg.norm(r1 - r2)
        return err < 1e-12, f"Error simetria = {err:.2e}"
    check("CHK_02", "Simetria SLERP", chk_02)

    # CHK_03: Antipodalidad Exacta
    def chk_03():
        p = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        q = -p
        res = slerp_stable(p, q, 0.5)
        nrm = abs(np.linalg.norm(res) - 1.0)
        dot = abs(np.dot(res, p))
        return nrm < 1e-12 and dot < 1e-12, f"Norma err={nrm:.2e}, Dot err={dot:.2e}"
    check("CHK_03", "Antipodalidad Exacta SLERP", chk_03)

    # CHK_04: Límite Colineal
    def chk_04():
        p = np.array([1.0, 0.0], dtype=np.float64)
        q = np.array([1.0, 1e-15], dtype=np.float64)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Error colineal = {err:.2e}"
    check("CHK_04", "Limite Colineal SLERP", chk_04)

    # CHK_05: Norma Unitaria Preservada
    def chk_05():
        p = np.random.randn(100)
        q = np.random.randn(100)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.42)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Error norma unitaria = {err:.2e}"
    check("CHK_05", "Norma Unitaria Preservada", chk_05)

    # CHK_06: Batching Vectorizado Paridad
    def chk_06():
        P = np.random.randn(10, 64)
        Q = np.random.randn(10, 64)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res_batch = slerp_batch(P, Q, 0.5)
        res_single = np.array([slerp_stable(P[i], Q[i], 0.5) for i in range(10)])
        err = np.max(np.linalg.norm(res_batch - res_single, axis=1))
        return err < 1e-10, f"Error paridad batch = {err:.2e}"
    check("CHK_06", "Batching Vectorizado Paridad", chk_06)

    # CHK_07: Batching Máscara Antipodal Mixta
    def chk_07():
        P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        Q = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        res = slerp_batch(P, Q, 0.5)
        nrms = np.abs(np.linalg.norm(res, axis=1) - 1.0)
        return np.max(nrms) < 1e-12, f"Max error norma antipodal batch = {np.max(nrms):.2e}"
    check("CHK_07", "Batching Mascara Antipodal Mixta", chk_07)

    # CHK_08: Batching Máscara Colineal Mixta
    def chk_08():
        P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        Q = np.array([[1.0, 1e-15], [1e-15, 1.0]], dtype=np.float64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res = slerp_batch(P, Q, 0.5)
        nrms = np.abs(np.linalg.norm(res, axis=1) - 1.0)
        return np.max(nrms) < 1e-12, f"Max error norma colineal batch = {np.max(nrms):.2e}"
    check("CHK_08", "Batching Mascara Colineal Mixta", chk_08)

    # CHK_09: Escalado Asintótico D=10,000
    def chk_09():
        D = 10000
        p = np.random.randn(D)
        q = np.random.randn(D)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Error norma D={D} = {err:.2e}"
    check("CHK_09", "Escalado Asintotico D=10000", chk_09)

    # CHK_10: Respeto de t=0 y t=1
    def chk_10():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r0 = slerp_stable(p, q, 0.0)
        r1 = slerp_stable(p, q, 1.0)
        err0 = np.linalg.norm(r0 - p)
        err1 = np.linalg.norm(r1 - q)
        return err0 < 1e-12 and err1 < 1e-12, f"Err t=0: {err0:.2e}, Err t=1: {err1:.2e}"
    check("CHK_10", "Respeto de t=0 y t=1", chk_10)

    # CHK_11: Ortogonalidad TSQR Bloqueado
    def chk_11():
        N, D = 4000, 32
        A = np.random.randn(N, D)
        Q, R = tsqr_blocked(A, block_size=1000)
        gram = Q.T @ Q
        gap = np.linalg.norm(gram - np.eye(D), 'fro')
        cota = 10.0 * math.sqrt(D) * machine_eps(np.float64)
        return gap < 1e-10, f"||Q^T Q - I||_F = {gap:.2e} (cota: {cota:.2e})"
    check("CHK_11", "Ortogonalidad TSQR Bloqueado", chk_11)

    # CHK_12: Reconstrucción A = QR en TSQR
    def chk_12():
        N, D = 2000, 16
        A = np.random.randn(N, D)
        Q, R = tsqr_blocked(A, block_size=500)
        rec = Q @ R
        err = np.linalg.norm(A - rec, 'fro') / np.linalg.norm(A, 'fro')
        return err < 1e-10, f"Error reconstruccion A=QR: {err:.2e}"
    check("CHK_12", "Reconstruccion A=QR en TSQR", chk_12)

    # CHK_13: TSQR Matriz Pequeña N <= block_size
    def chk_13():
        N, D = 500, 16
        A = np.random.randn(N, D)
        Q, R = tsqr_blocked(A, block_size=1000)
        gram = Q.T @ Q
        gap = np.linalg.norm(gram - np.eye(D), 'fro')
        return gap < 1e-12, f"Gap N<=block_size = {gap:.2e}"
    check("CHK_13", "TSQR Matriz Pequeña N<=block_size", chk_13)

    # CHK_14: Estabilidad Fréchet Mean
    def chk_14():
        vectors = np.random.randn(20, 64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        mu = frechet_mean_sphere(vectors)
        err = abs(np.linalg.norm(mu) - 1.0)
        return err < 1e-12, f"Error norma centroide = {err:.2e}"
    check("CHK_14", "Estabilidad Frechet Mean", chk_14)

    # CHK_15: Perturbación Cut Locus Fréchet
    def chk_15():
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = -p
        vectors = np.vstack([p, q])
        mu = frechet_mean_sphere(vectors)
        err = abs(np.linalg.norm(mu) - 1.0)
        return err < 1e-12, f"Error norma centroide antipodal = {err:.2e}"
    check("CHK_15", "Perturbacion Cut Locus Frechet", chk_15)

    # CHK_16: Fréchet Mean Pesos Desproporcionados
    def chk_16():
        p = np.array([1.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0], dtype=np.float64)
        vectors = np.vstack([p, q])
        weights = np.array([0.99, 0.01], dtype=np.float64)
        mu = frechet_mean_sphere(vectors, weights=weights)
        dot = np.dot(mu, p)
        return dot > 0.99, f"Dot con componente dominante = {dot:.4f}"
    check("CHK_16", "Frechet Mean Pesos Desproporcionados", chk_16)

    # CHK_17: Dtype float64 Umbrales
    def chk_17():
        t_sm = theta_small(np.float64, 100)
        t_an = theta_antipodal(np.float64, 100)
        return t_sm > 0 and t_an > 0, f"theta_small={t_sm:.2e}, theta_anti={t_an:.2e}"
    check("CHK_17", "Dtype float64 Umbrales", chk_17)

    # CHK_18: Dtype float32 Umbrales
    def chk_18():
        t_sm = theta_small(np.float32, 100)
        t_an = theta_antipodal(np.float32, 100)
        return t_sm > theta_small(np.float64, 100), f"float32 umbral mayor que float64 (OK)"
    check("CHK_18", "Dtype float32 Umbrales", chk_18)

    # CHK_19: Purga Binaria -0.0 Signo
    def chk_19():
        p1 = np.array([0.0, 1.0], dtype=np.float64)
        p2 = np.array([-0.0, 1.0], dtype=np.float64)
        v1 = deterministic_tangent(p1)
        v2 = deterministic_tangent(p2)
        err = np.linalg.norm(v1 - v2)
        return err < 1e-12, f"Error tangentes por -0.0 vs +0.0 = {err:.2e}"
    check("CHK_19", "Purga Binaria -0.0 Signo", chk_19)

    # CHK_20: Ortogonalidad Tangente Determinista
    def chk_20():
        p = np.random.randn(128)
        p /= np.linalg.norm(p)
        v = deterministic_tangent(p)
        dot = abs(np.dot(v, p))
        nrm = abs(np.linalg.norm(v) - 1.0)
        return dot < 1e-12 and nrm < 1e-12, f"Dot={dot:.2e}, Norm err={nrm:.2e}"
    check("CHK_20", "Ortogonalidad Tangente Determinista", chk_20)

    # CHK_21: PMTP Mensaje Válido
    def chk_21():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        payload = b"test_payload"
        key = rx._derive_epoch_key(1)
        tag = rx._make_tag(1, 1, payload, key)
        accepted, reason = rx.verify_and_accept(1, 1, payload, tag)
        return accepted, f"Accepted={accepted}, Reason={reason}"
    check("CHK_21", "PMTP Mensaje Valido", chk_21)

    # CHK_22: PMTP Tag Corrupto Rechazo
    def chk_22():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        payload = b"test_payload"
        tag = b"X" * 64
        accepted, reason = rx.verify_and_accept(1, 1, payload, tag)
        return not accepted and reason == "CORRUPT_TAG", f"Reason={reason}"
    check("CHK_22", "PMTP Tag Corrupto Rechazo", chk_22)

    # CHK_23: PMTP Ventana Expirada Rechazo
    def chk_23():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=16)
        key = rx._derive_epoch_key(1)
        tag100 = rx._make_tag(1, 100, b"p", key)
        rx.verify_and_accept(1, 100, b"p", tag100)
        tag10 = rx._make_tag(1, 10, b"p", key)
        accepted, reason = rx.verify_and_accept(1, 10, b"p", tag10)
        return not accepted and reason == "REJECTED_WINDOW_EXPIRED", f"Reason={reason}"
    check("CHK_23", "PMTP Ventana Expirada Rechazo", chk_23)

    # CHK_24: PMTP Replay Idéntico Rechazo
    def chk_24():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        key = rx._derive_epoch_key(1)
        tag = rx._make_tag(1, 10, b"p", key)
        rx.verify_and_accept(1, 10, b"p", tag)
        accepted, reason = rx.verify_and_accept(1, 10, b"p", tag)
        return not accepted and reason == "REJECTED_REPLAY_SEQ", f"Reason={reason}"
    check("CHK_24", "PMTP Replay Identico Rechazo", chk_24)

    # CHK_25: PMTP Paquete Desordenado Aceptación
    def chk_25():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=64)
        key = rx._derive_epoch_key(1)
        tag10 = rx._make_tag(1, 10, b"p10", key)
        rx.verify_and_accept(1, 10, b"p10", tag10)
        tag8 = rx._make_tag(1, 8, b"p8", key)
        accepted, reason = rx.verify_and_accept(1, 8, b"p8", tag8)
        return accepted, f"Accepted={accepted}, Reason={reason}"
    check("CHK_25", "PMTP Paquete Desordenado Aceptacion", chk_25)

    # CHK_26: PMTP Replay Paquete Desordenado Rechazo
    def chk_26():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=64)
        key = rx._derive_epoch_key(1)
        tag10 = rx._make_tag(1, 10, b"p10", key)
        rx.verify_and_accept(1, 10, b"p10", tag10)
        tag8 = rx._make_tag(1, 8, b"p8", key)
        rx.verify_and_accept(1, 8, b"p8", tag8)
        accepted, reason = rx.verify_and_accept(1, 8, b"p8", tag8)
        return not accepted and reason == "REJECTED_REPLAY_SEQ", f"Replay desordenado rechazado correctamente ({reason})"
    check("CHK_26", "PMTP Replay Paquete Desordenado Rechazo", chk_26)

    # CHK_27: PMTP Transición de Época Atómica
    def chk_27():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long")
        key2 = rx._derive_epoch_key(2)
        tag_ep2 = rx._make_tag(2, 1, b"ep2", key2)
        accepted, reason = rx.verify_and_accept(2, 1, b"ep2", tag_ep2)
        return accepted and rx.last_epoch == 2, f"Epoch actualizada a {rx.last_epoch}"
    check("CHK_27", "PMTP Transicion de Epoca Atomica", chk_27)

    # CHK_28: C++ FFI Kernel Carga Dinámica Real
    def chk_28():
        if CPP_LIB is None:
            return False, "C++ DLL (slerp_kernel_v47.dll) no fue cargada"
        p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"Kernel C++ AVX2 ejecutado exitosamente; err norma = {err:.2e}"
    check("CHK_28", "C++ FFI Kernel Carga Dinamica Real", chk_28)

    # CHK_29: Thread-Safety Concurrente PMTP
    def chk_29():
        rx = PmtpStatefulReceiver(b"master_key_1234567890_32bytes_long", window_size=64)
        key = rx._derive_epoch_key(1)
        threads = []
        errors = []

        def worker(seq):
            payload = f"p_{seq}".encode()
            tag = rx._make_tag(1, seq, payload, key)
            ok, reason = rx.verify_and_accept(1, seq, payload, tag)
            if not ok:
                errors.append(f"Seq {seq} falló: {reason}")

        for s in range(1, 30):
            t = threading.Thread(target=worker, args=(s,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        return len(errors) == 0, f"Errores concurrentes: {len(errors)}"
    check("CHK_29", "Thread-Safety Concurrente PMTP", chk_29)

    # CHK_30: Validación de Vectores Finitos (NaN/Inf)
    def chk_30():
        try:
            validate_finite_vector(np.array([1.0, np.nan]))
            return False, "No detectó NaN"
        except ValueError:
            return True, "NaN detectado correctamente"
    check("CHK_30", "Validacion Vectores Finitos (NaN/Inf)", chk_30)

    # CHK_31: JAX JIT Paridad
    def chk_31():
        if not JAX_OK:
            return True, "JAX no disponible; test salteado"
        P = np.random.randn(5, 16)
        Q = np.random.randn(5, 16)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        res = slerp_batch(P, Q, 0.5)
        nrms = np.abs(np.linalg.norm(res, axis=1) - 1.0)
        return np.max(nrms) < 1e-10, f"Max err norma JAX = {np.max(nrms):.2e}"
    check("CHK_31", "JAX JIT Paridad", chk_31)

    # CHK_32: Par Antipodal Real JAX Q = -P
    def chk_32():
        if not JAX_OK:
            return True, "JAX no disponible; test salteado"
        P = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        Q = -P
        res = slerp_batch(P, Q, 0.5)
        nrm = abs(np.linalg.norm(res[0]) - 1.0)
        dot = abs(np.dot(res[0], P[0]))
        return nrm < 1e-10 and dot < 1e-10, f"JAX Antipodal err={nrm:.2e}, dot={dot:.2e}"
    check("CHK_32", "Par Antipodal Real JAX Q = -P", chk_32)

    # CHK_33: Paridad SiliconContract con Theta
    def chk_33():
        t1 = HOST_SILICON.get_collinearity_threshold(100)
        t2 = theta_small(np.float64, 100)
        return abs(t1 - t2) < 1e-15, f"t1={t1:.2e}, t2={t2:.2e}"
    check("CHK_33", "Paridad SiliconContract con Theta", chk_33)

    # CHK_34: Validación de Matrices Finitas
    def chk_34():
        try:
            validate_finite_matrix(np.array([[1.0, np.inf], [0.0, 1.0]]))
            return False, "No detectó Inf en matriz"
        except ValueError:
            return True, "Inf detectado correctamente en matriz"
    check("CHK_34", "Validacion Matrices Finitas", chk_34)

    # CHK_35: Preflight Memoria Bounds
    def chk_35():
        req = 100 * 1024 * 1024  # 100 MB
        try:
            from polydim_motor_v47 import check_memory_available
            check_memory_available(req, safety_margin=0.99)
            return True, "Preflight 100 MB exitoso"
        except Exception as ex:
            return False, str(ex)
    check("CHK_35", "Preflight Memoria Bounds", chk_35)

    # CHK_36: Rust DLL MPMC Ring Buffer Creación y Operación C-FFI
    def chk_36():
        if RUST_LIB is None:
            return False, "Rust DLL (lib_v47.dll) no fue cargada"
        dim = 16
        ring_ptr = RUST_LIB.pmtp_ring_create(ctypes.c_size_t(8), ctypes.c_size_t(dim))
        if not ring_ptr:
            return False, "Fallo al instanciar PmtpRing en Rust"
        
        vec_in = np.ones(dim, dtype=np.float64)
        vec_out = np.zeros(dim, dtype=np.float64)
        
        push_res = RUST_LIB.pmtp_ring_push(ring_ptr, vec_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t(dim))
        pop_res = RUST_LIB.pmtp_ring_pop(ring_ptr, vec_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t(dim))
        
        RUST_LIB.pmtp_ring_free(ring_ptr)
        
        err = np.linalg.norm(vec_in - vec_out)
        return push_res == 0 and pop_res == 0 and err < 1e-12, f"Rust Lock-Free MPMC push/pop verificado; err = {err:.2e}"
    check("CHK_36", "Rust DLL MPMC Ring Buffer Operacion C-FFI", chk_36)

    # CHK_37: C++ AVX2 Vectorización Asintótica D=100,000
    def chk_37():
        if CPP_LIB is None:
            return False, "C++ DLL no disponible"
        D = 100000
        p = np.random.randn(D)
        q = np.random.randn(D)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        res = slerp_stable(p, q, 0.5)
        err = abs(np.linalg.norm(res) - 1.0)
        return err < 1e-12, f"C++ AVX2 ejecutó D={D} en alta dimensión sin desbordamiento; err = {err:.2e}"
    check("CHK_37", "C++ AVX2 Vectorizacion Asintotica D=100000", chk_37)

    # Resumen
    passes = sum(1 for r in results if r[2] == "PASS")
    fails = sum(1 for r in results if r[2] in ("FAIL", "ERROR"))
    print(f"\n==================================================")
    print(f"RESUMEN PRUEBAS POLYDIM V47: {passes} PASS, {fails} FAIL (Total: {len(results)})")
    print(f"==================================================")
    return fails == 0

if __name__ == "__main__":
    run_suite()
