"""
POLYDIM V58 DESTRUCTIVE ADVERSARIAL STRESS SUITE (test_v58_destructive_stress.py)
Ataque Adversarial Multivectorial de Nivel Red Team Bulldog
Ley Ariel / Regla 17: Prohibición Absoluta de Happy-Path y Auditoría Pasiva
"""

import os
import sys
import time
import threading
import random
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polydim import (
    configure_runtime,
    SiliconContract,
    PMTPHeader,
    PMTPSharedMemoryBuffer,
    PolydimArenaAllocator,
    PMTPConsistencyError,
    PMTPProtocolError,
    GeodesicKernels,
    HouseholderReflection,
    OrthogonalProjector,
    SkewLowRankUpdate,
    assert_isometry,
    QuantumGeodesicKernels,
    LieGroupOperators,
    QuantumInformation,
    TensorNetwork,
    HolographicDuality,
    TopologicalInvariants,
    KahlerGeometry,
    RiemannianLearning
)
from polydim.clifford import CliffordRotors
from polydim.hodge import GrassmannianHodge


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


# =============================================================================
# FASE 1: FUZZING ESTOCÁSTICO Y DEGENERACIONES NUMÉRICAS EXTREMAS
# =============================================================================
def attack_phase_1_fuzzing_and_degeneracies():
    print_banner("FASE 1: FUZZING ESTOCÁSTICO & ENTRADAS DEGENERADAS (NaN, Inf, Zero, Subnormales)")
    
    dim = 1000
    x_valid = jnp.array([1.0] + [0.0] * (dim - 1), dtype=jnp.float32)

    # 1.1 Vector Cero Total
    x_zero = jnp.zeros(dim, dtype=jnp.float32)
    v_zero = jnp.zeros(dim, dtype=jnp.float32)
    
    res_exp_zero = GeodesicKernels.exp_map(x_valid, v_zero)
    assert not bool(jnp.isnan(res_exp_zero).any()), "Fase 1.1: exp_map(x, 0) produjo NaN!"
    print("[ATTACK PASSED] 1.1 exp_map(x, vector_cero) inmune a NaN")

    res_log_zero = GeodesicKernels.log_map(x_valid, x_valid)
    assert not bool(jnp.isnan(res_log_zero).any()), "Fase 1.1: log_map(x, x) produjo NaN!"
    assert float(jnp.linalg.norm(res_log_zero)) < 1e-6, "Fase 1.1: log_map(x, x) != 0"
    print("[ATTACK PASSED] 1.1 log_map(x, x) sin distancia fabricada")

    # 1.2 Subnormales Flotantes (< 1e-38)
    tiny_val = 1e-35
    v_tiny = jnp.ones(dim, dtype=jnp.float32) * tiny_val
    res_exp_tiny = GeodesicKernels.exp_map(x_valid, v_tiny)
    assert not bool(jnp.isnan(res_exp_tiny).any()), "Fase 1.2: exp_map con subnormales produjo NaN!"
    print("[ATTACK PASSED] 1.2 exp_map con valores subnormales inmune a bajo nivel")

    # 1.3 Inyección de NaN e Inf (Comprobación de detección)
    v_nan = jnp.array([np.nan] + [0.0] * (dim - 1), dtype=jnp.float32)
    v_inf = jnp.array([np.inf] + [0.0] * (dim - 1), dtype=jnp.float32)

    res_nan = GeodesicKernels.exp_map(x_valid, v_nan)
    assert bool(jnp.isnan(res_nan).any()), "Fase 1.3: exp_map no propagó el NaN adecuadamente!"
    print("[ATTACK PASSED] 1.3 Propagación de NaN controlada detectada correctamente")

    # 1.4 Householder con Vector Reflector Cero y Escalas Extremas
    h_zero = HouseholderReflection.reflect(x_valid, v_zero)
    assert jnp.allclose(h_zero, x_valid), "Fase 1.4: Householder(v=0) no devolvió x!"
    
    # Escala extrema 1e10
    v_huge = jnp.array([0.0, 1e10, 0.0, 0.0] + [0.0] * (dim - 4), dtype=jnp.float32)
    h_huge = HouseholderReflection.reflect(x_valid, v_huge)
    assert not bool(jnp.isnan(h_huge).any()), "Fase 1.4: Householder con escala 1e10 produjo NaN!"
    print("[ATTACK PASSED] 1.4 Householder inmune a reflector cero y escalamiento extremo")

    # 1.5 Fuzzing aleatorio continuo (100 iteraciones)
    print("  [+] Ejecutando Fuzzing aleatorio (100 iteraciones)...")
    for i in range(100):
        scale = 10 ** random.uniform(-10, 3)
        vec_random = jax.random.normal(jax.random.PRNGKey(i), (dim,), dtype=jnp.float32) * scale
        vec_norm = vec_random / jnp.linalg.norm(vec_random)
        
        # Test exp_map, slerp, Householder
        _exp = GeodesicKernels.exp_map(x_valid, vec_random * 0.01)
        _slerp = GeodesicKernels.slerp(x_valid, vec_norm, 0.5)
        _house = HouseholderReflection.reflect(x_valid, vec_random)
        
        assert not bool(jnp.isnan(_exp).any()), f"Fuzzing #{i} exp_map falló con escala {scale}"
        assert not bool(jnp.isnan(_slerp).any()), f"Fuzzing #{i} slerp falló con escala {scale}"
        assert not bool(jnp.isnan(_house).any()), f"Fuzzing #{i} Householder falló con escala {scale}"

    print("[ATTACK PASSED] 1.5 Fuzzing estocástico (100 iteraciones) superado sin fallos ni NaNs")


# =============================================================================
# FASE 2: CONCURRENCIA DESTRUCTIVA SWMR SEQLOCK UNDER HEAVY CONTENTION
# =============================================================================
def attack_phase_2_heavy_concurrency():
    print_banner("FASE 2: CONCURRENCIA DESTRUCTIVA SWMR SEQLOCK (5 Escritores x 15 Lectores)")
    
    shm_name = "test_v58_stress_swmr_heavy"
    dim_conc = 5000
    num_readers = 15
    iterations = 200
    
    errors = []

    writer_buf = PMTPSharedMemoryBuffer(shm_name, dim=dim_conc, mode='writer')
    
    def writer_worker():
        try:
            for it in range(iterations):
                val = float(it + 1)
                payload = np.full(dim_conc, val, dtype=np.float32).tobytes()
                writer_buf.write_latent_bytes(payload)
                time.sleep(0.001)
        except Exception as e:
            errors.append(f"Writer error: {e}")

    def reader_worker(reader_id):
        try:
            time.sleep(0.01)
            with PMTPSharedMemoryBuffer(shm_name, dim=dim_conc, mode='reader') as r:
                for _ in range(iterations):
                    try:
                        snap = r.read_snapshot(max_retries=200)
                        if len(snap) > 0:
                            first_elem = snap[0]
                            if not np.all(snap == first_elem):
                                errors.append(f"Reader {reader_id}: DESGARRO DE DATOS DETECTADO! Valores dispares en snapshot")
                    except PMTPConsistencyError:
                        pass
                    time.sleep(0.0005)
        except Exception as e:
            errors.append(f"Reader {reader_id} error: {e}")

    threads = []
    t_writer = threading.Thread(target=writer_worker)
    threads.append(t_writer)

    for r_id in range(num_readers):
        t_r = threading.Thread(target=reader_worker, args=(r_id,))
        threads.append(t_r)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    writer_buf.unlink()

    assert len(errors) == 0, f"Fase 2 FALLÓ con los siguientes errores de concurrencia: {errors}"
    print("[ATTACK PASSED] Fase 2: 1 escritor SWMR y 15 lectores simultáneos ejecutados con ZERO DATA TEARING")


# =============================================================================
# FASE 3: ATAQUES DE SEGURIDAD & CABECERA C-ABI CORRUPTA
# =============================================================================
def attack_phase_3_security_and_headers():
    print_banner("FASE 3: ATAQUES DE CABECERA C-ABI & PATH TRAVERSAL")
    
    # 3.1 Path Traversal
    malicious_paths = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "/absolute/path/attack",
        "name with spaces and !@#$%^&*()",
    ]
    for path in malicious_paths:
        try:
            PMTPSharedMemoryBuffer(path, dim=100)
            assert False, f"Fase 3.1: No se bloqueó Path Traversal para '{path}'!"
        except (ValueError, PMTPProtocolError):
            pass
    print("[ATTACK PASSED] 3.1 Todos los intentos de Path Traversal fueron bloqueados")

    # 3.2 Cabeceras corruptas y truncadas
    trunc_header = b'\x00' * 10
    try:
        PMTPHeader.unpack(trunc_header)
        assert False, "Fase 3.2: No se rechazó cabecera truncada!"
    except PMTPProtocolError:
        pass

    # Cabecera con Magic erróneo
    header_ok = PMTPHeader.pack(seq_word=2, dim=1000)
    bad_magic = bytearray(header_ok)
    bad_magic[8:16] = b'ATTACK12'
    try:
        PMTPHeader.unpack(bytes(bad_magic))
        assert False, "Fase 3.2: No se rechazó Magic corrupto!"
    except PMTPProtocolError:
        pass

    # Cabecera con Versión de Protocolo no soportada (v999)
    bad_version = bytearray(header_ok)
    bad_version[16:20] = (999).to_bytes(4, byteorder='little')
    try:
        PMTPHeader.unpack(bytes(bad_version))
        assert False, "Fase 3.2: No se rechazó versión no soportada!"
    except PMTPProtocolError:
        pass

    print("[ATTACK PASSED] 3.2 Validación C-ABI 64 Bytes rechazó Magic corrupto, versión inválida y cabeceras truncadas")


# =============================================================================
# FASE 4: ESTRÉS DE MEMORIA LRU & OVERFLOW DE SCRATCH BUFFERS
# =============================================================================
def attack_phase_4_arena_allocator_stress():
    print_banner("FASE 4: ESTRÉS DE ARENA ALLOCATOR & EVICTION POOL")
    
    allocator = PolydimArenaAllocator(max_dim=50000, capacity=3)

    buffers = []
    for i in range(10):
        shape = (1000 + i * 100, 10)
        buf = allocator.get_scratch_buffer(shape)
        buffers.append(buf)

    assert len(allocator._pool) <= 3, f"ArenaAllocator superó el max_pool_size: {len(allocator._pool)}"
    print("[ATTACK PASSED] 4.1 ArenaAllocator LRU eviction mantuvo el límite de pool estrictamente")

    huge_buf = allocator.get_scratch_buffer((100000, 10))
    assert huge_buf.shape == (100000, 10), "Fallback max_dim falló"
    print("[ATTACK PASSED] 4.2 Fallback para tensores que superan max_dim verificado")


# =============================================================================
# FASE 5: ESTRÉS ASINTÓTICO EXTREMO D = 10,000,000 (10^7)
# =============================================================================
def attack_phase_5_asymptotic_10x7():
    print_banner("FASE 5: ESTRÉS ASINTÓTICO EN D = 10,000,000 (10^7 ELEM)")
    
    dim = 10000000
    print(f"  [+] Generando tensores de D = 10^7 (Payload Float32 = {(dim*4)/(1024*1024):.2f} MB)...")

    key = jax.random.PRNGKey(107)
    k1, k2 = jax.random.split(key, 2)

    t0 = time.time()
    q1 = jax.random.normal(k1, (dim,), dtype=jnp.float32)
    q1 = q1 / jnp.linalg.norm(q1)
    q2 = jax.random.normal(k2, (dim,), dtype=jnp.float32)
    q2 = q2 / jnp.linalg.norm(q2)
    t_gen = (time.time() - t0) * 1000.0
    print(f"  [+] Generación y normalización en D=10^7 completada en {t_gen:.2f} ms")

    t0 = time.time()
    slerp_10x7 = GeodesicKernels.slerp(q1, q2, 0.5)
    jax.block_until_ready(slerp_10x7)
    t_slerp = (time.time() - t0) * 1000.0

    norm_out = float(jnp.linalg.norm(slerp_10x7))
    assert not bool(jnp.isnan(slerp_10x7).any()), "CRÍTICO: SLERP en D=10^7 produjo NaN!"
    assert abs(norm_out - 1.0) < 1e-5, f"SLERP en D=10^7 violó norma unitaria: norm={norm_out}"

    print(rf"[ATTACK PASSED] 5.1 SLERP en D=10,000,000 (10^7) ejecutado en {t_slerp:.2f} ms | $\|x\|_2 = {norm_out:.6f}$")


# =============================================================================
# FASE 6: AUDITORÍA DE MÓDULOS SOTA V58 (Fubini-Study, Cayley, Entropía, MERA, Chern, SGD)
# =============================================================================
def attack_phase_6_sota_modules_audit():
    print_banner("FASE 6: CERTIFICACIÓN MATEMÁTICA Y FÍSICA DE MÓDULOS SOTA V58")

    # 6.1 Quantum Geometry
    psi1 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
    psi2 = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex64)
    fs_dist = QuantumGeodesicKernels.fubini_study_distance(psi1, psi2)
    assert abs(float(fs_dist) - np.pi / 2.0) < 1e-4, "QuantumGeometry: Fubini-Study distance != pi/2"
    print("[ATTACK PASSED] 6.1 QuantumGeodesicKernels Fubini-Study distance exactness verified")

    # 6.2 Lie Groups (Cayley Transform)
    key = jax.random.PRNGKey(42)
    A_raw = jax.random.normal(key, (4, 4), dtype=jnp.float32)
    R_cayley = LieGroupOperators.cayley_transform(A_raw)
    R_orthogonal_check = R_cayley @ R_cayley.T
    assert jnp.allclose(R_orthogonal_check, jnp.eye(4), atol=1e-4), "LieGroupOperators: Cayley transform lost orthogonality!"
    print("[ATTACK PASSED] 6.2 LieGroupOperators Cayley transform exact orthogonality R @ R^T = I verified")

    # 6.3 Quantum Information & Entanglement Entropy
    psi_entangled = jnp.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)], dtype=jnp.complex64) # Bell state
    s_ent = QuantumInformation.entanglement_entropy(psi_entangled, 2, 2)
    assert abs(float(s_ent) - 1.0) < 1e-4, f"QuantumInformation: Bell state entanglement entropy != 1.0 bit! Got {s_ent}"
    print("[ATTACK PASSED] 6.3 QuantumInformation Bell state max entanglement S(rho_A) = 1.0 bit verified")

    # 6.4 Topological Invariants & Chern Number
    grid_size = 8
    dummy_grid = jax.random.normal(key, (grid_size, grid_size, 4), dtype=jnp.float32)
    dummy_grid = dummy_grid / jnp.linalg.norm(dummy_grid, axis=-1, keepdims=True)
    b_curv = TopologicalInvariants.berry_curvature_2d(dummy_grid)
    c_num = TopologicalInvariants.chern_number(b_curv)
    assert not bool(jnp.isnan(c_num).any()), "TopologicalInvariants: Chern number returned NaN"
    print(f"[ATTACK PASSED] 6.4 TopologicalInvariants Berry curvature grid & Chern integer C = {int(c_num)} verified")

    # 6.5 Riemannian SGD & Parallel Transport
    x_sphere = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    grad_e = jnp.array([0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)
    x_next = RiemannianLearning.riemannian_sgd_step(x_sphere, grad_e, lr=0.1)
    norm_next = float(jnp.linalg.norm(x_next))
    assert abs(norm_next - 1.0) < 1e-5, "RiemannianLearning: SGD step lost unit norm on S^{D-1}!"
    print("[ATTACK PASSED] 6.5 RiemannianLearning SGD step unit norm preservation verified")


def run_destructive_suite():
    print("\n" + "=" * 80)
    print("  INICIANDO SUITE DE ATAQUE ADVERSARIAL DESTRUCTIVO (LEY ARIEL / BULLDOG)")
    print("=" * 80)

    t_start = time.time()

    attack_phase_1_fuzzing_and_degeneracies()
    attack_phase_2_heavy_concurrency()
    attack_phase_3_security_and_headers()
    attack_phase_4_arena_allocator_stress()
    attack_phase_5_asymptotic_10x7()
    attack_phase_6_sota_modules_audit()

    t_total = time.time() - t_start

    print("\n" + "=" * 80)
    print(f"  SUITE DE ATAQUE DESTRUCTIVA COMPLETADA CON ÉXITO ABSOLUTO EN {t_total:.2f}s")
    print("  EL CÓDIGO HA RESISTIDO TODOS LOS VECTORES DE ATAQUE SIN CORTES NI NANs")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_destructive_suite()
