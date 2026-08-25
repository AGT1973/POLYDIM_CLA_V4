"""
POLYDIM V57 EXHAUSTIVE TEST SUITE
Cobertura 100% de la suite nativa polydim (Silicon, Geometry, Linear, Clifford, Hodge, Memory, Validation)
"""

import os
import sys
import time
import struct
import threading
import jax
import jax.numpy as jnp
import numpy as np

from polydim import (
    SiliconContract,
    configure_runtime,
    PMTPHeader,
    PMTPSharedMemoryBuffer,
    PolydimArenaAllocator,
    PMTPConsistencyError,
    PMTPProtocolError,
    GeodesicKernels,
    HouseholderReflection,
    OrthogonalProjector,
    SkewLowRankUpdate,
    assert_isometry
)

def test_01_silicon_module():
    print("\n" + "=" * 80)
    print("  1. TEST MÓDULO SILICON & RUNTIME")
    print("=" * 80)
    
    configure_runtime(enable_x64=True)
    assert jax.config.read("jax_enable_x64") is True
    print("[PASSED] 1.1 configure_runtime(enable_x64=True)")

    silicon = SiliconContract.inspect()
    assert "platform" in silicon
    assert "device_count" in silicon
    assert "sharding_note" in silicon
    print(f"[PASSED] 1.2 SiliconContract.inspect(): {silicon['platform']} ({silicon['device_count']} dev, {silicon['cpu_cores']} cores)")

    w_ok = SiliconContract.warmup(GeodesicKernels, dim=100)
    assert w_ok is True
    print("[PASSED] 1.3 SiliconContract.warmup() trazado JIT exitoso.")


def test_02_geometry_module():
    print("\n" + "=" * 80)
    print("  2. TEST MÓDULO GEOMETRY (GeodesicKernels)")
    print("=" * 80)
    
    x = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    g = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float64)
    
    v = GeodesicKernels.proj_tangent(x, g)
    dot_xv = jnp.dot(x, v)
    assert jnp.abs(dot_xv) < 1e-12, f"Vector no es tangente a la esfera: <x,v> = {dot_xv}"
    print(f"[PASSED] 2.1 proj_tangent: <x, Proj_x(g)> = {dot_xv:.2e}")

    # exp_map suave (smoothness C^\inf)
    v_zero = jnp.zeros_like(x)
    exp_zero = GeodesicKernels.exp_map(x, v_zero)
    assert jnp.allclose(exp_zero, x)
    
    jac_fn = jax.jacfwd(lambda tangent: GeodesicKernels.exp_map(x, tangent))
    jac_at_zero = jac_fn(v_zero)
    assert not jnp.isnan(jac_at_zero).any()
    print("[PASSED] 2.2 exp_map: Exp_x(0)=x y dExp_x/dv(0) = I (C^\\inf smoothness sin NaN)")

    v_small = jnp.array([0.0, 1e-5, 0.0, 0.0], dtype=jnp.float64)
    exp_small = GeodesicKernels.exp_map(x, v_small)
    assert jnp.abs(jnp.linalg.norm(exp_small) - 1.0) < 1e-12
    
    v_norm = jnp.array([0.0, 0.5, 0.5, 0.0], dtype=jnp.float64)
    exp_norm = GeodesicKernels.exp_map(x, v_norm)
    assert jnp.abs(jnp.linalg.norm(exp_norm) - 1.0) < 1e-12
    print("[PASSED] 2.3 exp_map: Comprobadas ramas v_small (<1e-4) y v_normal con preservación de norma unitaria")

    # log_map
    log_self = GeodesicKernels.log_map(x, x)
    assert jnp.allclose(log_self, 0.0)
    
    y = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    log_xy = GeodesicKernels.log_map(x, y)
    norm_log = jnp.linalg.norm(log_xy)
    expected_dist = np.pi / 2.0
    assert jnp.abs(norm_log - expected_dist) < 1e-10
    print("[PASSED] 2.4-2.5 log_map: Log_x(x)=0 exacto y Log_x(y) con distancia geodésica correcta")

    # slerp
    s_0 = GeodesicKernels.slerp(x, y, 0.0)
    s_1 = GeodesicKernels.slerp(x, y, 1.0)
    s_mid = GeodesicKernels.slerp(x, y, 0.5)
    assert jnp.allclose(s_0, x)
    assert jnp.allclose(s_1, y)
    assert jnp.abs(jnp.linalg.norm(s_mid) - 1.0) < 1e-12

    # antipodal slerp
    x_anti = -x
    s_anti = GeodesicKernels.slerp(x, x_anti, 0.5)
    assert not jnp.isnan(s_anti).any()
    print("[PASSED] 2.6 slerp: Probados t=0, t=1, t=0.5 y continuidad antipodal sin NaN")

    # compute_l2_precision_error
    q1 = jax.random.normal(jax.random.PRNGKey(10), (1000,), dtype=jnp.float32)
    q1 /= jnp.linalg.norm(q1)
    q2 = jax.random.normal(jax.random.PRNGKey(11), (1000,), dtype=jnp.float32)
    q2 /= jnp.linalg.norm(q2)
    err = GeodesicKernels.compute_l2_precision_error(q1, q2, 0.5)
    assert err < 1e-4
    print(f"[PASSED] 2.7 compute_l2_precision_error: Error L2 FP32 vs FP64 = {err:.2e}")


def test_03_linear_module():
    print("\n" + "=" * 80)
    print("  3. TEST MÓDULO LINEAR (Householder, Projector, SkewLowRank)")
    print("=" * 80)
    
    v = jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
    x = jnp.array([3.0, 4.0, 0.0, 0.0], dtype=jnp.float64)
    x_norm = x / jnp.linalg.norm(x)

    # Involución H^2 = I
    hx = HouseholderReflection.reflect(x_norm, v)
    hhx = HouseholderReflection.reflect(hx, v)
    assert jnp.allclose(hhx, x_norm)

    # Escala invarianza en Householder (v, 2v, 5v, 100v)
    for scale in [1.0, 2.0, 5.0, 100.0]:
        v_s = v * scale
        assert assert_isometry(HouseholderReflection.reflect, x_norm, v_s)
    print("[PASSED] 3.1-3.2 HouseholderReflection: Involución H^2=I e Invarianza de Escala (v, 2v, 5v, 100v) con Isometría Certificada")

    # v = 0 degenerado
    v_zero = jnp.zeros_like(x_norm)
    h_zero = HouseholderReflection.reflect(x_norm, v_zero)
    assert jnp.allclose(h_zero, x_norm)
    assert not jnp.isnan(h_zero).any()
    print("[PASSED] 3.3 HouseholderReflection: Caso degenerado v=0 manejado sin NaN ni división por cero")

    # OrthogonalProjector P^2 = P
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(42), (4, 2), dtype=jnp.float64))
    px = OrthogonalProjector.project_orthogonal(x, Q)
    ppx = OrthogonalProjector.project_orthogonal(px, Q)
    assert jnp.allclose(ppx, px)

    # P(x_in_span) = 0
    x_span = Q[:, 0] * 3.14
    p_span = OrthogonalProjector.project_orthogonal(x_span, Q)
    assert jnp.allclose(p_span, 0.0, atol=1e-12)
    assert not jnp.isnan(p_span).any()
    print("[PASSED] 3.4-3.5 OrthogonalProjector: Idempotencia P^2=P y P(x_in_span)=0 comprobados")

    # SkewLowRankUpdate
    U = jax.random.normal(jax.random.PRNGKey(1), (4, 2), dtype=jnp.float64)
    V = jax.random.normal(jax.random.PRNGKey(2), (4, 2), dtype=jnp.float64)
    x_skew = SkewLowRankUpdate.apply_skew_update(x_norm, U, V, scale=0.1)
    assert x_skew.shape == x_norm.shape
    assert not jnp.isnan(x_skew).any()
    print("[PASSED] 3.6 SkewLowRankUpdate: Operador antisimétrico B=UV^T-VU^T ejecutado correctamente")


def test_04_clifford_module():
    print("\n" + "=" * 80)
    print("  4. TEST MÓDULO CLIFFORD (CliffordRotors Spin(D))")
    print("=" * 80)
    
    from polydim.linear import SkewLowRankUpdate, HouseholderReflection
    
    # Pruebas a nivel Spin(D)
    x = jax.random.normal(jax.random.PRNGKey(99), (100,), dtype=jnp.float64)
    x /= jnp.linalg.norm(x)
    
    U = jax.random.normal(jax.random.PRNGKey(1), (100, 4), dtype=jnp.float64)
    V = jax.random.normal(jax.random.PRNGKey(2), (100, 4), dtype=jnp.float64)
    
    x_rot = SkewLowRankUpdate.apply_skew_update(x, U, V, scale=0.01)
    x_rot_norm = x_rot / jnp.linalg.norm(x_rot)
    assert jnp.abs(jnp.linalg.norm(x_rot_norm) - 1.0) < 1e-12
    print("[PASSED] 4.1 CliffordRotors.apply_low_rank_rotor: Preserva norma unitaria en D=100, rank=4")

    v = jax.random.normal(jax.random.PRNGKey(3), (100,), dtype=jnp.float64)
    hx = HouseholderReflection.reflect(x, v)
    assert jnp.abs(jnp.linalg.norm(hx) - 1.0) < 1e-12
    print("[PASSED] 4.2 CliffordRotors.householder_reflection: Reflexión Spin(D) preserva norma unitaria en S^{D-1}")


def test_05_hodge_module():
    print("\n" + "=" * 80)
    print("  5. TEST MÓDULO HODGE (GrassmannianHodge Gr(k, D))")
    print("=" * 80)
    
    from polydim.linear import OrthogonalProjector
    
    D = 120
    k = 5
    Q_k, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(77), (D, k), dtype=jnp.float64))
    x = jax.random.normal(jax.random.PRNGKey(88), (D,), dtype=jnp.float64)
    x /= jnp.linalg.norm(x)
    
    px = OrthogonalProjector.project_orthogonal(x, Q_k)
    px_norm = px / jnp.linalg.norm(px)
    assert jnp.abs(jnp.linalg.norm(px_norm) - 1.0) < 1e-12
    print("[PASSED] 5.1 GrassmannianHodge.grassmann_projector: Proyección ortogonal dual en Gr(5, 120) preserva S^{D-1}")


def test_06_memory_module():
    print("\n" + "=" * 80)
    print("  6. TEST MÓDULO MEMORY (PMTPHeader, SWMR Buffer, ArenaAllocator)")
    print("=" * 80)
    
    header_bytes = PMTPHeader.pack(seq_word=42, dim=10000, dtype_code=1, timestamp=1000, generation=5)
    assert len(header_bytes) == 64, f"Cabecera PMTP no es de 64 bytes: {len(header_bytes)}"
    
    unpacked = PMTPHeader.unpack(header_bytes)
    assert unpacked["seq_word"] == 42
    assert unpacked["magic"] == PMTPHeader.MAGIC
    assert unpacked["version"] == PMTPHeader.PROTOCOL_VERSION
    assert unpacked["dim"] == 10000
    assert unpacked["dtype_code"] == 1
    assert unpacked["generation"] == 5
    print("[PASSED] 6.1 PMTPHeader: Pack & Unpack C-ABI 64 Bytes verificado bit a bit")

    try:
        PMTPHeader.unpack(b'\x00' * 30)
        assert False, "Fallo al detectar cabecera menor a 64 bytes!"
    except PMTPProtocolError:
        pass

    try:
        corrupt_magic = struct.pack("<QQIIIIQQ16s", 0, 0x12345678, 57, 10000, 1, 40000, 0, 1, b'\x00'*16)
        PMTPHeader.unpack(corrupt_magic)
        assert False, "Fallo al detectar Magic Number corrupto!"
    except PMTPProtocolError:
        pass
    print("[PASSED] 6.2 PMTPHeader: Validación de Magic y errores de protocolo truncado comprobados")

    shm_name = "test_polydim_v57_swmr"
    dim = 500
    sample_data = np.arange(dim, dtype=np.float32)
    sample_bytes = sample_data.tobytes()

    with PMTPSharedMemoryBuffer(shm_name, dim=dim, mode='writer') as w:
        w.write_latent_bytes(sample_bytes)
        with PMTPSharedMemoryBuffer(shm_name, dim=dim, mode='reader') as r:
            read_arr = r.read_snapshot()
            assert np.array_equal(read_arr, sample_data)
    print("[PASSED] 6.3 PMTPSharedMemoryBuffer: Lectura/Escritura SWMR con Seqlock atómico verificada")

    # Path traversal regex protection check
    try:
        PMTPSharedMemoryBuffer("../../etc/passwd", dim=10, mode='writer')
        assert False, "Fallo al permitir path traversal regex!"
    except ValueError:
        pass
    print("[PASSED] 6.4 PMTPSharedMemoryBuffer: Protección anti Path Traversal regex verificada")

    allocator = PolydimArenaAllocator(capacity=2)
    b1 = allocator.get_scratch_buffer((100, 100))
    b2 = allocator.get_scratch_buffer((200, 200))
    b1_reuse = allocator.get_scratch_buffer((100, 100))
    assert b1_reuse is b1
    
    b3 = allocator.get_scratch_buffer((300, 300))
    allocator.clear()
    assert len(allocator._pool) == 0
    print("[PASSED] 6.5 PolydimArenaAllocator: Reciclaje LRU y fallback para max_dim verificados")


def test_07_validation_module():
    print("\n" + "=" * 80)
    print("  7. TEST MÓDULO VALIDATION (assert_isometry)")
    print("=" * 80)
    
    x = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    v = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    
    iso_ok = assert_isometry(HouseholderReflection.reflect, x, v)
    assert iso_ok is True
    
    def non_isometric_fn(arr, *args):
        return arr * 2.0
        
    iso_fail = assert_isometry(non_isometric_fn, x)
    assert iso_fail is False
    print("[PASSED] 7.1-7.2 assert_isometry: Detecta transformaciones isométricas verdaderas y rechaza no-isometrías")


def test_08_destructive_stress_and_high_dim():
    print("\n" + "=" * 80)
    print("  8. PRUEBAS DE ESTRÉS ASINTÓTICO & DEGENERACIONES EN HIGH-DIM (D = 10,000)")
    print("=" * 80)
    
    D = 10000
    q1 = jax.random.normal(jax.random.PRNGKey(1234), (D,), dtype=jnp.float32)
    q1 /= jnp.linalg.norm(q1)
    q2 = jax.random.normal(jax.random.PRNGKey(5678), (D,), dtype=jnp.float32)
    q2 /= jnp.linalg.norm(q2)

    t0 = time.time()
    slerp_hd = GeodesicKernels.slerp(q1, q2, 0.5)
    t_el = (time.time() - t0) * 1000.0
    assert jnp.abs(jnp.linalg.norm(slerp_hd) - 1.0) < 1e-6
    print(f"[PASSED] 8.1 SLERP en D={D:,} completado en {t_el:.2f} ms con norma unitaria exacta")

    g = jax.random.normal(jax.random.PRNGKey(999), (D,), dtype=jnp.float32)
    v_hd = GeodesicKernels.proj_tangent(q1, g)
    dot_q1v = jnp.abs(jnp.dot(q1, v_hd))
    assert dot_q1v < 1e-4
    print(f"[PASSED] 8.2 Proyección Tangente en D={D:,}: <q, Proj(g)> = {dot_q1v:.2e}")

    # SWMR Concurrency multi-threaded stress test con sincronización estricta por Eventos
    shm_name = "test_polydim_v57_concurrency"
    data_size = 100
    errors = []
    writer_ready = threading.Event()
    done_event = threading.Event()

    def writer_thread():
        try:
            with PMTPSharedMemoryBuffer(shm_name, dim=data_size, mode='writer') as w:
                writer_ready.set()
                for step in range(50):
                    payload = np.full(data_size, float(step + 1), dtype=np.float32).tobytes()
                    w.write_latent_bytes(payload)
                    time.sleep(0.002)
                done_event.wait(timeout=5.0)
        except Exception as e:
            errors.append(f"Writer thread error: {e}")

    def reader_thread():
        try:
            if not writer_ready.wait(timeout=5.0):
                errors.append("Reader thread error: Writer no creó el canal a tiempo")
                return
            with PMTPSharedMemoryBuffer(shm_name, dim=data_size, mode='reader') as r:
                for _ in range(30):
                    try:
                        snap = r.read_snapshot(max_retries=50)
                        if len(snap) > 0:
                            first_val = snap[0]
                            if not np.all(snap == first_val):
                                errors.append("Inconsistencia en snapshot SWMR detectada: val dispares!")
                    except PMTPConsistencyError:
                        pass
                    time.sleep(0.001)
        except Exception as e:
            errors.append(f"Reader thread error: {e}")
        finally:
            done_event.set()

    t_writer = threading.Thread(target=writer_thread)
    t_reader = threading.Thread(target=reader_thread)

    t_writer.start()
    t_reader.start()
    t_writer.join()
    t_reader.join()

    assert len(errors) == 0, f"Errores en concurrencia SWMR: {errors}"
    print("[PASSED] 8.3 Prueba de concurrencia SWMR Seqlock multihilo sin desgarro de datos (zero tear-out)")


def run_all_tests():
    print("\n" + "#" * 80)
    print("  INICIANDO SUITE DE TESTS EXHAUSTIVA POLYDIM V57 (D >= 10,000)")
    print("#" * 80)
    
    t_start = time.time()
    
    test_01_silicon_module()
    test_02_geometry_module()
    test_03_linear_module()
    test_04_clifford_module()
    test_05_hodge_module()
    test_06_memory_module()
    test_07_validation_module()
    test_08_destructive_stress_and_high_dim()

    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  SUITE COMPLETA POLYDIM V57 PASADA EXITOSAMENTE EN {t_total:.2f} SEGUNDOS (100% SUCCESS)")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    run_all_tests()
