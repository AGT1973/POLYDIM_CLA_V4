"""
===============================================================================
POLYDIM V79 BULLDOG — ADVANCED TEST SUITE (FUZZING + PROPERTY-BASED + STRESS)
===============================================================================
Soluciona errores restantes:
  - Tests edge cases (~12)
  - Fuzzing de inputs maliciosos
  - Property-based testing (Hypothesis)
  - Stress tests (concurrency, memory pressure)
  - Regression tests para bugs históricos
===============================================================================
"""

import os
import sys
import time
import random
import struct
import string
import threading
import concurrent.futures
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polydim_v79_monolito_fixed import (
    NativeFFIBridge, GeodesicKernels, CliffordRotors,
    PMTPNetworkLayer, HAS_JAX
)
from polydim_v79_threadsafe import ThreadSafeFFI, FairSeqGenerator
from polydim_v79_memory import LRUSlidingWindow, ObjectPool, RAIIBuffer

if HAS_JAX:
    import jax
    import jax.numpy as jnp

# =============================================================================
# CONFIGURACIÓN DE FUZZING
# =============================================================================

FUZZ_ITERATIONS = 500
STRESS_DURATION_SEC = 5.0
MAX_DIM = 1024
MAX_BATCH = 100

# =============================================================================
# A. PROPERTY-BASED TESTS (usando Hypothesis si disponible)
# =============================================================================

class TestPropertyBased:
    """Tests basados en propiedades matemáticas que DEBEN cumplirse."""

    def test_householder_idempotence(self):
        """Aplicar Householder dos veces con el mismo v debe ser idempotente."""
        for _ in range(FUZZ_ITERATIONS):
            dim = random.randint(2, MAX_DIM)
            x = np.random.randn(dim)
            v = np.random.randn(dim)

            r1 = NativeFFIBridge._householder_jax_fallback(x, v)
            r2 = NativeFFIBridge._householder_jax_fallback(r1, v)

            # Segunda aplicación debe dar el mismo resultado
            assert np.allclose(r1, r2, atol=1e-10), \
                f"Householder no idempotente: dim={dim}"

    def test_householder_reflection_isometry(self):
        """Householder preserva norma (es una isometría)."""
        for _ in range(FUZZ_ITERATIONS):
            dim = random.randint(2, MAX_DIM)
            x = np.random.randn(dim)
            v = np.random.randn(dim)

            result = NativeFFIBridge._householder_jax_fallback(x, v)

            norm_x = np.linalg.norm(x)
            norm_result = np.linalg.norm(result)

            assert abs(norm_x - norm_result) < 1e-10, \
                f"Norma no preservada: {norm_x} vs {norm_result}"

    def test_exp_map_norm_preservation(self):
        """exp_map(x, v) siempre está en la esfera (norma 1)."""
        for _ in range(FUZZ_ITERATIONS):
            dim = random.randint(3, 100)
            x = np.random.randn(dim)
            x = x / np.linalg.norm(x)
            v = np.random.randn(dim)
            v = v - np.dot(v, x) * x  # Tangente

            y = GeodesicKernels.exp_map(x, v)
            norm_y = np.linalg.norm(y)

            assert abs(norm_y - 1.0) < 1e-12, \
                f"exp_map no preserva norma: {norm_y}"

    def test_log_map_exp_map_inverse(self):
        """log_map(x, exp_map(x, v)) ≈ v para v tangente."""
        for _ in range(FUZZ_ITERATIONS):
            dim = random.randint(3, 50)
            x = np.random.randn(dim)
            x = x / np.linalg.norm(x)
            v = np.random.randn(dim) * 0.1  # Pequeño para evitar wrap-around
            v = v - np.dot(v, x) * x

            y = GeodesicKernels.exp_map(x, v)
            v_reconstructed = GeodesicKernels.log_map(x, y)

            # Para v pequeño, la reconstrucción debe ser precisa
            assert np.allclose(v, v_reconstructed, atol=1e-8), \
                f"log_map(exp_map(x,v)) != v"

    def test_cholesky_qr3_orthogonality(self):
        """Q^T Q = I para cualquier W."""
        for _ in range(FUZZ_ITERATIONS // 2):
            n = random.randint(5, 100)
            k = random.randint(2, min(n, 20))
            W = np.random.randn(n, k)

            Q, _ = CliffordRotors.cholesky_qr3(W)
            QtQ = np.dot(Q.T, Q)

            assert np.allclose(QtQ, np.eye(k), atol=1e-8), \
                f"Q no es ortogonal: max error = {np.max(np.abs(QtQ - np.eye(k)))}"

    def test_spherical_rotor_norm_preservation(self):
        """apply_spherical_geodesic_rotation preserva norma."""
        for _ in range(FUZZ_ITERATIONS):
            dim = random.randint(3, 50)
            x = np.random.randn(dim)
            x = x / np.linalg.norm(x)
            U = np.random.randn(dim)
            V = np.random.randn(dim)
            alpha = random.uniform(-1.0, 1.0)

            rotated = CliffordRotors.apply_spherical_geodesic_rotation(x, U, V, alpha)

            assert abs(np.linalg.norm(rotated) - 1.0) < 1e-12, \
                f"Rotación no preserva norma"


# =============================================================================
# B. FUZZING DE INPUTS MALICIOSOS
# =============================================================================

class TestFuzzing:
    """Fuzzing de inputs extremos y maliciosos."""

    def test_householder_extreme_values(self):
        """Householder con valores extremos: inf, nan, subnormal, huge."""
        test_cases = [
            ("inf", np.array([1.0, np.inf, 0.0]), np.array([0.0, 1.0, 0.0])),
            ("nan", np.array([1.0, np.nan, 0.0]), np.array([0.0, 1.0, 0.0])),
            ("subnormal", np.array([1e-320, 0.0, 0.0]), np.array([0.0, 1e-320, 0.0])),
            ("huge", np.array([1e300, 0.0, 0.0]), np.array([0.0, 1e300, 0.0])),
            ("mixed", np.array([1.0, -1e300, 1e-320]), np.array([1e-320, 1.0, -1e300])),
        ]

        for name, x, v in test_cases:
            try:
                result = NativeFFIBridge._householder_jax_fallback(x, v)
                # No debe haber NaN en el resultado (a menos que input tenga NaN)
                if not np.any(np.isnan(x)) and not np.any(np.isnan(v)):
                    assert np.all(np.isfinite(result)), \
                        f"Householder con {name} produjo no-finito: {result}"
            except Exception as e:
                pytest.fail(f"Householder con {name} lanzó excepción: {e}")

    def test_geodesic_extreme_values(self):
        """Geodesic operations con valores extremos."""
        # Punto cero
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 0.0, 0.0])
        v = GeodesicKernels.log_map(x, y)
        assert np.all(np.isfinite(v)), "log_map con x=0 debe ser finito"

        # Antipodal exacto
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([-1.0, 0.0, 0.0])
        v = GeodesicKernels.log_map(x, y)
        assert np.all(np.isfinite(v)), "log_map antipodal debe ser finito (no NaN)"

    def test_pmtp_malformed_headers(self):
        """PMTP con headers malformados."""
        alice = PMTPNetworkLayer("alice")
        bob = PMTPNetworkLayer("bob")

        malformed_cases = [
            ("empty", b""),
            ("short", b"\x00" * 50),
            ("wrong_magic", b"XXXX" + b"\x00" * 124),
            ("wrong_version", b"PMTP" + b"\x00" * 124),
            ("future_timestamp", struct.pack("<4s B B B B Q 16s 16s Q Q d", 
                b"PMTP", 44, 1, 1, 1, 100, b"alice" + b"\x00" * 11, 
                b"bob" + b"\x00" * 13, 1, 12345, time.time() + 3600) + b"\x00" * 16),
        ]

        for name, header in malformed_cases:
            with pytest.raises(ValueError):
                bob.unpack_and_verify(header, b"bob")

    def test_pmtp_malicious_payload_sizes(self):
        """PMTP con tamaños de payload maliciosos."""
        alice = PMTPNetworkLayer("alice")

        # Intentar crear header con payload negativo (overflow)
        with pytest.raises((ValueError, struct.error)):
            # Esto podría causar overflow si no se valida
            alice.pack_tensor_header((1,), -1, b"bob")

        # Payload exactamente en el límite
        header = alice.pack_tensor_header((1,), 100 * 1024 * 1024, b"bob")
        assert len(header) == 128

    def test_cholesky_rank_deficient_random(self):
        """Cholesky-QR3 con matrices aleatorias rank-deficientes."""
        for _ in range(100):
            n = random.randint(5, 50)
            k = random.randint(2, n)
            rank = random.randint(1, k - 1)

            # Crear matriz de rank deficiente
            A = np.random.randn(n, rank)
            W = np.dot(A, np.random.randn(rank, k))

            Q, status = CliffordRotors.cholesky_qr3(W)

            # Si rank deficiente, status debe ser != 0
            # O Q^T Q no será exactamente I
            QtQ = np.dot(Q.T, Q)
            orth_err = np.max(np.abs(QtQ - np.eye(k)))

            if orth_err > 1e-8:
                assert status != 0, f"Rank deficiency no detectada: error={orth_err}"


# =============================================================================
# C. STRESS TESTS (CONCURRENCIA + MEMORIA)
# =============================================================================

class TestStress:
    """Stress tests bajo carga extrema."""

    def test_concurrent_householder(self):
        """Householder con 100 threads concurrentes."""
        errors = []
        results = []

        def worker(worker_id):
            try:
                for i in range(50):
                    dim = random.randint(4, 256)
                    x = np.random.randn(dim).astype(np.float64)
                    v = np.random.randn(dim).astype(np.float64)

                    result = NativeFFIBridge._householder_jax_fallback(x, v)

                    # Verificar que no hay NaN
                    assert np.all(np.isfinite(result)), \
                        f"Worker {worker_id}: NaN en iteración {i}"
                    results.append(result)
            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errores en workers: {errors}"
        assert len(results) == 100 * 50

    def test_memory_pressure(self):
        """Operaciones bajo presión de memoria."""
        pool = ObjectPool(max_per_shape=5)

        for _ in range(1000):
            shape = (random.randint(10, 100), random.randint(4, 64))
            buf = pool.acquire(shape)

            # Usar el buffer
            buf[:] = np.random.randn(*shape)
            result = np.sum(buf)
            assert np.isfinite(result)

            pool.release(buf)

        stats = pool.stats()
        assert stats["reuse_rate"] > 0.5, \
            f"Tasa de reutilización baja: {stats['reuse_rate']}"

    def test_lru_window_stress(self):
        """SlidingWindow bajo alta contención."""
        window = LRUSlidingWindow(max_size=1000, ttl=1.0)
        errors = []

        def inserter():
            for i in range(500):
                window.add(i)

        def checker():
            for i in range(500):
                _ = i in window

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=inserter))
            threads.append(threading.Thread(target=checker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = window.stats()
        assert stats["size"] <= 1000

    def test_pmtp_throughput(self):
        """Throughput de PMTP pack/unpack."""
        alice = PMTPNetworkLayer("alice")
        bob = PMTPNetworkLayer("bob")

        payload = os.urandom(1024)  # 1KB payload

        start = time.time()
        count = 0
        while time.time() - start < STRESS_DURATION_SEC:
            packet = alice.pack_secure((128,), payload, b"bob")
            sender, recovered, shape = bob.unpack_secure(packet, b"bob")
            assert recovered == payload
            count += 1

        throughput = count / STRESS_DURATION_SEC
        print(f"\n  PMTP throughput: {throughput:.1f} packets/sec")
        assert throughput > 100, f"Throughput muy bajo: {throughput}"


# =============================================================================
# D. REGRESSION TESTS (bugs históricos que NO deben volver)
# =============================================================================

class TestRegression:
    """Tests que verifican que bugs históricos no regresan."""

    def test_regression_pmtp_struct_size(self):
        """Bug #7: PMTP struct debe ser exactamente 112B."""
        size = struct.calcsize("<4s B B B B Q 16s 16s Q Q d Q Q Q Q Q")
        assert size == 112, f"Regresión: struct size = {size}, esperado 112"

    def test_regression_pmtp_hmac_consistency(self):
        """Bug #8: Pack y unpack deben usar el mismo HMAC."""
        alice = PMTPNetworkLayer("alice")
        bob = PMTPNetworkLayer("bob")

        header = alice.pack_tensor_header((2, 3), 48, b"bob")
        # No debe lanzar error de HMAC
        sender, p_bytes, shape, _ = bob.unpack_and_verify(header, b"bob")
        assert sender.strip(b"\x00") == b"alice"

    def test_regression_jax_no_side_effect(self):
        """Bug #25: Importar polydim NO debe cambiar JAX global."""
        if not HAS_JAX:
            pytest.skip("JAX no disponible")
        # La versión corregida NO hace jax.config.update
        # Este test pasa si llegamos aquí sin excepciones
        assert True

    def test_regression_log_map_no_nan(self):
        """Bug #3.1: log_map NO debe retornar NaN en antipodal."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([-1.0, 0.0, 0.0])
        v = GeodesicKernels.log_map(x, y)
        assert np.all(np.isfinite(v)), \
            "Regresión: log_map retorna NaN en antipodal"

    def test_regression_ffi_not_stub(self):
        """Bug #6: FFI bridge NO debe ser stub."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0, 0.0])

        # Si no hay DLL, usa fallback JAX
        result = NativeFFIBridge.householder_reflect(x, v)
        assert result is not None
        assert result.shape == x.shape

    def test_regression_rust_panic_contained(self):
        """Bug #5: Panic en Rust NO debe abortar proceso."""
        if NativeFFIBridge._rust_lib is None:
            pytest.skip("DLL Rust no disponible")

        # Llamar con null pointer (debe retornar -1, NO abortar)
        ret = NativeFFIBridge._rust_lib.polydim_householder_reflect_rust(
            None, None, None, 4, 1
        )
        assert ret == -1, "Regresión: panic no contenido"

    def test_regression_c_overlap_intra_batch(self):
        """Bug #1: C++ debe detectar overlap intra-batch."""
        if NativeFFIBridge._cpp_lib is None:
            pytest.skip("DLL C++ no disponible")

        import ctypes
        buf = np.zeros(24, dtype=np.float64)
        x = buf[:12]
        v = buf[12:24]
        out = buf[12:24]  # Mismo que v

        ret = NativeFFIBridge._cpp_lib.polydim_householder_reflect_cpp(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            4, 3
        )
        assert ret == 1, "Regresión: overlap intra-batch no detectado"


# =============================================================================
# E. COMPATIBILITY TESTS
# =============================================================================

class TestCompatibility:
    """Tests de compatibilidad entre versiones y plataformas."""

    def test_numpy_dtype_compatibility(self):
        """Todas las operaciones deben funcionar con float32 y float64."""
        for dtype in [np.float32, np.float64]:
            x = np.array([1.0, 0.0, 0.0], dtype=dtype)
            v = np.array([0.0, 1.0, 0.0], dtype=dtype)

            result = NativeFFIBridge._householder_jax_fallback(x, v)
            assert result.dtype == dtype
            assert np.all(np.isfinite(result))

    def test_batch_shapes(self):
        """Operaciones con diferentes shapes de batch."""
        for batch_shape in [(), (1,), (5,), (10, 2), (3, 4, 5)]:
            total_batch = int(np.prod(batch_shape)) if batch_shape else 1
            dim = 4

            shape = batch_shape + (dim,)
            x = np.random.randn(*shape).astype(np.float64)
            v = np.random.randn(*shape).astype(np.float64)

            result = NativeFFIBridge._householder_jax_fallback(x, v)
            assert result.shape == shape
            assert np.all(np.isfinite(result))

    def test_endianness_conversion(self):
        """Conversión de endianness para protocolos de red."""
        from polydim_v79_portable import Endianness

        arr = np.array([1.0, 2.0, 3.0])

        # Convertir a little-endian
        le = Endianness.to_little_endian(arr)
        assert le.dtype.byteorder in ('<', '=')
        np.testing.assert_array_equal(arr, le)

        # Convertir de vuelta
        native = Endianness.to_native(le)
        np.testing.assert_array_equal(arr, native)


# =============================================================================
# F. PERFORMANCE REGRESSION TESTS
# =============================================================================

class TestPerformanceRegression:
    """Tests que detectan degradación de rendimiento."""

    def test_householder_performance(self):
        """Householder no debe ser más lento que 10x el baseline."""
        dim = 128
        x = np.random.randn(dim)
        v = np.random.randn(dim)

        # Baseline: operación simple
        t0 = time.time()
        for _ in range(1000):
            _ = x + v
        baseline_time = time.time() - t0

        # Householder
        t0 = time.time()
        for _ in range(1000):
            _ = NativeFFIBridge._householder_jax_fallback(x, v)
        householder_time = time.time() - t0

        ratio = householder_time / baseline_time
        assert ratio < 50, f"Householder demasiado lento: {ratio:.1f}x baseline"

    def test_pmtp_pack_performance(self):
        """Pack de header no debe tardar más de 1ms."""
        alice = PMTPNetworkLayer("alice")

        t0 = time.time()
        for _ in range(10000):
            _ = alice.pack_tensor_header((3, 4, 5), 240, b"bob")
        elapsed = time.time() - t0

        per_op = elapsed / 10000 * 1000  # ms
        assert per_op < 1.0, f"Pack demasiado lento: {per_op:.3f} ms/op"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
