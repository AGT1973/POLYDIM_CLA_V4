"""
===============================================================================
POLYDIM V79 BULLDOG — BENCHMARKS COMPARATIVOS
===============================================================================
Compara:
  - Código actual (simulado) vs código corregido (SIMD)
  - NumPy puro vs JAX vs C++ vs Rust
  - PMTP pack/unpack throughput
  - Geodesic operations latency
===============================================================================
"""

import os
import sys
import time
import numpy as np
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 1000
DIMS = [64, 128, 256, 512, 1024]
BATCH_SIZES = [1, 10, 100, 1000]

# =============================================================================
# HELPERS
# =============================================================================

def benchmark(func, setup=None, iterations=1000, name="op"):
    """Benchmark simple con warmup."""
    if setup:
        setup()

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        func()

    # Medición
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)  # ms

    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    min_t = min(times)
    max_t = max(times)

    return {
        "name": name,
        "mean_ms": mean,
        "stdev_ms": stdev,
        "min_ms": min_t,
        "max_ms": max_t,
        "throughput": 1000.0 / mean if mean > 0 else float('inf'),
    }

def print_results(results):
    print(f"\n{'Operación':<40} {'Mean(ms)':<12} {'Stdev':<10} {'Min':<10} {'Max':<10} {'Ops/sec':<12}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<40} {r['mean_ms']:<12.4f} {r['stdev_ms']:<10.4f} "
              f"{r['min_ms']:<10.4f} {r['max_ms']:<10.4f} {r['throughput']:<12.1f}")


# =============================================================================
# BENCHMARK 1: HOUSEHOLDER REFLECT
# =============================================================================

def benchmark_householder():
    print("=" * 95)
    print("BENCHMARK 1: Householder Reflect")
    print("=" * 95)

    results = []

    for dim in DIMS:
        x = np.random.randn(dim).astype(np.float64)
        v = np.random.randn(dim).astype(np.float64)

        # Simular C++ actual (escalar, división en loop)
        def householder_actual():
            v_max = np.max(np.abs(v))
            if v_max < 1e-30:
                return x.copy()
            v_norm = v / v_max
            v_sq = np.dot(v_norm, v_norm)
            xv_dot = np.dot(x, v_norm)
            factor = 2.0 * xv_dot / v_sq
            return x - factor * v_norm

        # Simular C++ corregido (inv_v_max, vectorizado)
        def householder_fixed():
            v_max = np.max(np.abs(v))
            if v_max < 1e-30:
                return x.copy()
            inv_v_max = 1.0 / v_max
            v_norm = v * inv_v_max
            v_sq = np.dot(v_norm, v_norm)
            xv_dot = np.dot(x, v_norm)
            factor = (2.0 * xv_dot / v_sq) * inv_v_max
            return x - factor * v

        # NumPy vectorizado
        def householder_numpy():
            return x - (2.0 * np.dot(x, v) / np.dot(v, v)) * v

        r1 = benchmark(householder_actual, iterations=BENCHMARK_ITERATIONS, 
                      name=f"actual_scalar_dim{dim}")
        r2 = benchmark(householder_fixed, iterations=BENCHMARK_ITERATIONS,
                      name=f"fixed_scalar_dim{dim}")
        r3 = benchmark(householder_numpy, iterations=BENCHMARK_ITERATIONS,
                      name=f"numpy_vec_dim{dim}")

        results.extend([r1, r2, r3])

    print_results(results)

    # Análisis de speedup
    print("\n📊 ANÁLISIS DE SPEEDUP:")
    for dim in DIMS:
        actual = next(r for r in results if r['name'] == f"actual_scalar_dim{dim}")
        fixed = next(r for r in results if r['name'] == f"fixed_scalar_dim{dim}")
        numpy = next(r for r in results if r['name'] == f"numpy_vec_dim{dim}")
        speedup_fixed = actual['mean_ms'] / fixed['mean_ms']
        speedup_numpy = actual['mean_ms'] / numpy['mean_ms']
        print(f"  dim={dim:4d}: fixed={speedup_fixed:.2f}x, numpy={speedup_numpy:.2f}x")


# =============================================================================
# BENCHMARK 2: BATCH PROCESSING
# =============================================================================

def benchmark_batch():
    print("\n" + "=" * 95)
    print("BENCHMARK 2: Batch Processing")
    print("=" * 95)

    results = []
    dim = 128

    for batch in BATCH_SIZES:
        xb = np.random.randn(batch, dim).astype(np.float64)
        vb = np.random.randn(batch, dim).astype(np.float64)

        # Loop Python (simula C++ actual)
        def batch_loop():
            out = np.empty_like(xb)
            for b in range(batch):
                v_max = np.max(np.abs(vb[b]))
                if v_max < 1e-30:
                    out[b] = xb[b]
                    continue
                v_norm = vb[b] / v_max
                v_sq = np.dot(v_norm, v_norm)
                xv_dot = np.dot(xb[b], v_norm)
                factor = 2.0 * xv_dot / v_sq
                out[b] = xb[b] - factor * v_norm
            return out

        # Vectorizado NumPy
        def batch_vectorized():
            v_max = np.max(np.abs(vb), axis=1, keepdims=True)
            safe_v_max = np.where(v_max < 1e-30, 1.0, v_max)
            v_norm = vb / safe_v_max
            v_sq = np.sum(v_norm * v_norm, axis=1, keepdims=True)
            is_zero = v_sq < 1e-30
            safe_v_sq = np.where(is_zero, 1.0, v_sq)
            factor = 2.0 * np.sum(xb * v_norm, axis=1, keepdims=True) / safe_v_sq
            reflect = xb - factor * v_norm
            return np.where(is_zero, xb, reflect)

        r1 = benchmark(batch_loop, iterations=max(10, 1000 // batch),
                      name=f"loop_batch{batch}")
        r2 = benchmark(batch_vectorized, iterations=max(10, 1000 // batch),
                      name=f"vec_batch{batch}")

        results.extend([r1, r2])

    print_results(results)


# =============================================================================
# BENCHMARK 3: GEODESIC OPERATIONS
# =============================================================================

def benchmark_geodesic():
    print("\n" + "=" * 95)
    print("BENCHMARK 3: Geodesic Operations")
    print("=" * 95)

    try:
        from polydim_v79_monolito_fixed import GeodesicKernels
        has_module = True
    except ImportError:
        has_module = False
        print("  (polydim_v79_monolito_fixed.py no disponible, usando simulación)")

    results = []

    for dim in [3, 10, 50, 100]:
        x = np.random.randn(dim)
        x = x / np.linalg.norm(x)
        v = np.random.randn(dim)
        v = v - np.dot(v, x) * x  # Tangente

        if has_module:
            def exp_map():
                return GeodesicKernels.exp_map(x, v)

            def log_map():
                y = GeodesicKernels.exp_map(x, v)
                return GeodesicKernels.log_map(x, y)
        else:
            def exp_map():
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    return x
                return np.cos(v_norm) * x + (np.sin(v_norm) / v_norm) * v

            def log_map():
                y = exp_map()
                c = np.dot(x, y)
                c = np.clip(c, -1.0, 1.0)
                theta = np.arccos(c)
                y_perp = y - c * x
                y_perp_norm = np.linalg.norm(y_perp)
                if y_perp_norm < 1e-12:
                    return np.zeros_like(x)
                return theta * y_perp / y_perp_norm

        r1 = benchmark(exp_map, iterations=BENCHMARK_ITERATIONS,
                      name=f"exp_map_dim{dim}")
        r2 = benchmark(log_map, iterations=BENCHMARK_ITERATIONS,
                      name=f"log_map_dim{dim}")

        results.extend([r1, r2])

    print_results(results)


# =============================================================================
# BENCHMARK 4: PMTP PACK/UNPACK
# =============================================================================

def benchmark_pmtp():
    print("\n" + "=" * 95)
    print("BENCHMARK 4: PMTP Pack/Unpack")
    print("=" * 95)

    try:
        from polydim_v79_monolito_fixed import PMTPNetworkLayer
        has_pmtp = True
    except ImportError:
        has_pmtp = False
        print("  (PMTP no disponible)")
        return

    results = []
    alice = PMTPNetworkLayer("alice")
    bob = PMTPNetworkLayer("bob")

    shapes = [(2,), (10, 10), (100, 100), (1000,)]

    for shape in shapes:
        payload_size = int(np.prod(shape)) * 8  # float64

        def pack_header():
            return alice.pack_tensor_header(shape, payload_size, b"bob")

        def pack_secure():
            payload = os.urandom(payload_size)
            return alice.pack_secure(shape, payload, b"bob")

        r1 = benchmark(pack_header, iterations=10000,
                      name=f"pack_header_{shape}")

        if payload_size <= 10 * 1024 * 1024:  # Solo para payloads pequeños
            r2 = benchmark(pack_secure, iterations=1000,
                          name=f"pack_secure_{shape}")
            results.append(r2)

        results.append(r1)

    print_results(results)


# =============================================================================
# BENCHMARK 5: CHOLESKY-QR3
# =============================================================================

def benchmark_cholesky():
    print("\n" + "=" * 95)
    print("BENCHMARK 5: Cholesky-QR3")
    print("=" * 95)

    try:
        from polydim_v79_monolito_fixed import CliffordRotors
        has_clifford = True
    except ImportError:
        has_clifford = False
        print("  (CliffordRotors no disponible)")
        return

    results = []

    for K in [5, 10, 20, 50]:
        W = np.random.randn(100, K).astype(np.float64)

        def cholesky_qr3():
            return CliffordRotors.cholesky_qr3(W, max_iter=3)

        def numpy_qr():
            return np.linalg.qr(W)

        r1 = benchmark(cholesky_qr3, iterations=100,
                      name=f"cholesky_qr3_K{K}")
        r2 = benchmark(numpy_qr, iterations=100,
                      name=f"numpy_qr_K{K}")

        results.extend([r1, r2])

    print_results(results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 95)
    print("POLYDIM V79 BULLDOG — BENCHMARK SUITE")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print("=" * 95)

    benchmark_householder()
    benchmark_batch()
    benchmark_geodesic()
    benchmark_pmtp()
    benchmark_cholesky()

    print("\n" + "=" * 95)
    print("BENCHMARKS COMPLETADOS")
    print("=" * 95)
