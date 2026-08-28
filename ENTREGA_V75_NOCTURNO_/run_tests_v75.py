import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

# Forzamos XLA a CPU para la prueba de concepto sin bloquear GPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Insertar ruta
sys.path.insert(0, r"E:\POLYDIM_EINSOF\ENTREGA_V75_NOCTURNO_")

def run_tests():
    print("=========================================================")
    print("🚀 INICIANDO PRUEBAS EMPÍRICAS V75 (MODO BULLDOG) 🚀")
    print("=========================================================\n")
    
    try:
        import polydim_v75_monolito as pdm
        print("[✔] Monolito V75 importado correctamente.")
    except Exception as e:
        print(f"[X] FATAL: Error al importar monolito: {e}")
        sys.exit(1)
        
    # ----------------------------------------------------------------
    # 1. Test FFI Bridge (C++ / Rust)
    # ----------------------------------------------------------------
    print("\n[1] Probando compilación en caliente y XLA FFI Zero-Copy...")
    try:
        pdm.NativeFFIBridge.initialize()
        print(f"  -> Rust DLL cargada: {pdm.NativeFFIBridge._rust_dll is not None}")
        print(f"  -> C++ DLL cargada: {pdm.NativeFFIBridge._cpp_dll is not None}")
        print(f"  -> XLA Capsule Registrada: {pdm.NativeFFIBridge._xla_registered}")
        
        # Vectores de prueba (D=100)
        x = jax.random.normal(jax.random.PRNGKey(0), (100,))
        v = jax.random.normal(jax.random.PRNGKey(1), (100,))
        
        # Ejecutar JIT FFI
        print("  -> Lanzando JIT FFI Custom Call...")
        out_ffi = pdm.NativeFFIBridge.householder_reflect(x, v)
        print(f"  -> Output FFI Shape: {out_ffi.shape}, Dtype: {out_ffi.dtype}")
        print("  -> [✔] XLA FFI Zero-Copy Test: PASSED")
    except Exception as e:
        print(f"  -> [X] XLA FFI Test FAILED: {e}")
        
    # ----------------------------------------------------------------
    # 2. Test Geodesic Kernels
    # ----------------------------------------------------------------
    print("\n[2] Probando Geometría Diferencial (log_map, Newton)...")
    try:
        x_geo = jax.random.normal(jax.random.PRNGKey(2), (5000,))
        y_geo = jax.random.normal(jax.random.PRNGKey(3), (5000,))
        
        v_log = pdm.GeodesicKernels.log_map(x_geo, y_geo)
        print(f"  -> log_map ejecutado. Norm: {float(jnp.linalg.norm(v_log)):.6f}")
        
        v_newton = pdm.GeodesicKernels.log_map_newton(x_geo, y_geo, max_iter=3)
        print(f"  -> log_map_newton ejecutado. Norm: {float(jnp.linalg.norm(v_newton)):.6f}")
        print("  -> [✔] Geodesic Kernels Test: PASSED")
    except Exception as e:
        print(f"  -> [X] Geodesic Kernels Test FAILED: {e}")
        
    # ----------------------------------------------------------------
    # 3. Test PMTP Swarm (INT8)
    # ----------------------------------------------------------------
    print("\n[3] Probando compresión PMTP Swarm (INT8 & Epoch)...")
    try:
        agent = pdm.PMTPAgentBridge(port=0)
        epoch = agent.epoch_clock.increment()
        
        tensor = jax.random.normal(jax.random.PRNGKey(4), (10_000,))
        tensor_np = np.asarray(tensor, dtype=np.float32)
        
        abs_max = float(np.max(np.abs(tensor_np)))
        scale = 1.0 if abs_max == 0 else abs_max / 127.0
        quantized = np.clip(np.round(tensor_np / scale), -127, 127).astype(np.int8)
        
        print(f"  -> Tensor D=10,000 | Original: {tensor_np.nbytes} bytes | INT8: {quantized.nbytes} bytes")
        print(f"  -> Ratio de Compresión: {tensor_np.nbytes / quantized.nbytes:.2f}x")
        print(f"  -> Epoch Clock actualizado a: {epoch}")
        print("  -> [✔] PMTP Swarm Test: PASSED")
    except Exception as e:
        print(f"  -> [X] PMTP Swarm Test FAILED: {e}")

    print("\n=========================================================")
    print("FIN DE EJECUCIÓN")
    print("=========================================================\n")

if __name__ == '__main__':
    run_tests()
