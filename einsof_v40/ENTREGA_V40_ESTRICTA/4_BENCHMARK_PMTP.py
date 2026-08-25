import jax
import jax.numpy as jnp
import time
import sys

def run_benchmark():
    try:
        import einsof_rust
    except ImportError:
        print("[!] ERROR: 'einsof_rust' aún no está compilado. Esperando a que termine MSVC...")
        sys.exit(0)

    N = 10000
    print(f"[*] Inicializando Anillo PMTP en Rust (Capacidad=16, Dimensiones={N})")
    
    # Instanciamos el objeto Rust directamente en Python!
    ring = einsof_rust.PmtpRing(16, N)

    print(f"[*] Generando tensor ND en JAX (N={N})...")
    key = jax.random.PRNGKey(42)
    tensor_jax = jax.random.normal(key, (N,))

    # Convertimos a formato CPU para cruzar el puente
    tensor_np = jax.device_get(tensor_jax).tolist() 

    print("[*] Empujando tensor a memoria Rust (Lock-free)...")
    t0 = time.time()
    success = ring.push(tensor_np)
    t1 = time.time()

    if success:
        print(f"  -> [OK] Tensor escrito en Rust en {(t1-t0)*1000:.3f} ms")
    else:
        print("  -> [FAIL] Buffer lleno")

    print("[*] Consumiendo tensor desde Rust (Lock-free)...")
    t2 = time.time()
    out_tensor = ring.pop()
    t3 = time.time()

    if out_tensor:
        print(f"  -> [OK] Tensor leído desde Rust en {(t3-t2)*1000:.3f} ms")
        
        # Volvemos a inyectarlo en el Acelerador (GPU/TPU) via JAX
        out_jax = jnp.array(out_tensor)
        
        # Verificamos la conservación de entropía
        drift = jnp.linalg.norm(tensor_jax - out_jax)
        print(f"[*] Deriva topológica (Drift de Precisión): {drift:.6e}")
        
        if drift < 1e-5:
            print("\n[SUCCESS] ¡PIPELINE JAX <-> RUST CERTIFICADO Y SIN COLAPSO 1D!")
        else:
            print("\n[WARNING] Corrupción de datos detectada en el puente.")
    else:
        print("  -> [FAIL] Buffer vacío")

if __name__ == "__main__":
    run_benchmark()
