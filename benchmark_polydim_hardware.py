import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import jit

print("="*60)
print("BENCHMARK ASINTÓTICO JAX / TPU - MODO DIOS")
print("="*60)
print(f"Dispositivos JAX detectados: {jax.devices()}")

# Obligar a usar float64 en JAX (por defecto suele ser float32 en TPU)
jax.config.update("jax_enable_x64", True)

@jit
def slerp_jax(q1, q2, t):
    dot = jnp.sum(q1 * q2)
    q2_sign = jnp.where(dot < 0.0, -1.0, 1.0)
    dot = jnp.abs(dot)
    
    # Clip para evitar NaNs en el límite
    dot = jnp.clip(dot, -1.0, 1.0)
    theta = jnp.arccos(dot)
    sin_theta = jnp.sin(theta)
    
    is_antipodal = sin_theta < 1e-12
    
    w1 = jnp.where(is_antipodal, 1.0 - t, jnp.sin((1.0 - t) * theta) / sin_theta)
    w2_abs = jnp.where(is_antipodal, t, jnp.sin(t * theta) / sin_theta)
    w2 = w2_abs * q2_sign
    
    return w1 * q1 + w2 * q2

def run_benchmark():
    for exp in range(4, 11): # Hasta 10^10
        D = 10**exp
        print(f"--- Preparando D=10^{exp} ---")
        try:
            key = jax.random.PRNGKey(42)
            k1, k2 = jax.random.split(key)
            
            # Forzamos sharding implicito dejando que JAX maneje la particion
            q1 = jax.random.normal(k1, (D,), dtype=jnp.float64)
            q1 = q1 / jnp.linalg.norm(q1)
            q2 = jax.random.normal(k2, (D,), dtype=jnp.float64)
            q2 = q2 / jnp.linalg.norm(q2)
            
            # Warmup / JIT Compile
            out = slerp_jax(q1, q2, 0.5)
            jax.block_until_ready(out)
            
            n_runs = 10 if exp <= 7 else 2
            t0 = time.time()
            for _ in range(n_runs):
                out = slerp_jax(q1, q2, 0.5)
            jax.block_until_ready(out)
            tf = time.time()
            
            ms_per_call = ((tf - t0) * 1000) / n_runs
            bytes_moved = D * 8 * 3 # q1, q2, out (float64)
            gb_per_sec = (bytes_moved / 1e9) / (ms_per_call / 1000)
            
            print(f"D=10^{exp} SUPERADO | {ms_per_call:.4f} ms | Throughput: {gb_per_sec:.2f} GB/s")
            
            del q1, q2, out
        
        except Exception as e:
            print(f"¡EXPLOSIÓN ESTRUCTURAL en D=10^{exp} (float64)! Causa: {e}")
            
            if exp >= 9:
                print(f"--- Reintentando D=10^{exp} en float32 ---")
                try:
                    q1 = jax.random.normal(k1, (D,), dtype=jnp.float32)
                    q1 = q1 / jnp.linalg.norm(q1)
                    q2 = jax.random.normal(k2, (D,), dtype=jnp.float32)
                    q2 = q2 / jnp.linalg.norm(q2)
                    out = slerp_jax(q1, q2, 0.5)
                    jax.block_until_ready(out)
                    
                    t0 = time.time()
                    out = slerp_jax(q1, q2, 0.5)
                    jax.block_until_ready(out)
                    tf = time.time()
                    
                    ms_per_call = (tf - t0) * 1000
                    bytes_moved = D * 4 * 3
                    gb_per_sec = (bytes_moved / 1e9) / (ms_per_call / 1000)
                    print(f"D=10^{exp} (float32) SUPERADO | {ms_per_call:.4f} ms | Throughput: {gb_per_sec:.2f} GB/s")
                except Exception as e2:
                    print(f"¡EXPLOSIÓN TOTAL incluso en float32!: {e2}")
            break

if __name__ == '__main__':
    run_benchmark()
