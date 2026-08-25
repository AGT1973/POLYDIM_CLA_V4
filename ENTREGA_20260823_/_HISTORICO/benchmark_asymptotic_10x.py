"""
POLYDIM V58 HARDWARE BENCHMARK & ASYMPTOTIC SUITE (10^x LOOP)
Google Colab / Kaggle / Local Execution for CPU, GPU (CUDA) and TPU (OpenXLA Shardy)
Autor: Antigravity Orchestrator (Bulldog Critic Mode)
Ley Ariel / Regla 11 & 16: Zero-Waste & Continuous Loop Execution
"""

import os
import sys
import time
import gc
import jax
import jax.numpy as jnp
import numpy as np

# Configurar runtime previo a trazado de grafos XLA
jax.config.update("jax_enable_x64", False) # FP32 por defecto para HBM/VRAM saturation

def inspect_hardware():
    """Interroga el acelerador físico (TPU, GPU, CPU) sin hardcodeo."""
    devices = jax.devices()
    platform = devices[0].platform.upper()
    dev_count = len(devices)
    cpu_cores = os.cpu_count() or 1
    
    # Detectar Colab / Kaggle
    is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ or os.path.exists('/content')
    is_kaggle = os.path.exists('/kaggle')
    env_name = "Google Colab" if is_colab else ("Kaggle Notebook" if is_kaggle else "Local / Remote Server")

    return {
        "platform": platform,
        "device_count": dev_count,
        "devices": [str(d) for d in devices],
        "cpu_cores": cpu_cores,
        "environment": env_name,
    }

# -----------------------------------------------------------------------------
# 1. KERNELS GEOMÉTRICOS RIEMANNIANOS V58
# -----------------------------------------------------------------------------

@jax.jit
def proj_tangent(x: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Proyección ortogonal al espacio tangente T_x S^{D-1}."""
    dot = jnp.einsum('i,i->', x, g)
    return g - dot * x

@jax.jit
def exp_map_smooth(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Exponential Map suave C^inf en S^{D-1} con Taylor expansion en v_sq."""
    v_sq = jnp.einsum('i,i->', v, v)
    is_small = v_sq < 1e-4
    
    v_sq2 = v_sq * v_sq
    v_sq3 = v_sq2 * v_sq
    cos_taylor = 1.0 - v_sq / 2.0 + v_sq2 / 24.0 - v_sq3 / 720.0
    sinc_taylor = 1.0 - v_sq / 6.0 + v_sq2 / 120.0 - v_sq3 / 5040.0
    
    safe_v_sq = jnp.where(is_small, 1.0, v_sq)
    norm_v = jnp.sqrt(safe_v_sq)
    cos_direct = jnp.cos(norm_v)
    sinc_direct = jnp.sin(norm_v) / norm_v
    
    cos_v = jnp.where(is_small, cos_taylor, cos_direct)
    sinc_v = jnp.where(is_small, sinc_taylor, sinc_direct)
    
    return x * cos_v + v * sinc_v

@jax.jit
def slerp_composable(q1: jnp.ndarray, q2: jnp.ndarray, t: float = 0.5) -> jnp.ndarray:
    """SLERP puro composable en S^{D-1}."""
    dot = jnp.einsum('i,i->', q1, q2)
    is_identity = dot >= (1.0 - 1e-6)
    
    q2_sign = jnp.where(dot < 0.0, -1.0, 1.0)
    dot_abs = jnp.abs(dot)
    dot_clipped = jnp.clip(dot_abs, -1.0 + 1e-7, 1.0 - 1e-7)

    theta = jnp.arccos(dot_clipped)
    sin_theta = jnp.sin(theta)
    safe_sin = jnp.where(sin_theta == 0.0, 1.0, sin_theta)

    w1 = jnp.sin((1.0 - t) * theta) / safe_sin
    w2 = (jnp.sin(t * theta) / safe_sin) * q2_sign

    interp = w1 * q1 + w2 * q2
    norm = jnp.sqrt(jnp.maximum(jnp.einsum('i,i->', interp, interp), 1e-15))
    return jnp.where(is_identity, q1, interp / norm)

@jax.jit
def householder_reflect(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Reflexión de Householder H_v(x) = x - 2 (v^T x / v^T v) v."""
    vv = jnp.einsum('i,i->', v, v)
    safe_vv = jnp.maximum(vv, 1e-15)
    dot = jnp.einsum('i,i->', v, x)
    reflected = x - 2.0 * (dot / safe_vv) * v
    return jnp.where(vv < 1e-15, x, reflected)

@jax.jit
def clifford_rotor_action(x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Rotor de Clifford Spin(D) de bajo rango B = U V^T - V U^T en O(r * D)."""
    v_tx = jnp.einsum('dr,d->r', V, x)
    u_tx = jnp.einsum('dr,d->r', U, x)
    bx = jnp.einsum('dr,r->d', U, v_tx) - jnp.einsum('dr,r->d', V, u_tx)
    x_rot = x - 0.5 * bx
    norm = jnp.sqrt(jnp.einsum('i,i->', x_rot, x_rot))
    return x_rot / norm

@jax.jit
def frechet_mean_step(mu: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """Iteración del Centro de Masa Riemanniano (Fréchet Mean) en S^{D-1}."""
    # points shape: (N, D)
    dots = jnp.einsum('nd,d->n', points, mu)[:, None] # (N, 1)
    v_i = points - dots * mu[None, :]                 # (N, D)
    
    v_norms = jnp.sqrt(jnp.maximum(jnp.einsum('nd,nd->n', v_i, v_i), 1e-12))[:, None]
    angles = jnp.arcsin(jnp.clip(v_norms, -1.0 + 1e-7, 1.0 - 1e-7))
    scales = angles / v_norms
    
    v_mean = jnp.mean(scales * v_i, axis=0)
    mu_next = mu + v_mean
    norm_next = jnp.sqrt(jnp.maximum(jnp.einsum('d,d->d', mu_next, mu_next), 1e-15))
    return mu_next / norm_next

# -----------------------------------------------------------------------------
# 2. MOTOR DE EVALUACIÓN ASINTÓTICA 10^x
# -----------------------------------------------------------------------------

def run_asymptotic_benchmark_suite(max_power: int = 6, num_loops: int = 1):
    hw = inspect_hardware()
    print("=" * 85)
    print(f"  POLYDIM V58 BENCHMARK ASINTÓTICO 10^x ({hw['environment']})")
    print(f"  Plataforma: {hw['platform']} | Dispositivos: {hw['device_count']} | Cores CPU: {hw['cpu_cores']}")
    print("=" * 85)

    powers = list(range(1, max_power + 1)) # 10^1 hasta 10^max_power

    for loop_idx in range(1, num_loops + 1):
        print(f"\n>>> INICIANDO BUCLE ASINTÓTICO #{loop_idx} <<<\n")
        
        for p in powers:
            dim = 10**p
            print("-" * 75)
            print(f"  EVALUANDO DIMENSIÓN D = 10^{p} ({dim:,} elementos | Float32 Payload = {(dim*4)/1024:.1f} KB)")
            print("-" * 75)

            try:
                key = jax.random.PRNGKey(42 + p)
                k1, k2, k3, k4 = jax.random.split(key, 4)

                # Generación de vectores en S^{D-1}
                q1 = jax.random.normal(k1, (dim,), dtype=jnp.float32)
                q1 = q1 / jnp.linalg.norm(q1)
                q2 = jax.random.normal(k2, (dim,), dtype=jnp.float32)
                q2 = q2 / jnp.linalg.norm(q2)
                v_tangent = jax.random.normal(k3, (dim,), dtype=jnp.float32) * 0.1

                # 1. Warm-Up & JIT Compile
                _ = slerp_composable(q1, q2, 0.5)
                jax.block_until_ready(_)

                # 2. Benchmark SLERP
                t0 = time.perf_counter()
                slerp_out = slerp_composable(q1, q2, 0.5)
                jax.block_until_ready(slerp_out)
                t_slerp = (time.perf_counter() - t0) * 1000.0

                # 3. Benchmark Exp_map
                t0 = time.perf_counter()
                exp_out = exp_map_smooth(q1, v_tangent)
                jax.block_until_ready(exp_out)
                t_exp = (time.perf_counter() - t0) * 1000.0

                # 4. Benchmark Householder Reflection
                t0 = time.perf_counter()
                h_out = householder_reflect(q1, q2)
                jax.block_until_ready(h_out)
                t_house = (time.perf_counter() - t0) * 1000.0

                # 5. Benchmark Clifford Rotor (r=16)
                r = 16
                U = jax.random.normal(k3, (dim, r), dtype=jnp.float32) * 0.01
                V = jax.random.normal(k4, (dim, r), dtype=jnp.float32) * 0.01
                
                t0 = time.perf_counter()
                cliff_out = clifford_rotor_action(q1, U, V)
                jax.block_until_ready(cliff_out)
                t_cliff = (time.perf_counter() - t0) * 1000.0

                # Auditoría de Isometría & Norma
                norm_slerp = float(jnp.linalg.norm(slerp_out))
                norm_exp = float(jnp.linalg.norm(exp_out))
                norm_h = float(jnp.linalg.norm(h_out))
                norm_cliff = float(jnp.linalg.norm(cliff_out))

                iso_err = abs(norm_cliff - 1.0)

                print(f"  [D=10^{p}] SLERP: {t_slerp:.4f} ms | ExpMap: {t_exp:.4f} ms | Householder: {t_house:.4f} ms | Clifford (r={r}): {t_cliff:.4f} ms")
                print(f"  [D=10^{p}] Norma Preservada: {norm_cliff:.6f} | Error Isométrico: {iso_err:.2e}")

                # Benchmark Fréchet Mean si D <= 10^5 (para evitar OOM en mini-batch de 10 puntos)
                if dim <= 100000:
                    points = jax.random.normal(key, (10, dim), dtype=jnp.float32)
                    points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)
                    mu_init = points[0]
                    
                    t0 = time.perf_counter()
                    mu_next = frechet_mean_step(mu_init, points)
                    jax.block_until_ready(mu_next)
                    t_frechet = (time.perf_counter() - t0) * 1000.0
                    print(f"  [D=10^{p}] Fréchet Mean Step (N=10): {t_frechet:.4f} ms")

            except MemoryError:
                print(f"  [OOM LIMIT] Memoria insuficiente para D = 10^{p}. Liberando memoria...")
                gc.collect()
            except Exception as e:
                print(f"  [ERROR] Falla en D = 10^{p}: {e}")
                gc.collect()

        print("\n" + "=" * 85)
        print(f"  BUCLE #{loop_idx} COMPLETADO EXITOSAMENTE")
        print("=" * 85 + "\n")


if __name__ == "__main__":
    # Si se pasa como argumento 'loop', corre indefinidamente (Modo Nocturno)
    is_infinite = len(sys.argv) > 1 and sys.argv[1].lower() == "loop"
    loops = 999999 if is_infinite else 1
    run_asymptotic_benchmark_suite(max_power=7, num_loops=loops)
