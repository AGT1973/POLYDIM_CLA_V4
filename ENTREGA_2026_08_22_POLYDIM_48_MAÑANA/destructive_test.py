import numpy as np
import time
import os
import sys

import jax
import jax.numpy as jnp

os.environ["JAX_ENABLE_X64"] = "true"

from polydim_matrix_free_clifford_engine_v48_fixed import slerp_nd, cayley_smw_retraction, EPS_MACH

def stress_jax():
    print("Iniciando Asintotica Destructiva JAX...")
    D = 1000000 # 1 Millon!
    K = 16
    
    # Test 1: Vectores Colineales y Antipodales puros
    p0 = jnp.ones(D, dtype=jnp.float64) / jnp.sqrt(D)
    p1 = -p0 # Antipodal
    try:
        res = slerp_nd(p0, p1, 0.5)
        norm = jnp.linalg.norm(res)
        print(f"[SLERP ANTIPODAL] D={D} -> Norm: {norm:.15f}")
        if jnp.isnan(norm):
            print("❌ FALLO CRITICO: SLERP retorno NaN en antipodal")
            return False
    except Exception as e:
        print(f"❌ FALLO SLERP Antipodal: {e}")
        return False
        
    # Test 2: Subnormales y Underflow
    p_tiny1 = jnp.full(D, 1e-300, dtype=jnp.float64)
    p_tiny2 = jnp.full(D, 1e-300, dtype=jnp.float64)
    res_tiny = slerp_nd(p_tiny1, p_tiny2, 0.5)
    print(f"[SLERP SUBNORMAL] D={D} -> Norm: {jnp.linalg.norm(res_tiny):.15f}")
    
    # Test 3: Cayley Retraction OOM y Overflow
    print("Asignando Tensores Pesados para Cayley...")
    U = jax.random.normal(jax.random.PRNGKey(0), (D, K), dtype=jnp.float64)
    V = jax.random.normal(jax.random.PRNGKey(1), (D, K), dtype=jnp.float64)
    x = jax.random.normal(jax.random.PRNGKey(2), (D,), dtype=jnp.float64)
    
    U = U / jnp.linalg.norm(U, axis=0)
    V = V / jnp.linalg.norm(V, axis=0)
    
    t0 = time.time()
    x_rot = cayley_smw_retraction(U, V, x)
    x_rot.block_until_ready()
    t1 = time.time()
    
    norm_rot = jnp.linalg.norm(x_rot)
    print(f"[CAYLEY 1 MILLON D] Tiempo: {t1-t0:.4f}s, Norm: {norm_rot:.15f}")
    
    if jnp.isnan(norm_rot) or abs(norm_rot - 1.0) > 1e-8:
        print("❌ FALLO CRITICO: Cayley perdio isometria")
        return False
        
    print("✅ TEST DESTRUCTIVO JAX SUPERADO")
    return True

if __name__ == "__main__":
    success = stress_jax()
    sys.exit(0 if success else 1)
