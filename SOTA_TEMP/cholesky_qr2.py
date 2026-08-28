import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)

@jax.jit
def cholesky_qr2(A: jnp.ndarray) -> jnp.ndarray:
    '''
    [SOTA] Iterated Cholesky QR (Cholesky-QR2).
    Reemplaza al Gram-Schmidt clásico. Computación pura de GEMM (matmul),
    altamente eficiente en TPUs/GPUs.
    '''
    # Iteración 1
    G1 = jnp.dot(A.T, A, precision=jax.lax.Precision.HIGHEST)
    L1 = jax.scipy.linalg.cholesky(G1, lower=True)
    Q1 = jax.scipy.linalg.solve_triangular(L1.T, A.T, lower=False).T
    
    # Iteración 2
    G2 = jnp.dot(Q1.T, Q1, precision=jax.lax.Precision.HIGHEST)
    L2 = jax.scipy.linalg.cholesky(G2, lower=True)
    Q2 = jax.scipy.linalg.solve_triangular(L2.T, Q1.T, lower=False).T
    
    # Iteración 3 (QR3 para matrices ultra mal condicionadas D >= 8192)
    G3 = jnp.dot(Q2.T, Q2, precision=jax.lax.Precision.HIGHEST)
    L3 = jax.scipy.linalg.cholesky(G3, lower=True)
    Q3 = jax.scipy.linalg.solve_triangular(L3.T, Q2.T, lower=False).T
    
    return Q3

def test_qr():
    print("Iniciando prueba empírica de Cholesky-QR2...")
    D, K = 8192, 128
    A = jax.random.normal(jax.random.PRNGKey(0), (D, K), dtype=jnp.float64)
    
    # Warmup
    _ = cholesky_qr2(A)
    
    start = time.perf_counter()
    Q = cholesky_qr2(A)
    Q.block_until_ready()
    end = time.perf_counter()
    
    # Evaluar ortogonalidad: Q^T Q debería ser muy cercana a la Identidad
    I_approx = jnp.dot(Q.T, Q, precision=jax.lax.Precision.HIGHEST)
    I_exact = jnp.eye(K, dtype=jnp.float64)
    error = jnp.max(jnp.abs(I_approx - I_exact))
    
    print(f"Latencia (D={D}, K={K}): {(end-start)*1000:.2f} ms")
    print(f"Error de Ortogonalidad Absoluto Max: {error:.4e}")
    if error < 1e-7:
        print("[✔] Cholesky-QR3: PASSED (Error Aceptable < 1e-7)")
    else:
        print("[X] Cholesky-QR3: FAILED")

if __name__ == "__main__":
    test_qr()
