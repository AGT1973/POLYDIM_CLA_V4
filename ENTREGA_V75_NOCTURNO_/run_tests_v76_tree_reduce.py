import os
import time
import jax
import jax.numpy as jnp
import numpy as np

# Silenciar advertencias
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
jax.config.update("jax_enable_x64", True)

try:
    from polydim_v75_monolito import XLAQuantizer
except ImportError:
    print("Error: Monolito V75 no encontrado o sin parche XLAQuantizer.")
    exit(1)

def test_tree_reduce():
    print("=========================================================")
    print("🚀 INICIANDO PRUEBAS EMPÍRICAS V76 (XLA TREE-REDUCE) 🚀")
    print("=========================================================")
    
    # Crear un tensor masivo (simulando 1M dimensiones)
    D = 1000000
    print(f"-> Generando tensor denso D={D}...")
    tensor = jax.random.normal(jax.random.PRNGKey(42), (D,), dtype=jnp.float64) * 100.0
    
    # Trigger JIT compile (Warmup)
    print("-> Compilando kernel Tree-Reduce fusionado...")
    _ = XLAQuantizer.quantize_int8_tree_reduce(jnp.array([1.0, -1.0], dtype=jnp.float64))
    
    # Ejecutar medición
    start = time.perf_counter()
    quantized, scale = XLAQuantizer.quantize_int8_tree_reduce(tensor)
    quantized.block_until_ready() # Forzar sincronización de XLA
    end = time.perf_counter()
    
    # Validar
    max_val = float(scale * 127.0)
    print(f"-> Escala calculada: {scale:.4f} (Max Abs: {max_val:.4f})")
    print(f"-> Shape cuantizado: {quantized.shape} | Tipo: {quantized.dtype}")
    print(f"-> Latencia Tree-Reduce: {(end-start)*1000:.2f} ms")
    
    if str(quantized.dtype) == "int8":
        print("-> [✔] XLA Tree-Reduce Test: PASSED")
    else:
        print("-> [X] XLA Tree-Reduce Test: FAILED")

if __name__ == "__main__":
    test_tree_reduce()
