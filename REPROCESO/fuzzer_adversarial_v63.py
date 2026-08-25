import jax
import jax.numpy as jnp
import numpy as np
import time
import random
import importlib.util
import os

# Importar dinámicamente V63 desde ENTREGA
spec = importlib.util.spec_from_file_location("polydim_v63", r"E:\POLYDIM_EINSOF\ENTREGA_20260824_\polydim_v63_monolito.py")
v63 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v63)

def run_stochastic_fuzz():
    seed = random.randint(1, 1000000)
    key = jax.random.PRNGKey(seed)
    print(f"=== RUNNING STOCHASTIC FUZZER V63 (SEED: {seed}) ===")
    
    # 1. Random High Dimension Slerp
    dim = random.choice([5000, 10000, 50000, 100000])
    k1, k2 = jax.random.split(key)
    q1 = jax.random.normal(k1, (dim,), dtype=jnp.float32)
    q2 = jax.random.normal(k2, (dim,), dtype=jnp.float32)
    q1 = q1 / jnp.linalg.norm(q1)
    q2 = q2 / jnp.linalg.norm(q2)
    
    t = random.random()
    res = v63.GeodesicKernels.slerp(q1, q2, t)
    jax.block_until_ready(res)
    norm = float(jnp.linalg.norm(res))
    assert abs(norm - 1.0) < 1e-4, f"Fuzz SLERP norm error: {norm}"
    
    # 2. Random Disk Persist
    tmp_path = f"E:\\POLYDIM_EINSOF\\REPROCESO\\fuzz_{seed}.pmtp"
    arr = np.random.randn(dim).astype(np.float32)
    v63.PMTPPersistentStorage.save_tensor(tmp_path, arr)
    loaded = v63.PMTPPersistentStorage.load_tensor(tmp_path)
    assert np.allclose(arr, loaded), "Fuzz Disk Persist mismatch"
    if os.path.exists(tmp_path): os.remove(tmp_path)
    
    # 3. Device Transfer Fuzz
    np_rand = np.random.randn(100, 100).astype(np.float32)
    g_arr = v63.DeviceTransferManager.to_gpu(np_rand)
    c_arr = v63.DeviceTransferManager.to_cpu(g_arr)
    assert np.allclose(np_rand, c_arr), "DeviceTransfer Fuzz mismatch"

    print(f"[OK] Fuzz Seed {seed} passed on D={dim}")

if __name__ == "__main__":
    run_stochastic_fuzz()
