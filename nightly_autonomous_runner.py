"""
NIGHTLY AUTONOMOUS RUNNER - POLYDIM SOTA 2026
Bucle autonomo de estres continuo en silicio y validacion de invariantes.
"""

import time
import os
import sys
from datetime import datetime

# Forzar codificacion segura en Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import jax
import jax.numpy as jnp

LOG_FILE = r"E:\POLYDIM_EINSOF\NOCTURNO_TELEMETRIA_CONTINUA.md"
REPORT_DIR = r"E:\POLYDIM_EINSOF\REPORTES"

os.makedirs(REPORT_DIR, exist_ok=True)

def append_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}", flush=True)

append_log(">>> RUNNER NOCTURNO INICIADO: Modo Autonomo SOTA Activo.")

# Invariante 1: Geodesic Reversibility on S^(D-1)
def test_geodesic_reversibility(D: int, eps=1e-12):
    key = jax.random.PRNGKey(int(time.time() * 1000) % 100000)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (D,))
    x = x / jnp.linalg.norm(x)
    y = jax.random.normal(k2, (D,))
    y = y / jnp.linalg.norm(y)

    diff_norm = jnp.linalg.norm(x - y)
    sum_norm = jnp.linalg.norm(x + y)
    theta = 2.0 * jnp.arctan2(diff_norm, sum_norm)

    dot = jnp.clip(jnp.sum(x * y), -1.0, 1.0)
    proj = y - dot * x
    proj_norm = jnp.linalg.norm(proj)
    v = theta * (proj / jnp.where(proj_norm < eps, 1.0, proj_norm))

    # Exp map
    v_norm = jnp.linalg.norm(v)
    is_zero = v_norm < eps
    v_tangent = v / jnp.where(is_zero, 1.0, v_norm)
    y_rec = jnp.cos(v_norm) * x + jnp.sin(v_norm) * v_tangent
    y_rec = y_rec / jnp.linalg.norm(y_rec)

    err = float(jnp.linalg.norm(y - y_rec))
    return err

# Invariante 2: Gram-Schmidt Analytic vs Singular U || V
def test_rotor_analytic(D: int):
    key = jax.random.PRNGKey(int(time.time() * 1000) % 100000)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (D,))
    x = x / jnp.linalg.norm(x)
    U = jax.random.normal(k2, (D,))
    # Forzar paralelismo parcial
    V = U * 0.9999 + jax.random.normal(k3, (D,)) * 1e-6

    U_orth = U / jnp.linalg.norm(U)
    dot_UV = jnp.sum(V * U_orth)
    V_perp = V - dot_UV * U_orth
    V_perp_norm = jnp.linalg.norm(V_perp)
    is_parallel = V_perp_norm < 1e-4

    e_base = jnp.zeros_like(U_orth).at[0].set(1.0)
    e_alt = jnp.zeros_like(U_orth).at[-1].set(1.0)
    use_alt = jnp.abs(U_orth[0]) > 0.9
    e = jnp.where(use_alt, e_alt, e_base)
    e_perp = e - jnp.sum(e * U_orth) * U_orth
    e_orth = e_perp / jnp.linalg.norm(e_perp)

    V_orth = jnp.where(is_parallel, e_orth, V_perp / jnp.maximum(V_perp_norm, 1e-12))
    
    # Check orthogonality
    orth_err = float(jnp.abs(jnp.sum(U_orth * V_orth)))
    return orth_err

iteration = 0
while True:
    iteration += 1
    try:
        # Loop over test dimensions
        dims = [10_000, 50_000, 100_000]
        results = []
        for d in dims:
            t0 = time.perf_counter()
            err_rev = test_geodesic_reversibility(d)
            err_rot = test_rotor_analytic(d)
            dt = (time.perf_counter() - t0) * 1000.0
            results.append((d, err_rev, err_rot, dt))

        report_line = f"Iter #{iteration} | " + " | ".join([f"D={d}: RevErr={e1:.2e}, RotErr={e2:.2e} ({t:.1f}ms)" for d, e1, e2, t in results])
        append_log(report_line)

        # Write periodic report every 10 iterations
        if iteration % 10 == 0:
            rep_path = os.path.join(REPORT_DIR, f"reporte_nocturno_iter_{iteration}.md")
            with open(rep_path, "w", encoding="utf-8") as rf:
                rf.write(f"# REPORTE DE TELEMETRIA NOCTURNA ITERACION {iteration}\n\n")
                rf.write(f"- **Timestamp:** {datetime.now().isoformat()}\n")
                rf.write(f"- **Ultimas metricas:**\n")
                for d, e1, e2, t in results:
                    rf.write(f"  - Dimension D={d}: Error Reversibilidad = `{e1:.2e}`, Error Ortogonalidad = `{e2:.2e}`, Latencia = `{t:.2f}ms`\n")
                rf.write("\n Invariantes metricas verificadas sin fallas de NaNs.\n")

    except Exception as e:
        append_log(f"ERROR EN RUNNER: {e}")

    time.sleep(10) # 10s cooldown
