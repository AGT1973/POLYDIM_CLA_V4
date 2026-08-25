import jax
import jax.numpy as jnp
import traceback
import sys

# Importar las funciones de nuestro monolito purgado
# Extraemos el código mediante import dinámico para probarlo
import importlib.util
spec = importlib.util.spec_from_file_location("polydim", "polydim_v62_monolito.py")
polydim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(polydim)

# Simular GeodesicKernels y demás
GeodesicKernels = polydim.GeodesicKernels
fubini_study_distance = polydim.TopologicalInvariants.fubini_study_distance
chern_number = polydim.TopologicalInvariants.chern_number

print("================================================================================")
print("  POLYDIM V62 - STRESS TESTER & FUZZER ADVERSARIAL")
print("================================================================================\n")

def run_fuzzer(name, func, *args):
    print(f"[!] Fuzzeando {name}...")
    try:
        res = func(*args)
        jax.block_until_ready(res)
        if jnp.any(jnp.isnan(res)) or jnp.any(jnp.isinf(res)):
            print(f"  [X] FALLO: {name} produjo NaN/Inf con inputs extremos.")
            return False
        else:
            print(f"  [OK] {name} sobrevivió.")
            return True
    except Exception as e:
        print(f"  [X] FALLO EXCEPCIÓN: {name} - {str(e)}")
        return False

# 1. ATAQUE: Fubini-Study con overlap = 1.0 + epsilon (Fuera de rango por FP inexactitud)
print(">>> ATAQUE 1: Fubini-Study con estados idénticos + ruido de coma flotante")
q1 = jnp.array([1.0, 0.0], dtype=jnp.complex64)
q2 = jnp.array([1.0000001, 0.0], dtype=jnp.complex64) # No normalizado a propósito
run_fuzzer("fubini_study_distance (overlap > 1)", fubini_study_distance, q1, q2)

# 2. ATAQUE: Log_Map en la singularidad antipodal exacta
print("\n>>> ATAQUE 2: Log_Map en Polo Antipodal (x = -y)")
x = jnp.array([1.0, 0.0, 0.0, 0.0])
y = jnp.array([-1.0, 0.0, 0.0, 0.0])
run_fuzzer("log_map (Antipodal)", GeodesicKernels.log_map, x, y)

# 3. ATAQUE: Log_Map en la Identidad con Subnormales (h = 1e-15)
print("\n>>> ATAQUE 3: Log_Map con vectores virtualmente idénticos")
x = jnp.array([1.0, 0.0, 0.0, 0.0])
y = jnp.array([1.0, 1e-15, 0.0, 0.0])
y = y / jnp.linalg.norm(y)
run_fuzzer("log_map (Identidad extrema)", GeodesicKernels.log_map, x, y)

# 4. ATAQUE: MERA con dimensiones degeneradas y dispares
print("\n>>> ATAQUE 4: MERA con dimensionamiento inconsistente")
state = jnp.ones(4) / 2.0
unitary = jnp.eye(4)  # 4x4
isometry = jnp.ones((2, 4)) # 2x4
run_fuzzer("mera_disentangle_and_coarsen", polydim.HolographicDuality.mera_disentangle_and_coarsen, state, unitary, isometry)

# 5. ATAQUE: Density Matrix con Batch gigante (D=10^4, B=100) -> Test OOM
print("\n>>> ATAQUE 5: Matriz de Densidad (Stress OOM Batching)")
try:
    B = 100
    D = 1000
    batch_states = jax.random.normal(jax.random.PRNGKey(0), (B, D), dtype=jnp.complex64)
    res = polydim.QuantumInformation.density_matrix(batch_states)
    jax.block_until_ready(res)
    if res.shape == (B, D, D):
        print(f"  [OK] density_matrix retuvo dimensiones de batch {res.shape} sin OOM.")
    else:
        print(f"  [X] FALLO: shape incorrecto {res.shape}")
except Exception as e:
    print(f"  [X] FALLO OOM: {e}")

# 6. ATAQUE: Transformada de Cayley con singularidad -1
print("\n>>> ATAQUE 6: Transformada de Cayley con rotación de 180°")
A = jnp.array([[0.0, jnp.pi], [-jnp.pi, 0.0]]) # Generador que da exp(A) = -I
# Transformada de Cayley es para so(D), vamos a meterle valores masivos
A_massive = A * 1e5
run_fuzzer("cayley_transform", polydim.LieGroupOperators.cayley_transform, A_massive)

print("\n================================================================================")
print("  FUZZING COMPLETADO")
print("================================================================================")
