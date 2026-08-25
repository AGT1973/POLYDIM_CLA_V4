import re

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Patch hermitian_inner (Coerción FP64)
    old_inner = '''@jit
def hermitian_inner(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """<a,b> real, correcto para real y complejo (conjuga el primer argumento)."""
    return jnp.real(jnp.vdot(a, b))'''

    new_inner = '''@jit
def hermitian_inner(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """<a,b> real. Parche SOTA: Coerción FP64 para evitar Cancelación Catastrófica en D=10^6."""
    # JAX config enable x64 is required for this to work
    jax.config.update("jax_enable_x64", True)
    prod = jnp.real(jnp.conj(a) * b)
    return jnp.sum(prod, axis=-1, dtype=jnp.float64).astype(a.dtype)'''

    if old_inner in content:
        content = content.replace(old_inner, new_inner)
        print(f"Patched hermitian_inner in {filepath}")

    # 2. Patch TopologicalInvariants.berry_curvature_2d
    old_berry = '''    @staticmethod
    @jit
    def berry_curvature_2d(psi_grid: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la curvatura de Berry en una malla 2D de estados cuantizados F_12 = Im log(U1 U2 U3 U4)
        psi_grid de forma (N1, N2, D)
        """
        # Plaquette link variables
        u1 = jnp.sum(jnp.conj(psi_grid) * jnp.roll(psi_grid, -1, axis=0), axis=-1)
        u2 = jnp.sum(jnp.conj(jnp.roll(psi_grid, -1, axis=0)) * jnp.roll(jnp.roll(psi_grid, -1, axis=0), -1, axis=1), axis=-1)
        u3 = jnp.sum(jnp.conj(jnp.roll(jnp.roll(psi_grid, -1, axis=0), -1, axis=1)) * jnp.roll(psi_grid, -1, axis=1), axis=-1)
        u4 = jnp.sum(jnp.conj(jnp.roll(psi_grid, -1, axis=1)) * psi_grid, axis=-1)
        
        plaquette = u1 * u2 * u3 * u4
        return jnp.angle(plaquette)'''

    new_berry = '''    @staticmethod
    @jit
    def berry_curvature_2d(psi_grid: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la curvatura de Berry en una malla 2D de estados cuantizados.
        PARCHE SOTA FHS: Gauge Fixing Estricto sobre variedad U(1).
        Evita ruido topológico por branch cuts y fases aleatorias del solver.
        """
        eps = 1e-15
        
        # 1. Overlaps adyacentes
        psi_x = jnp.roll(psi_grid, shift=-1, axis=0)
        psi_y = jnp.roll(psi_grid, shift=-1, axis=1)
        
        link_x_raw = jnp.sum(jnp.conj(psi_grid) * psi_x, axis=-1)
        link_y_raw = jnp.sum(jnp.conj(psi_grid) * psi_y, axis=-1)
        
        # 2. Gauge Fixing Estricto U(1)
        U_x = link_x_raw / (jnp.abs(link_x_raw) + eps)
        U_y = link_y_raw / (jnp.abs(link_y_raw) + eps)
        
        # 3. Flujo en la Plaqueta
        U_x_shifted_y = jnp.roll(U_x, shift=-1, axis=1)
        U_y_shifted_x = jnp.roll(U_y, shift=-1, axis=0)
        
        plaquette_product = U_x * U_y_shifted_x * jnp.conj(U_x_shifted_y) * jnp.conj(U_y)
        
        # jnp.angle garantiza confinamiento a (-pi, pi] (rama principal)
        return jnp.angle(plaquette_product)'''

    if old_berry in content:
        content = content.replace(old_berry, new_berry)
        print(f"Patched berry_curvature_2d in {filepath}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

patch_file('polydim_v62_monolito.py')
patch_file('codigo_consolidado_v62.txt')
