"""
POLYDIM GRASSMANNIAN IMPLICIT HODGE DUAL
Proyección ortogonal sobre la variedad de Grassmann Gr(r, D):
P_{V^perp}(x) = x - V (V^T x).
Evita la explosión combinatoria C(D, k) de k-formas de alto grado.
 Complejidad: O(r * D).
"""

import jax
import jax.numpy as jnp
from jax import jit

class GrassmannianHodge:
    @staticmethod
    @jit
    def grassmann_projector(x: jnp.ndarray, V_k: jnp.ndarray) -> jnp.ndarray:
        """
        Proyección Ortogonal en Gr(k, D).
        V_k in R^{D x k} representa la k-frame ortonormal.
        Acción del complemento ortogonal V_k^perp sobre x in S^{D-1}.
        """
        v_tx = jnp.einsum('dk,d->k', V_k, x)
        proj_v = jnp.einsum('dk,k->d', V_k, v_tx)
        star_x = x - proj_v
        
        # Eliminada normalización falsa. Si pertenece al subespacio -> 0
        # Esto es lineal, idempotente P(P(x)) = P(x) y puro.
        return star_x
