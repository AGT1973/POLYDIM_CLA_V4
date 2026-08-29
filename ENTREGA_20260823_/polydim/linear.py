"""
POLYDIM V55 LINEAR OPERATORS & GRASSMANNIAN PROJECTORS
Operadores lineales exactos con invariantes matemáticos demostrables:
- HouseholderReflection: H_v^2 = I, ||H_v(x)|| = ||x|| (sin normalizaciones falsas)
- OrthogonalProjector: P^2 = P (idempotente), P(x) = 0 si x in span(Q) (sin NaN)
- SkewLowRankUpdate: Operador antisimétrico B = U V^T - V U^T en so(D) de rango <= 2r
"""

import jax
import jax.numpy as jnp
from jax import jit

class HouseholderReflection:
    @staticmethod
    @jit
    def reflect(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Reflexión de Householder lineal exacta H_v(x) = x - 2 (u^T x) u con u = v / ||v||.
        Propiedades demostrables: H_v(H_v(x)) = x, ||H_v(x)|| = ||x|| para todo x.
        """
        vv = jnp.einsum('i,i->', v, v)
        safe_norm = jnp.sqrt(jnp.maximum(vv, 1e-15))
        u = v / safe_norm
        dot = jnp.einsum('i,i->', u, x)
        
        # H_v(0) = 0 exactamente si x=0
        reflected = x - 2.0 * dot * u
        return jnp.where(vv < 1e-15, x, reflected)


class OrthogonalProjector:
    @staticmethod
    @jit
    def project_orthogonal(x: jnp.ndarray, Q_k: jnp.ndarray) -> jnp.ndarray:
        """
        Proyector Ortogonal lineal en Gr(k, D): P(x) = x - Q_k (Q_k^T x).
        Q_k in R^{D x k} debe tener columnas ortonormales (Q_k^T Q_k = I).
        Propiedades demostrables: P(P(x)) = P(x) (Idempotente), P(x) = 0 si x in span(Q_k) sin NaN.
        """
        q_tx = jnp.einsum('dk,d->k', Q_k, x)
        proj_q = jnp.einsum('dk,k->d', Q_k, q_tx)
        return x - proj_q


class SkewLowRankUpdate:
    @staticmethod
    @jit
    def apply_skew_update(x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray, scale: float = 0.1) -> jnp.ndarray:
        """
        Actualización lineal antisimétrica B = U V^T - V U^T in so(D) de rango <= 2r.
        B x = U (V^T x) - V (U^T x).
        Aproximación de Lie de 1er orden: x_new = x - scale * B x.
        """
        v_tx = jnp.einsum('dr,d->r', V, x)
        u_tx = jnp.einsum('dr,d->r', U, x)
        bx = jnp.einsum('dr,r->d', U, v_tx) - jnp.einsum('dr,r->d', V, u_tx)
        return x - scale * bx
