"""
POLYDIM CLIFFORD ROTORS (SPIN(D) RANK-R)
Rotaciones de Lie so(D) en S^{D-1} parametrizadas por Bivectores de Bajo Rango:
B = U V^T - V U^T, con U, V in R^{D x r}, r << D.
Complejidad: O(r * D) en lugar de O(D^2) o O(2^D).
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import jit

class CliffordRotors:
    @staticmethod
    @jit
    def apply_low_rank_rotor(x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica un Rotor de Clifford Spin(D) Rank-r exacto en O(r^2 D + r^3) sobre x in S^{D-1}.
        Bivector B = U V^T - V U^T.
        Calcula exp(B)x exactamente mediante proyección a la base ortonormal Q in R^{D x 2r}.
        """
        W = jnp.concatenate([U, V], axis=-1)
        Q, _ = jnp.linalg.qr(W)
        
        QtU = jnp.einsum('dk,dr->kr', Q, U)
        QtV = jnp.einsum('dk,dr->kr', Q, V)
        M_2r = jnp.einsum('kr,lr->kl', QtU, QtV) - jnp.einsum('kr,lr->kl', QtV, QtU)
        
        R_2r = jax.scipy.linalg.expm(M_2r)
        
        q_tx = jnp.einsum('dk,d->k', Q, x)
        rot_q = jnp.einsum('kl,l->k', R_2r - jnp.eye(R_2r.shape[0], dtype=x.dtype), q_tx)
        x_rot = x + jnp.einsum('dk,k->d', Q, rot_q)
        
        norm_sq = jnp.einsum('i,i->', x_rot, x_rot)
        safe_norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-15))
        return jnp.where(norm_sq < 1e-15, x, x_rot / safe_norm)

    @staticmethod
    @jit
    def householder_reflection(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Reflexión de Householder H_v(x) = x - 2 (v^T x / v^T v) v sobre S^{D-1}."""
        vv = jnp.einsum('i,i->', v, v)
        safe_vv = jnp.maximum(vv, 1e-15)
        dot = jnp.einsum('i,i->', v, x)
        out = x - 2.0 * (dot / safe_vv) * v
        norm_sq = jnp.einsum('i,i->', out, out)
        safe_norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-15))
        return jnp.where(vv < 1e-15, x, out / safe_norm)
