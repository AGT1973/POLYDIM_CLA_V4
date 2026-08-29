"""
POLYDIM V58 - LIE GROUPS & LIE ALGEBRAS MODULE
Operadores en Grupos de Lie SO(D), Transformada de Cayley y Exponencial de Rodrigues
"""

import jax
import jax.numpy as jnp
from jax import jit

class LieGroupOperators:
    @staticmethod
    @jit
    def skew_symmetric_projector(A: jnp.ndarray) -> jnp.ndarray:
        """
        Proyecta una matriz A a la álgebra de Lie so(D): (A - A^T) / 2
        """
        return 0.5 * (A - jnp.swapaxes(A, -1, -2))

    @staticmethod
    @jit
    def cayley_transform(A: jnp.ndarray) -> jnp.ndarray:
        """
        Transformada de Cayley R = (I - A)^{-1} (I + A) para A in so(D).
        Usa proyección SVD para garantizar det(R) = +1.
        """
        A_skew = LieGroupOperators.skew_symmetric_projector(A)
        dim = A.shape[-1]
        I = jnp.eye(dim, dtype=A.dtype)
        R_raw = jnp.linalg.solve(I - A_skew, I + A_skew)
        
        # Proyección SVD para forzar SO(D) puro (det = +1)
        U, S, Vh = jnp.linalg.svd(R_raw, full_matrices=False)
        R_proj = U @ Vh
        
        # Forzar determinante +1 corrigiendo el último vector singular
        det_sign = jnp.sign(jnp.linalg.det(R_proj))
        D = jnp.diag(jnp.where(jnp.arange(S.shape[0]) == S.shape[0] - 1, det_sign, 1.0))
        return U @ D @ Vh

    @staticmethod
    @jit
    def lie_exp_so_d(A: jnp.ndarray) -> jnp.ndarray:
        """
        Mapa exponencial en SO(D) usando matriz exponencial por Padé (jax.scipy.linalg.expm)
        """
        A_skew = LieGroupOperators.skew_symmetric_projector(A)
        return jax.scipy.linalg.expm(A_skew)

    @staticmethod
    @jit
    def lie_log_so_d(R: jnp.ndarray) -> jnp.ndarray:
        """
        Transformada inversa de Cayley: A = (R - I)(R + I)^{-1}
        Con regularización Tikhonov en el polo -1 (rotaciones de pi) para evitar NaNs.
        """
        dim = R.shape[-1]
        I = jnp.eye(dim, dtype=R.dtype)
        
        # Epsilon dinámico para estabilizar la inversión cerca de R = -I
        reg_eps = 1e-6
        R_safe = R + reg_eps * I
        
        A_approx = jnp.linalg.solve(R_safe + I, R - I)
        return LieGroupOperators.skew_symmetric_projector(A_approx)
