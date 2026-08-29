"""
POLYDIM V58 - QUANTUM GEOMETRY MODULE
Geometría Cuántica: Fubini-Study Metric, Berry Phase y Proyección de Unitariedad
"""

import jax
import jax.numpy as jnp
from jax import jit

class QuantumGeodesicKernels:
    @staticmethod
    @jit
    def fubini_study_distance(psi1: jnp.ndarray, psi2: jnp.ndarray) -> jnp.ndarray:
        """
        Distancia de Fubini-Study en CP^{D-1}: s(psi1, psi2) = arccos(|<psi1, psi2>|)
        Garantiza estabilidad numérica en [0, pi/2] sin NaNs.
        """
        norm1 = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(psi1)**2), 1e-15))
        norm2 = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(psi2)**2), 1e-15))
        
        u1 = psi1 / norm1
        u2 = psi2 / norm2
        
        overlap = jnp.abs(jnp.sum(jnp.conj(u1) * u2))
        overlap_clipped = jnp.clip(overlap, 0.0, 1.0 - 1e-15)
        return jnp.arccos(overlap_clipped)

    @staticmethod
    @jit
    def geometric_phase(psi1: jnp.ndarray, psi2: jnp.ndarray) -> jnp.ndarray:
        """
        Fase geométrica de Berry entre dos estados cuánticos <psi1, psi2>
        """
        overlap = jnp.sum(jnp.conj(psi1) * psi2)
        return jnp.angle(overlap)

    @staticmethod
    @jit
    def project_to_unitary(U: jnp.ndarray) -> jnp.ndarray:
        """
        Proyecta una matriz arbitraria N x N a la matriz unitaria más cercana vía SVD: U_unit = U_svd @ V_svd^H
        """
        u, s, vh = jnp.linalg.svd(U, full_matrices=False)
        return u @ vh

    @staticmethod
    @jit
    def apply_unitary_gate(x: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica un operador unitario U a x, garantizando conservación de norma.
        """
        U_safe = QuantumGeodesicKernels.project_to_unitary(U)
        return jnp.dot(U_safe, x)
