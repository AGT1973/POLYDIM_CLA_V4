"""
POLYDIM V58 - QUANTUM INFORMATION MODULE
Matrices de Densidad, Entropía de Von Neumann y Entropía de Entrelazamiento
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

class QuantumInformation:
    @staticmethod
    @jit
    def density_matrix(pure_state: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la matriz de densidad de un estado puro rho = |psi><psi|
        """
        norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(pure_state)**2), 1e-15))
        psi_norm = pure_state / norm
        return jnp.outer(psi_norm, jnp.conj(psi_norm))

    @staticmethod
    @jit
    def von_neumann_entropy(rho: jnp.ndarray) -> jnp.ndarray:
        """
        Entropía de Von Neumann S(rho) = -Tr(rho log2 rho)
        Maneja autovalores nulos o subnormales sin producir NaNs (0 * log(0) -> 0).
        """
        eigenvalues = jnp.linalg.eigvalsh(rho)
        # Cortar a [0, 1] asegura que no haya entropía negativa por FP32 overshoot
        bounded_evs = jnp.clip(eigenvalues, 0.0, 1.0)
        # xlogy(x, y) = x * log(y), exacto y retorna 0 si x=0
        entropy_terms = jax.scipy.special.xlogy(bounded_evs, bounded_evs) / jnp.log(2.0)
        return -jnp.sum(entropy_terms)

    @staticmethod
    @jit
    def purity(rho: jnp.ndarray) -> jnp.ndarray:
        """
        Pureza gamma = Tr(rho^2)
        """
        return jnp.real(jnp.trace(rho @ rho))

    @staticmethod
    @partial(jit, static_argnums=(1, 2))
    def partial_trace_bipartite(rho: jnp.ndarray, dim_a: int, dim_b: int) -> jnp.ndarray:
        """
        Traza parcial sobre el subsistema B en un sistema bipartito H_A tensor H_B.
        rho_tensor tiene forma (dim_a, dim_b, dim_a, dim_b).
        Traza sobre B: i_B = j_B = k -> einsum('ikjk->ij', rho_tensor)
        """
        rho_tensor = rho.reshape((dim_a, dim_b, dim_a, dim_b))
        return jnp.einsum('ikjk->ij', rho_tensor)

    @staticmethod
    @partial(jit, static_argnums=(1, 2))
    def entanglement_entropy(pure_state: jnp.ndarray, dim_a: int, dim_b: int) -> jnp.ndarray:
        """
        Entropía de entrelazamiento de Von Neumann del subsistema A
        """
        rho = QuantumInformation.density_matrix(pure_state)
        rho_a = QuantumInformation.partial_trace_bipartite(rho, dim_a, dim_b)
        return QuantumInformation.von_neumann_entropy(rho_a)
