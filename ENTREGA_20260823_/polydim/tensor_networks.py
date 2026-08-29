"""
POLYDIM V58 - TENSOR NETWORKS & HOLOGRAPHY MODULE
Redes Tensoriales MERA, Holografía y Entropía de Ryu-Takayanagi
"""

import jax
import jax.numpy as jnp
from jax import jit

class TensorNetwork:
    @staticmethod
    @jit
    def mera_disentangle_and_coarsen(state: jnp.ndarray, unitary: jnp.ndarray, isometry: jnp.ndarray) -> jnp.ndarray:
        """
        Capa MERA: Desentrelazamiento local seguido de compresión isométrica de grano grueso.
        """
        dim = state.shape[0]
        state_pairs = state.reshape(dim // 2, 2)
        disentangled = jnp.einsum('ij,kj->ki', unitary, state_pairs)
        coarsened = jnp.einsum('ij,kj->ki', isometry, disentangled)
        return coarsened.reshape(-1)

class HolographicDuality:
    @staticmethod
    @jit
    def ryu_takayanagi_entropy(subsystem_state: jnp.ndarray, ads_radius: float = 1.0) -> jnp.ndarray:
        """
        Entropía de Ryu-Takayanagi S = Area(gamma_A) / (4 G_N) basada en entrelazamiento de superficie minimal
        """
        norm_sq = jnp.sum(jnp.abs(subsystem_state)**2)
        minimal_area = 4.0 * jnp.pi * ads_radius**2 * jnp.log(jnp.maximum(norm_sq, 1e-15) + 1.0)
        return 0.25 * minimal_area
