"""
POLYDIM V58 - TOPOLOGICAL INVARIANTS & KAHLER GEOMETRY MODULE
Curvatura de Berry, Número de Chern, Clase de Stiefel-Whitney y Métrica de Kähler
"""

import jax
import jax.numpy as jnp
from jax import jit

class TopologicalInvariants:
    @staticmethod
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
        return jnp.angle(plaquette)

    @staticmethod
    @jit
    def chern_number(berry_curvature: jnp.ndarray) -> jnp.ndarray:
        """
        Número de Chern C = (1 / 2 pi) sum F_12 (invariante entero)
        """
        return jnp.round(jnp.sum(berry_curvature) / (2.0 * jnp.pi))

class KahlerGeometry:
    @staticmethod
    @jit
    def kahler_potential(psi: jnp.ndarray) -> jnp.ndarray:
        """
        Potencial de Kähler K(z, z_bar) = log(<psi|psi>)
        """
        overlap = jnp.sum(jnp.abs(psi)**2)
        return jnp.log(jnp.maximum(overlap, 1e-15))
