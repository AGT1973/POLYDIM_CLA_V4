"""
POLYDIM V58 - RIEMANNIAN LEARNING MODULE
Gradiente Riemanniano, SGD Riemanniano en S^{D-1} y Transporte Paralelo
"""

import jax
import jax.numpy as jnp
from jax import jit

class RiemannianLearning:
    @staticmethod
    @jit
    def riemannian_gradient(euclidean_grad: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """
        Gradiente Riemanniano en la esfera S^{D-1}: grad_R f = grad_E f - <grad_E f, x> x
        """
        norm_sq = jnp.sum(x**2)
        x_norm = x / jnp.sqrt(jnp.maximum(norm_sq, 1e-15))
        inner = jnp.sum(euclidean_grad * x_norm)
        return euclidean_grad - inner * x_norm

    @staticmethod
    @jit
    def riemannian_sgd_step(x: jnp.ndarray, euclidean_grad: jnp.ndarray, lr: float = 0.01) -> jnp.ndarray:
        """
        Paso de optimización SGD Riemanniano sobre S^{D-1} usando retracción exponencial (normalización L2)
        """
        r_grad = RiemannianLearning.riemannian_gradient(euclidean_grad, x)
        x_updated = x - lr * r_grad
        return x_updated / jnp.sqrt(jnp.maximum(jnp.sum(x_updated**2), 1e-15))

    @staticmethod
    @jit
    def parallel_transport(v: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transporte paralelo exacto (Levi-Civita) del vector v in T_x S^{D-1} a T_y S^{D-1}.
        Garantiza que el resultado w pertenezca a T_y S^{D-1} (i.e. <w, y> = 0).
        """
        dot = jnp.clip(jnp.sum(x * y), -1.0 + 1e-15, 1.0 - 1e-15)
        theta = jnp.arccos(dot)
        sin_theta = jnp.sin(theta)
        cos_theta = dot
        is_identity = dot >= (1.0 - 1e-6)
        
        h = 1.0 - dot
        # Expansión de Taylor analítica para theta/sin(theta) en términos de h = 1 - cos(theta)
        sinc_inv_taylor = 1.0 + h / 3.0 + (2.0 / 15.0) * (h * h) + (2.0 / 35.0) * (h * h * h)
        safe_sinc_inv = jnp.where(is_identity, sinc_inv_taylor, theta / jnp.maximum(sin_theta, 1e-12))
        
        log_xy = safe_sinc_inv * (y - dot * x)
        inner_v_log = jnp.sum(v * log_xy)
        
        safe_theta = jnp.maximum(theta, 1e-15)
        safe_theta_sq = jnp.maximum(theta**2, 1e-15)
        
        term_x = (sin_theta / safe_theta) * x
        term_log = ((1.0 - cos_theta) / safe_theta_sq) * log_xy
        
        transported = v - inner_v_log * (term_x + term_log)
        
        # En caso límite donde x==y, devolver v sin tocar
        return jnp.where(is_identity, v, transported)

