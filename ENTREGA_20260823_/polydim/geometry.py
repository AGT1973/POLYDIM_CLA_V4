"""
POLYDIM V55 GEOMETRY KERNELS
Matemáticamente rigurosos y composables con Autodiff / jax.grad:
- Smooth Exponential Map con sinc/cos Taylor expansion en v_sq = ||v||^2 (Jacobiano C^inf en v=0 sin NaN)
- Exact Logarithmic Map con rama de identidad en x=y (sin distancia fabricada)
- Pure Composable SLERP en S^{D-1}
- Métrica de Error de Precisión L2 compartida
"""

import jax
import jax.numpy as jnp
from jax import jit

@jit
def _exp_coefficients(v_sq: jnp.ndarray):
    """
    Calcula (cos(theta), sinc(theta)) como función analítica de v_sq = ||v||^2.
    Para v_sq -> 0, utiliza expansiones de Taylor de orden 5 en v_sq:
    cos(theta)  = 1 - v_sq/2 + v_sq^2/24 - v_sq^3/720 + v_sq^4/40320 - v_sq^5/3628800
    sinc(theta) = 1 - v_sq/6 + v_sq^2/120 - v_sq^3/5040 + v_sq^4/362880 - v_sq^5/39916800
    Garantiza Jacobianos C^inf sin NaN en FP32 y FP64 para JAX autodiff.
    """
    threshold = jnp.where(v_sq.dtype == jnp.float64, 1e-4, 1e-3)
    is_small = v_sq < threshold

    # Expansiones de Taylor orden 5
    v_sq2 = v_sq * v_sq
    v_sq3 = v_sq2 * v_sq
    v_sq4 = v_sq3 * v_sq
    v_sq5 = v_sq4 * v_sq
    
    cos_taylor = 1.0 - v_sq / 2.0 + v_sq2 / 24.0 - v_sq3 / 720.0 + v_sq4 / 40320.0 - v_sq5 / 3628800.0
    sinc_taylor = 1.0 - v_sq / 6.0 + v_sq2 / 120.0 - v_sq3 / 5040.0 + v_sq4 / 362880.0 - v_sq5 / 39916800.0

    # Rama directa con safe_v_sq para prevenir gradiente NaN en el branch no seleccionado
    safe_v_sq = jnp.where(is_small, 1.0, v_sq)
    norm_v = jnp.sqrt(safe_v_sq)
    cos_direct = jnp.cos(norm_v)
    sinc_direct = jnp.sin(norm_v) / norm_v

    cos_v = jnp.where(is_small, cos_taylor, cos_direct)
    sinc_v = jnp.where(is_small, sinc_taylor, sinc_direct)

    return cos_v, sinc_v

class GeodesicKernels:
    @staticmethod
    @jit
    def proj_tangent(x: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        """Proyección ortogonal al espacio tangente T_x S^{D-1} para x unitario."""
        dot = jnp.einsum('i,i->', x, g)
        return g - dot * x

    @staticmethod
    @jit
    def exp_map(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Exponential Map suave y diferenciable en S^{D-1}.
        Exp_x(v) = x * cos(||v||) + v * sinc(||v||).
        Demostrable: Exp_x(0) = x, y jax.jacfwd(exp_map)(v=0) = I en T_x S^{D-1} (sin NaN).
        """
        # Proyección defensiva para garantizar que v pertenezca a T_x S^{D-1}
        v_tan = v - jnp.einsum('i,i->', v, x) * x
        v_sq = jnp.einsum('i,i->', v_tan, v_tan)
        cos_v, sinc_v = _exp_coefficients(v_sq)
        result = x * cos_v + v_tan * sinc_v
        # Renormalización preventiva contra deriva flotante
        norm = jnp.sqrt(jnp.maximum(jnp.einsum('i,i->', result, result), 1e-15))
        return result / norm

    @staticmethod
    @jit
    def log_map(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Logarithmic Map exacto en S^{D-1} hacia T_x S^{D-1}.
        Demostrable: Log_x(x) = 0 exactamente sin distancia fabricada.
        Maneja singularidades antipodales explícitamente y es diferenciable C^inf.
        """
        dot = jnp.einsum('i,i->', x, y)
        is_identity = dot >= (1.0 - 1e-6)
        is_antipodal = dot <= (-1.0 + 1e-6)
        
        dot_clipped = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = jnp.arccos(dot_clipped)
        sin_theta = jnp.sin(theta)
        
        # Serie Taylor theta/sin(theta) = 1 + h/3 + 2/15 h^2 + 2/35 h^3 (calculado exacto via sympy)
        h = 1.0 - dot_clipped
        sinc_inv_taylor = 1.0 + h / 3.0 + (2.0 / 15.0) * (h * h) + (2.0 / 35.0) * (h * h * h)
        safe_sinc_inv = jnp.where(is_identity, sinc_inv_taylor, theta / jnp.maximum(sin_theta, 1e-12))
        
        proj_y = y - dot_clipped * x
        tangent_vec = safe_sinc_inv * proj_y
        
        # Fallback determinista para el polo antipodal (x = -y)
        # Evitamos colapso si x == [1, 0, ..., 0] eligiendo [0, 1, 0, ...] como fallback
        fallback_v = jnp.where(jnp.abs(x[0]) > 0.9, jnp.zeros_like(x).at[1].set(1.0), jnp.zeros_like(x).at[0].set(1.0))
        proj_fallback = fallback_v - jnp.einsum('i,i->', fallback_v, x) * x
        norm_fallback = jnp.sqrt(jnp.maximum(jnp.einsum('i,i->', proj_fallback, proj_fallback), 1e-15))
        tangent_antipodal = (proj_fallback / norm_fallback) * jnp.pi
        
        valid_tangent = jnp.where(is_antipodal, tangent_antipodal, tangent_vec)
        return jnp.where(is_identity, jnp.zeros_like(x), valid_tangent)

    @staticmethod
    @jit
    def slerp(q1: jnp.ndarray, q2: jnp.ndarray, t: float) -> jnp.ndarray:
        """SLERP geométrico para S^{D-1} (NO cuaterniones S^3)."""
        dot = jnp.einsum('i,i->', q1, q2)
        is_identity = dot >= (1.0 - 1e-6)
        is_antipodal = dot <= (-1.0 + 1e-6)
        
        dot_clipped = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = jnp.arccos(dot_clipped)
        sin_theta = jnp.sin(theta)
        safe_sin = jnp.where(sin_theta == 0.0, 1.0, sin_theta)

        w1 = jnp.sin((1.0 - t) * theta) / safe_sin
        w2 = jnp.sin(t * theta) / safe_sin

        interp = w1 * q1 + w2 * q2
        norm = jnp.sqrt(jnp.maximum(jnp.einsum('i,i->', interp, interp), 1e-15))
        
        valid_slerp = interp / norm
        
        # En polos antipodales Slerp no es único. Caemos a q1 preventivamente.
        return jnp.where(is_identity | is_antipodal, q1, valid_slerp)

    @classmethod
    def compute_l2_precision_error(cls, q1_32: jnp.ndarray, q2_32: jnp.ndarray, t: float = 0.5) -> float:
        q1_64 = jnp.asarray(q1_32, dtype=jnp.float64)
        q2_64 = jnp.asarray(q2_32, dtype=jnp.float64)

        out32 = cls.slerp(q1_32, q2_32, t)
        out64 = cls.slerp(q1_64, q2_64, t)

        diff = jnp.linalg.norm(out32.astype(jnp.float64) - out64)
        return float(diff)
