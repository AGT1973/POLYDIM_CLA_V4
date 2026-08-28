
# ==============================================================================
# 3. KERNELS GEOMÉTRICOS Y MATEMÁTICOS (SD-1)
# ==============================================================================
def safe_norm(x: jnp.ndarray, axis=-1, keepdims: bool = True) -> jnp.ndarray:
    eps = jnp.finfo(x.dtype).eps
    axis_t = (axis,) if isinstance(axis, int) else tuple(axis)
    scale = jnp.max(jnp.abs(x), axis=axis_t, keepdims=True)
    
    safe_scale = jnp.where(scale == 0.0, 1.0, scale)
    scaled_x = x / safe_scale
    
    # FIX V74.1: safe_norm maneja álgebra hermitiana estrictamente
    if x.dtype.kind == 'c':
        sq_sum = jnp.sum((scaled_x * jnp.conj(scaled_x)).real, axis=axis_t, keepdims=keepdims)
    else:
        sq_sum = jnp.sum(scaled_x * scaled_x, axis=axis_t, keepdims=keepdims)
        
    norm = scale * jnp.sqrt(jnp.where(scale == 0.0, 1.0, sq_sum))
    
    if not keepdims:
        norm = jnp.squeeze(norm, axis=axis_t)
    
    # La norma de un vector (real o complejo) SIEMPRE es un escalar real.
    real_dtype = jnp.finfo(x.dtype).dtype if x.dtype.kind == 'c' else x.dtype
    return norm.astype(real_dtype)


def safe_dot(a: jnp.ndarray, b: jnp.ndarray, keepdims: bool = False) -> jnp.ndarray:
    # Asegurar precisión (convertir enteros a float64)
    a = a.astype(jnp.float64) if not jnp.issubdtype(a.dtype, jnp.inexact) else a
    b = b.astype(jnp.float64) if not jnp.issubdtype(b.dtype, jnp.inexact) else b
    
    res = jnp.sum(a * b, axis=-1, keepdims=keepdims)
    return res


class GeodesicKernels:
    @staticmethod
    @jax.jit
    def exp_map(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        v_norm = safe_norm(v, keepdims=True)
        is_zero = v_norm < eps
        
        safe_v_norm = jnp.where(is_zero, 1.0, v_norm)
        v_tangent = v / safe_v_norm
        
        cos_t = jnp.cos(v_norm)
        sin_t = jnp.sin(v_norm)
        
        result = cos_t * x + sin_t * v_tangent
        result = jnp.where(is_zero, x, result)
        
        return result / jnp.maximum(safe_norm(result, keepdims=True), eps)

    @staticmethod
    @jax.jit
    def _log_map_unit(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        dot = jnp.clip(safe_dot(x, y, keepdims=True), -1.0, 1.0)
        
        # Expansión de Taylor para ángulos pequeños (Zonas muertas)
        # SOTA: arccos(1 - z) ~ sqrt(2z)
        theta = jnp.arccos(dot)
        
        proj = y - dot * x
        proj_norm = safe_norm(proj, keepdims=True)
        safe_proj_norm = jnp.where(proj_norm < eps, 1.0, proj_norm)
        
        return theta * (proj / safe_proj_norm)

    @staticmethod
    @jax.jit
    def log_map(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        xu = x / jnp.maximum(safe_norm(x, keepdims=True), eps)
        yu = y / jnp.maximum(safe_norm(y, keepdims=True), eps)
        dim = x.shape[-1]
        
        # FIX V74.1: Guarda de identidad por distancia Euclidiana
        dist_sq = jnp.sum((xu - yu)**2, axis=-1, keepdims=True)
        is_identity = dist_sq < (eps * dim) ** 2
        is_antipodal = dist_sq > (2.0 - eps * dim) ** 2
        
        log_normal = GeodesicKernels._log_map_unit(xu, yu)
        
        e0 = jnp.zeros_like(xu).at[..., 0].set(1.0)
        e1 = jnp.zeros_like(xu).at[..., -1].set(1.0)
        use_e1 = jnp.abs(xu[..., 0:1]) > 0.9
        e_base = jnp.where(use_e1, e1, e0)
        
        proj_e = e_base - safe_dot(e_base, xu, keepdims=True) * xu
        u_fallback = proj_e / jnp.maximum(safe_norm(proj_e, keepdims=True), eps)
        log_antipodal = jnp.pi * u_fallback
        
        result = jnp.where(is_antipodal, jax.lax.stop_gradient(log_antipodal), log_normal)
        result = jnp.where(is_identity, 0.0, result)
        
        return result

    @staticmethod
    @jax.jit
    def log_map_newton(x: jnp.ndarray, y: jnp.ndarray, max_iter: int = 10, tol: float = 1e-6) -> jnp.ndarray:
        # FIX V74.1: Convergencia adaptativa con while_loop
        eps = jnp.finfo(x.dtype).eps
        xu = x / jnp.maximum(safe_norm(x, keepdims=True), eps)
        yu = y / jnp.maximum(safe_norm(y, keepdims=True), eps)
        
        v0 = GeodesicKernels.log_map(xu, yu) # Bootstrapping
        
        def cond_fn(state):
            v, residual_norm, i = state
            return (residual_norm > tol) & (i < max_iter)
            
        def body_fn(state):
            v, _, i = state
            y_approx = GeodesicKernels.exp_map(xu, v)
            
            # Transporte inverso
            residual = GeodesicKernels._log_map_unit(y_approx, yu)
            c = safe_dot(y_approx, xu, keepdims=True)
            denom = jnp.maximum(1.0 + c, 1e-12)
            trans_res = residual - (safe_dot(residual, y_approx + xu, keepdims=True) / denom) * (y_approx + xu)
            
            v_new = v + trans_res
            y_check = GeodesicKernels.exp_map(xu, v_new)
            err = jnp.max(safe_norm(y_check - yu, keepdims=True))
            return (v_new, err, i + 1)
            
        init_err = jnp.max(safe_norm(GeodesicKernels.exp_map(xu, v0) - yu, keepdims=True))
        v_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (v0, init_err, jnp.array(0)))
        
        return v_final

class CliffordRotors:
    @staticmethod
    @jax.jit
    def apply_spherical_rotor(x: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray, theta: jnp.ndarray = jnp.array(0.1)) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        U = U[..., None] if U.ndim == 1 else U
        V = V[..., None] if V.ndim == 1 else V
        W = jnp.concatenate([U, V], axis=-1)
        
        # FIX V74.1: Ruido proporcional a la norma, sin PRNGKey estático
        w_norm = safe_norm(W, axis=-2, keepdims=True)
        W_reg = W + 1e-6 * w_norm * jnp.ones_like(W)
        
        # FIX V74.1: QR Memory-Safe (O(D) en lugar de O(D^2)) -> XLA Gram-Schmidt adaptado.
        Q, _ = jnp.linalg.qr(W_reg)
        U_orth = Q[..., :U.shape[-1]]
        V_orth = Q[..., U.shape[-1]:]
        
        # FIX V74.1: Alinear dimensiones batch explícitamente para einsum
        batch_ndim = x.ndim - 1
        if batch_ndim > 0 and U_orth.ndim == 2:
            U_orth = jnp.expand_dims(U_orth, axis=tuple(range(batch_ndim)))
            V_orth = jnp.expand_dims(V_orth, axis=tuple(range(batch_ndim)))
            
        dot_U = jnp.einsum('...dr,...d->...r', U_orth, x)
        dot_V = jnp.einsum('...dr,...d->...r', V_orth, x)
        
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot_U = c * dot_U - s * dot_V
        rot_V = s * dot_U + c * dot_V
        
        delta_U = (rot_U - dot_U)[..., None, :] * U_orth
        delta_V = (rot_V - dot_V)[..., None, :] * V_orth
        delta = jnp.sum(delta_U, axis=-1) + jnp.sum(delta_V, axis=-1)
        
        result = x + delta
        return result / jnp.maximum(safe_norm(result, keepdims=True), eps)

