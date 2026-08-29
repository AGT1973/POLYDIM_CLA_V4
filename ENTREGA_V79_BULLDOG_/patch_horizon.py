import re

py_file = 'E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito_fixed.py'
with open(py_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. log_map_newton: while_loop -> fori_loop
content = re.sub(
    r'def cond_fun.*?lax\.while_loop\(cond_fun, body_fun, \(0, v_init, err_init\)\)',
    r'''def body_fun(i, state):\n            v, err = state\n            y_approx = GeodesicKernels.exp_map(xu, v)\n            res = GeodesicKernels.log_map(y_approx, yu)\n            has_nan = jnp.any(jnp.isnan(res))\n            c = GeodesicKernels.safe_dot(y_approx, xu, keepdims=True)\n            safe_denom = jnp.where(jnp.abs(1.0 + c) < eps, 1.0, 1.0 + c)\n            trans_res = res - (GeodesicKernels.safe_dot(res, y_approx + xu, keepdims=True) / safe_denom) * (y_approx + xu)\n            v_new = v + trans_res\n            v_new = v_new - GeodesicKernels.safe_dot(v_new, xu, keepdims=True) * xu\n            err = jnp.max(jnp.where(jnp.isnan(trans_res), 0.0, GeodesicKernels.safe_norm(trans_res, keepdims=False, eps=eps)))\n            err = jnp.where(has_nan, 0.0, err)\n            v_new = jnp.where(has_nan, jnp.nan, v_new)\n            return v_new, err\n\n        v_final, _ = lax.fori_loop(0, max_iter, body_fun, (v_init, 1.0))''',
    content, flags=re.DOTALL
)

# 2. log_map: remove clamp
content = re.sub(
    r'safe_sum_norm = jnp\.where\(sum_norm < eps, 1\.0, sum_norm\)\n\s*theta = 2\.0 \* jnp\.arctan2\(diff_norm, safe_sum_norm\)',
    r'theta = 2.0 * jnp.arctan2(diff_norm, sum_norm)',
    content
)

# 3. Cholesky shift
content = re.sub(
    r'shift = jnp\.maximum\(eps, 11\.0 \* eps \* trace_G / K\) \* I',
    r'fro_norm = jnp.linalg.norm(G, axis=(-2, -1), keepdims=True)[..., None, None]\n        shift = jnp.maximum(1e-8, 1e-6 * fro_norm) * I',
    content
)

with open(py_file, 'w', encoding='utf-8') as f:
    f.write(content)
print('Python monolith patched.')
