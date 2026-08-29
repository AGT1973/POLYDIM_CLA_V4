
class XLAQuantizer:
    @staticmethod
    @jax.jit
    def quantize_int8_tree_reduce(tensor: jnp.ndarray):
        '''[SOTA] Reduccion Jerarquica O(log N) nativa en TPU/GPU'''
        abs_max = jnp.max(jnp.abs(tensor))
        safe_max = jnp.where(abs_max == 0, 1.0, abs_max)
        scale = safe_max / 127.0
        quantized = jnp.clip(jnp.round(tensor / scale), -127, 127).astype(jnp.int8)
        return quantized, scale
