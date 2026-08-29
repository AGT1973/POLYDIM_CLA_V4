"""
POLYDIM V57 SILICON & RUNTIME CONFIGURATION
- Configuración de runtime previa a la trazada de grafos XLA
- Inspección pasiva de hardware con advertencia de sharding multi-chip
- Warm-Up Multi-Dispositivo (CPU / TPU Multi-Core con pmap)
"""

import os
import jax
import jax.numpy as jnp

def configure_runtime(enable_x64: bool = True):
    """Configura el runtime de JAX al inicio del proceso de forma explícita."""
    jax.config.update("jax_enable_x64", enable_x64)

class SiliconContract:
    @staticmethod
    def inspect():
        """Inspección pasiva del silicio sin mutación de estado global."""
        devices = jax.devices()
        platform = devices[0].platform.upper()
        cpu_count = os.cpu_count() or 1
        float_info = jnp.finfo(jnp.float32)
        dev_count = len(devices)
        
        sharding_note = "Single-chip execution por defecto" if dev_count == 1 else f"⚠ device_count={dev_count} detectado; single-chip salvo sharding explícito con NamedSharding/pmap"

        return {
            "platform": platform,
            "device_count": dev_count,
            "devices": devices,
            "cpu_cores": cpu_count,
            "eps": float(float_info.eps),
            "tiny": float(float_info.tiny),
            "cache_line_bytes": 64,
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "sharding_note": sharding_note
        }

    @staticmethod
    def warmup(geometry_class, dim: int = 1000):
        """Warm-Up JIT multi-dispositivo con split de claves y trazado completo."""
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key, 2)
        
        q1 = jax.random.normal(k1, (dim,), dtype=jnp.float32)
        q1 = q1 / jnp.linalg.norm(q1)
        q2 = jax.random.normal(k2, (dim,), dtype=jnp.float32)
        q2 = q2 / jnp.linalg.norm(q2)

        num_devices = jax.device_count()
        if num_devices > 1:
            q1_multi = jnp.broadcast_to(q1, (num_devices, dim))
            q2_multi = jnp.broadcast_to(q2, (num_devices, dim))
            pmap_slerp = jax.pmap(geometry_class.slerp, in_axes=(0, 0, None))
            _ = pmap_slerp(q1_multi, q2_multi, 0.5)
            jax.block_until_ready(_)
        else:
            _ = geometry_class.slerp(q1, q2, 0.5)
            jax.block_until_ready(_)

        _ = geometry_class.exp_map(q1, jnp.zeros_like(q1))
        jax.block_until_ready(_)
        _ = geometry_class.log_map(q1, q1)
        jax.block_until_ready(_)
        return True
