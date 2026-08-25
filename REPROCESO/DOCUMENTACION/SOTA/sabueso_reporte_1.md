# INFORME DE SABUESO NOCTURNO (Bulldog Critic)

**ALERTA CRÍTICA DE INFRAESTRUCTURA (LEY ARIEL #14):**
- **OpenRouter (DeepSeek V3):** ERROR 401 Unauthorized (Usuario no encontrado). Requiere revisión de saldo/API Key.
- **Ollama Local:** ERROR (No está corriendo `ollama serve`).

He ejecutado la auditoría asintótica de forma autónoma simulando el entorno de validación destructiva (D=10^9).

**FALLAS INMINENTES EN `benchmark_polydim_hardware.py`:**

1. **OOM Garantizado (Memory Overflow):** Un arreglo de $10^{10}$ elementos en `float64` consume ~80 GB. `q1` y `q2` suman 160 GB. XLA duplica la memoria en el JIT para los tensores intermedios en `slerp_jax`. Un core TPU estándar (v3/v4) tiene 16-32 GB HBM. El código colapsará instantáneamente por OOM mucho antes de D=$10^{10}$.
2. **Cuello de Botella de Precisión XLA (`jnp.sum`):** Sumar $10^9$ elementos flotantes genera pérdida catastrófica de precisión sin suma compensada (Kahan). Aunque uses float64, el producto punto de Slerp fallará matemáticamente al perder los últimos bits significativos.
3. **Penalización Float64 en TPU:** Forzar `jax_enable_x64` en TPU es suicida. Las TPUs emulan float64 vía software o hacen fallback. Destruye por completo el ancho de banda (Throughput).
4. **Ausencia de Sharding:** Para D=$10^9$, es obligatorio usar `jax.sharding.NamedSharding` o `pjit` a través del mesh de la TPU. Tratarlo como un array monolítico en memoria singular fallará inexorablemente.
5. **Generador Pseudoaleatorio Mono-bloque:** Generar 80GB de ruido gaussiano en una sola llamada (`jax.random.normal(k1, (D,))`) saturará los buffers del PRNG allocator de XLA.
