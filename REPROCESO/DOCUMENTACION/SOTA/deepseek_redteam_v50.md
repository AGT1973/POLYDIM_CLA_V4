Evaluación Bulldog Critic del benchmark SLERP JAX/TPU:

## Análisis de Exploits Reportados

### 1. **Float64 emulado en TPU destruye throughput** - **CONFIRMADO PARCIALMENTE**
- **Verdad**: TPU v3 tiene unidades de 32-bit nativas. Float64 se emula en software, típicamente 10-100x más lento.
- **Matiz**: Si el script usa `jnp.float64` explícitamente, es un suicidio de rendimiento. Pero si usa `jnp.float32` (default), el exploit es irrelevante.
- **Ataque real**: El vector de ataque es forzar `dtype=float64` en las operaciones de normalización, no en la entrada.

### 2. **Sin jax.sharding = OOM en D=10^9** - **CONFIRMADO**
- **Verdad**: Con 8 núcleos TPU y memoria HBM compartida (~32GB), un vector de 10^9 floats32 = 4GB. Sin sharding explícito, JAX puede replicar en cada núcleo, causando OOM.
- **Ataque adicional**: No solo OOM, sino que sin `jax.experimental.mesh_utils` y `jax.sharding.NamedSharding`, el compilador puede hacer fallos de memoria intermedia (buffer fragmentation).

### 3. **jnp.sum sin Kahan pierde precisión** - **CONFIRMADO PERO IRRELEVANTE**
- **Verdad**: La suma ingenua de 10^9 elementos pierde ~1e-7 relativo.
- **Pero**: Para SLERP, la normalización requiere `norm = sqrt(sum(x^2))`. La pérdida de precisión en la norma es despreciable para la dirección del vector (error < 1e-7 en ángulo).
- **Ataque real**: El exploit crítico es en la **resta** `v - (v·u)u` donde la cancelación catastrófica puede destruir la ortogonalidad. Kahan en el producto punto es más crítico que en la suma.

### 4. **PRNG mono-bloque satura allocator** - **CONFIRMADO**
- **Verdad**: `jax.random.normal` con un solo key para 10^9 elementos genera un bloque contiguo de 4GB. En TPU con memoria fragmentada, esto puede fallar.
- **Ataque adicional**: Usar `jax.random.split` en 8 particiones y generar en chunks de 128MB es la mitigación estándar. Si el script no lo hace, es un exploit legítimo.

### 5. **XLA duplica memoria en JIT** - **CONFIRMADO PARCIALMENTE**
- **Verdad**: XLA puede crear buffers temporales para fusiones de kernels. Para SLERP, la secuencia `u·v`, `u - (u·v)v`, `normalize` puede requerir 3-4 buffers temporales.
- **Matiz**: Con `jax.jit` y `static_argnums` bien definidos, XLA puede fusionar operaciones. El exploit real es si el script usa `jax.grad` o `jax.vmap` innecesariamente, forzando materialización de gradientes.

## Vectores de Ataque Faltantes (Críticos)

### 6. **Falta de `jax.checkpoint` en la normalización** - **EXPLOIT GRAVE**
- Si el script calcula `norm = jnp.sqrt(jnp.sum(x**2))` y luego `x / norm`, XLA puede recomputar `x**2` en el backward pass si se usa autodiff. Sin checkpointing, la memoria se duplica.

### 7. **Uso de `jnp.dot` en lugar de `jnp.einsum`** - **EXPLOIT DE RENDIMIENTO**
- `jnp.dot` para vectores 1D puede no aprovechar las unidades de matriz TPU. `einsum('i,i->', u, v)` es más eficiente en TPU.

### 8. **Falta de `jax.lax.associative_scan` para sumas parciales** - **EXPLOIT DE PRECISIÓN**
- Para D=10^9, usar `jnp.sum` con reducción en árbol (tree reduction) es mejor que la suma secuencial. Si el script usa `jnp.sum` sin especificar `axis`, JAX puede elegir una estrategia subóptima.

### 9. **No usar `jax.device_put` con `sharding` explícito para el vector de entrada** - **EXPLOIT DE MEMORIA**
- Si el vector se crea en host y se transfiere, hay doble copia. Debe crearse directamente en el dispositivo con sharding.

### 10. **Falta de `jax.profiler` para verificar fusiones** - **EXPLOIT DE DIAGNÓSTICO**
- Sin profiling, no se puede confirmar si XLA está fusionando correctamente. El benchmark debería incluir `jax.profiler.start_trace` para validar.

## Veredicto Brutal

**El benchmark es mediocre**. Los 5 exploits reportados son en su mayoría "verdaderos pero triviales" - cualquier ingeniero JAX competente los evitaría. Los exploits reales (6-10) son más sutiles y no fueron detectados.

**Puntuación**: 4/10
- **Rigor técnico**: 5/10 (los exploits son correctos pero superficiales)
- **Profundidad**: 3/10 (no cubre los problemas de autodiff, einsum, o sharding fino)
- **Utilidad**: 4/10 (los exploits 2 y 4 son útiles, el resto es ruido)

**Recomendación**: Rehacer el benchmark con:
1. `jax.jit` con `donate_argnums` para reutilizar buffers
2. `jax.sharding` con `P('batch')` para distribución
3. `jax.lax.associative_scan` para reducción estable
4. `jax.checkpoint` para control de memoria
5. Profiling con `jax.profiler` para validar fusiones XLA

El script actual es un "hello world" de JAX, no un benchmark de TPU.