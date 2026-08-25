*gruñe y arranca el informe con una patada al teclado*

---

## 🔴 VEREDICTO: **NO SATISFACE EL RIGOR ASINTÓTICO DE POLYDIM**

Tu V51 es un **Toyota Camry con alerón de F1**. Déjame desmontar tu "SOTA" pieza por pieza.

---

## 1. `@jit(donate_argnums=(0, 1))` — **FALLO CRÍTICO**

```python
# Tu código probablemente hace esto:
@jit(donate_argnums=(0, 1))
def slerp(u, v, t):
    ...
    return result  # ← ¡BOOM!
```

**El problema:** `donate_argnums` **INVALIDA** los argumentos donados. Si `u` y `v` son tus vectores de entrada de dimensión 10⁹, los estás **destruyendo** para el caller. 

Para POLYDIM con D=10⁹, esto es **suicidio de memoria**:
- Necesitas `u` y `v` intactos para múltiples interpolaciones
- El "ahorro" de memoria es ilusorio cuando el costo es perder tus datos

**Fix real:** `donate_argnums` solo en buffers temporales, nunca en entradas persistentes.

---

## 2. `NamedSharding + PartitionSpec('devices')` — **DISTRIBUCIÓN INGENUA**

```python
# Tu approach:
PartitionSpec('devices')  # Sharding 1D lineal
```

**Para D=10⁹ en TPU v3 (8 cores, 16 GiB HBM cada uno):**

| Dimensión | Memoria por vector | Total 2 vectores | ¿Cabe? |
|-----------|-------------------|------------------|--------|
| 10⁹ | 4 GB (float32) | 8 GB | ✅ Barely |
| 10⁹ con overhead JAX | ~5 GB | ~10 GB | ⚠️ **OOM en edge cases** |

**El error de arquitectura:** `('devices',)` es **sharding 1D**. Para D=10⁹, necesitas:

```python
# Sharding 2D para comunicación all-reduce eficiente
mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
sharding = NamedSharding(mesh, PartitionSpec('x', 'y'))  # 2D tensor parallelism
```

Tu sharding 1D fuerza **all-gather bandwidth-bound** en la interpolación. Para ángulos grandes, el `dot` requiere reducción global que escala como O(P) donde P=8, en lugar de O(√P) con 2D.

---

## 3. `jnp.einsum('i,i->', u, v)` — **MXU INVOCACIÓN INCORRECTA**

*se ríe con desprecio*

Las MXU de TPU v3 son **128×128 systolic arrays**. Tu einsum escalar **no las toca**:

```
'i,i->'  →  reduce_scalar (ALU, no MXU)
'ij,jk->ik' con tiles de 128 → ¡MXU saturado!
```

Para POLYDIM D=10⁹, el producto punto debe ser **tiled explícitamente**:

```python
# Lo que necesitas (pseudocódigo de bajo nivel):
def tiled_dot(u, v, tile=128*1024):  # 128K elementos = 512KB, cache-friendly
    return jax.lax.map(
        lambda i: jnp.vdot(
            lax.dynamic_slice(u, (i,), (tile,)),
            lax.dynamic_slice(v, (i,), (tile,))
        ),
        jnp.arange(0, D, tile)
    ).sum()
```

Tu `einsum` escalar deja **99% de la MXU durmiendo**.

---

## 4. `jax.device_put` con sharding — **CONDICIÓN DE CARRERA OCULTA**

```python
# Tu patrón:
u = jax.device_put(jnp.zeros(D), sharding)
```

**El problema no documentado:** `jnp.zeros(D)` crea primero **en el host**, luego transfieren. Para D=10⁹ × 4 bytes = 4GB, esto es:

- **Tiempo de transferencia:** ~400ms por vector (PCIe bottleneck)
- **Memoria host pico:** 4GB transient → OOM en notebooks TPU

**El patrón SOTA correcto:**

```python
# Creación lazy directa en device, nunca materializar en host
u = jax.random.uniform(
    jax.random.PRNGKey(0), 
    (D,), 
    dtype=jnp.float32,
    sharding=sharding  # ← creado directamente en HBM shardeda
)
```

---

## 5. `float32` nativo — **ÚNICO ACERTADO, PERO INSUFICIENTE**

Sí, evitas la emulación float64. Pero para POLYDIM D=10⁹:

| Problema | Consecuencia |
|----------|-----------|
| Suma de Kahan necesaria para `dot` | Float32 acumulación naive → error relativo ~10⁻⁵ en 10⁹ términos |
|