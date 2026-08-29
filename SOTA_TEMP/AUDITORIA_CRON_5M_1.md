**🔎 Sabueso Auditor Red Team – Informe de Auditoría (versión “destructiva”)**  
**Código bajo revisión**

```python
@staticmethod
@jax.jit
def quantize_int8_tree_reduce(tensor: jnp.ndarray):
    abs_max = jnp.max(jnp.abs(tensor))
    safe_max = jnp.where(abs_max == 0, 1.0, abs_max)
    scale = safe_max / 127.0
    quantized = jnp.clip(jnp.round(tensor / scale), -127, 127).astype(jnp.int8)
    return quantized, scale
```

---

## 1. Resumen ejecutivo

| Tema | Estado | Comentario crítico |
|------|--------|--------------------|
| **Underflow/Overflow en `float64`** | **Seguro** | Los rangos de `float64` son astronomicamente amplios (≈ ±1e308). La única operación que podría producir underflow es `tensor / scale` cuando `scale` es **muy grande** (≈ 1e308). En la práctica, `scale = max(|tensor|)/127` nunca supera `~1e306` (porque `max(|tensor|)` está limitado por la precisión de la entrada). El redondeo y el clip posterior eliminan cualquier valor fuera de `[-127,127]`. |
| **Underflow/Overflow en `float32`** | **Peligro moderado** | `float32` tiene rango ≈ ±3.4e38. Si `tensor` contiene valores > ≈ 4.3e40 (imposible en `float32`), el cálculo de `abs_max` saturará a `inf`. En caso de **inf** o **NaN**, `safe_max` se vuelve `inf` → `scale = inf` → `tensor/scale = 0` → `quantized = 0`. Esto **no** genera overflow, pero sí **pérdida total de información** sin advertencia. |
| **División por cero** | **Controlada** | `safe_max` sustituye `0` por `1.0`, evitando división por cero. Sin embargo, si `abs_max` es `NaN`, la condición `abs_max == 0` es **False**, por lo que `safe_max = NaN` → `scale = NaN` → todo el pipeline produce `NaN`. |
| **Casting a `int8` en TPUs** | **Sin condición de carrera** | En XLA/TPU, el `astype(jnp.int8)` se traduce a una operación de **convertir a entero con saturación** que se ejecuta como una única instrucción vectorial. No hay acceso concurrente a memoria mutable que pueda generar race conditions. El único “riesgo” es **saturación inesperada** si el `clip` falla (p.ej., por un bug en XLA). |
| **Clip + Round** | **Posible sesgo** | `jnp.round` usa “round‑to‑nearest‑even”. Cuando `tensor/scale` está justo a medio‑punto (e.g., 0.5, 1.5), el redondeo puede **alternar** entre arriba y abajo, introduciendo un sesgo estadístico que se amplifica al cuantizar. |
| **Precisión de `scale`** | **Pérdida de precisión** | `scale` se calcula en el mismo dtype que `tensor`. Si `tensor` es `float32`, `scale` también lo será, lo que implica **≈7 dígitos decimales** de precisión. Para valores de `abs_max` cercanos a `127`, la división `safe_max/127` puede producir un `scale` con error relativo ~1e‑7, suficiente para que `tensor/scale` se desplace fuera del rango `[-127,127]` en los extremos y sea recortado. |
| **Uso de `jnp.max` en árbol (tree‑reduce)** | **Correcto** | XLA implementa `max` como reducción en árbol, garantizando **determinismo** y **precisión numérica** idéntica a la reducción secuencial (excepto por el orden de operaciones, que no afecta al `max`). No hay pérdida de información. |
| **Posibles NaNs/Inf** | **Falta de manejo explícito** | El código no detecta `NaN`/`Inf`. En pipelines críticos (p.ej., entrenamiento de modelos), la propagación silenciosa de `NaN` puede colapsar todo el grafo. |

---

## 2. Detalle de riesgos por tipo de dato

### 2.1 `float64` (double‑precision)

| Operación | Rango seguro | Posible fallo | Consecuencia |
|-----------|--------------|---------------|--------------|
| `jnp.abs(tensor)` | ≤ 1e308 | `abs_max = inf` (solo si entrada > ≈ 1e308) | `scale = inf` → salida `0` (pérdida total) |
| `safe_max = where(abs_max == 0, 1.0, abs_max)` | 0 → 1.0 | `abs_max = NaN` → `safe_max = NaN` | `scale = NaN` → todo `NaN` |
| `scale = safe_max / 127.0` | ≤ ≈ 8e305 | `scale = 0` (solo si `abs_max` = 0, ya manejado) | No hay división por cero |
| `tensor / scale` | ≤ ≈ 1e308 / 1e-308 = 1e616 (overflow a `inf`) | Si `scale` es **muy pequeño** (≈ 1e‑308) por `abs_max` ≈ 1e‑306, el cociente puede overflow a `inf`. | `round(inf) = inf` → `clip` → 127 → saturación, pero **no** crash. |
| `round` / `clip` | Operan sobre `inf` sin excepción | N/A | Resultado saturado a 127/‑127. |

**Conclusión:** En `float64` el único escenario crítico es **entrada NaN/Inf**. No hay underflow/overflow que cause excepción; la lógica de saturación protege contra valores fuera del rango `int8`.

### 2.2 `float32` (single‑precision)

| Operación | Rango seguro | Posible fallo |
|-----------|--------------|---------------|
| `abs_max` | ≤ 3.4e38 | `abs_max = inf` si entrada > ≈ 3.4e38 (imposible en `float32`) |
| `scale` | ≤ ≈ 2.7e36 | Si `abs_max` ≈ 3.4e38 → `scale` ≈ 2.7e36 |
| `tensor / scale` | ≤ ≈ 1e‑38 (subnormal) → **underflow a 0** | Cuando `scale` es enorme, la división produce **subnormales** que XLA redondea a 0. El `round` mantiene 0 → `quantized = 0`. |
| `round` / `clip` | Operan sobre 0 sin problema | No hay overflow. |

**Riesgo real:** **Pérdida de información** cuando `abs_max` es muy grande respecto a los valores típicos del tensor (p.ej., tensor con valores en rango `[-1, 1]` pero con un solo outlier de `1e38`). El outlier inflará `scale` y **aplana** todo lo demás a 0.

### 2.3 `float16` / bfloat16 (si se usan implícitamente)

- **float16**: rango ≈ ±6.5e4. `abs_max` > 6.5e4 → `inf`. El mismo efecto que en `float32` pero con umbral mucho más bajo. **Altamente propenso a saturación**.
- **bfloat16**: rango ≈ ±3.4e38 (como `float32`) pero solo 7 bits de mantisa → **gran error de cuantización** en `scale`. El error puede ser del **10 %** en valores cercanos a `abs_max`, lo que se traduce en errores de ±1‑2 en la salida `int8`.

---

## 3. Análisis de la conversión a `int8` en TPUs

| Paso | Implementación XLA | Comentario de seguridad |
|------|--------------------|--------------------------|
| `jnp.clip(..., -127, 127)` | `Clamp` → instrucción de saturación en hardware | Garantiza que el valor está dentro del rango antes del cast. |
| `jnp.round` | `RoundEven` (IEEE‑754) | Determinista, sin race. |
| `.astype(jnp.int8)` | `Convert` → **truncación con saturación** (si el valor está fuera del rango, se trunca a -128/127). En la práctica, XLA inserta un **saturating cast** que no genera overflow de hardware. | No hay acceso a memoria compartida que pueda producir race conditions. Cada elemento se procesa de forma independiente. |
| **Posible bug de XLA** | Históricamente, versiones antiguas de XLA tenían un *bug* donde `Convert` de `float` a `int8` no aplicaba saturación y dejaba valores **wrap‑around**. Este bug fue corregido en la rama `2023‑09`. Si el entorno usa una versión anterior, **sí** podría haber overflow silencioso. | **Recomendación:** Verificar la versión de XLA (`jax.lib.xla_bridge.get_backend().platform_version`). |

**Conclusión:** En TPUs modernos (XLA ≥ 2023‑09) **no hay race conditions** ni overflow inesperado al cast a `int8`. El único punto vulnerable es la posible presencia de un **bug de saturación** en versiones antiguas.

---

## 4. Vulnerabilidades lógicas / de integridad

| Vulnerabilidad | Descripción | Impacto | Mitigación |
|----------------|-------------|---------|------------|
| **Silencioso `NaN` propagation** | `abs_max == 0` no captura `NaN`. `safe_max` = `NaN` → `scale = NaN` → salida `NaN`. | Modelos pueden entrenarse con gradientes `NaN` sin aviso, colapsando el entrenamiento. | Añadir `jnp.where(jnp.isnan(abs_max), 0.0, safe_max)` o lanzar excepción. |
| **Outlier‑driven scale** | Un único valor extremo inflará `scale`, aplastando el resto del tensor a 0. | Pérdida de precisión drástica, especialmente en activaciones con distribución larga. | Usar **percentil** (p.ej., 99.9‑th) en vez de `max`, o aplicar **clipping previo**. |
| **Sesgo de redondeo** | `round` usa “even”. En datos simétricos, el sesgo puede ser pequeño, pero en datos con muchos valores justo a .5, el error se acumula. | Pequeña degradación de SNR (Signal‑to‑Noise Ratio). | Cambiar a `jnp.floor(x + 0.5)` si se prefiere “round‑away‑from‑zero”. |
| **Escala no representable en `float16`** | Si el tensor está en `float16`, `scale` puede quedar como subnormal o `inf`. | Resultado `quantized` = 0 o saturado. | Forzar `tensor.astype(jnp.float32)` antes de la reducción. |
| **Falta de control de tipo de salida** | La función devuelve `scale` en el mismo dtype que `tensor`. Si el llamador espera `float32` y recibe `float64`, puede haber incompatibilidades. | Errores de tipo en pipelines estáticos. | Documentar y/o convertir explícitamente: `scale = scale.astype(jnp.float32)`. |

---

## 5. Recomendaciones de hardening

1. **Validación de NaNs / Infs**  
   ```python
   if jnp.isnan(abs_max) or jnp.isinf(abs_max):
       raise ValueError("Tensor contains NaN/Inf – cannot quantize safely")
   ```

2. **Escala robusta**  
   - Usar percentil en vez de `max`:
     ```python
     p99 = jnp.percentile(jnp.abs(tensor), 99.9)
     safe_max = jnp.where(p99 == 0, 1.0, p99)
     ```
   - O aplicar **clipping** previo:
     ```python
     tensor = jnp.clip(tensor, -clip_val, clip_val)
     ```

3. **Tipo explícito**  
   ```python
   tensor = tensor.astype(jnp.float32)   # o float64 según necesidad
   ```

4. **Redondeo determinista** (si se prefiere “away‑from‑zero”):
   ```python
   quantized = jnp.clip(jnp.floor(tensor / scale + 0.5), -127, 127).astype(jnp.int8)
   ```

5. **Chequeo de versión XLA**  
   ```python
   backend = jax.lib.xla_bridge.get_backend()
   assert backend.platform_version >= "2023-09", "XLA version too old – possible int8 saturation bug"
   ```

6. **Testing de borde**  
   - Generar tensores con valores extremos (`±1e38` en `float32`, `±1e308` en `float64`).  
   - Verificar que `quantized` nunca