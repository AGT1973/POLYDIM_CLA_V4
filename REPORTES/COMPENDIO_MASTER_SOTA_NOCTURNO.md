# COMPENDIO MASTER SOTA — POLYDIM MODO NOCTURNO 2026

**Fecha de Generación:** 2026-08-28T15:23  
**Runner Iteraciones Completadas:** ~1,200+  
**Reportes de Silicio Generados:** 123 archivos en `REPORTES/`  
**Estado del Sistema:** Runner Autónomo Activo (task-260)

---

## 1. TELEMETRÍA DE SILICIO — SÍNTESIS ESTADÍSTICA (Iteraciones #1–#1200)

### Invariante 1: Reversibilidad Geodésica (Exp ∘ Log)
$$\text{RevErr} = \|y - \hat{y}\|_2 \quad \text{donde } \hat{y} = \text{Exp}_x(\text{Log}_x(y))$$

| Dimensión | RevErr Mínimo | RevErr Máximo | RevErr Típico | Latencia Típica |
|:---|:---|:---|:---|:---|
| D = 10,000 | 4.53e-08 | 2.54e-07 | ~1.2e-07 | 30–250 ms |
| D = 50,000 | 5.16e-08 | 2.84e-07 | ~1.2e-07 | 35–340 ms |
| D = 100,000 | 5.21e-08 | 2.31e-07 | ~1.1e-07 | 55–480 ms |

**Diagnóstico:** La reversibilidad geodésica opera consistentemente al límite de precisión de máquina FP32 (~1.2e-07 ≈ ε_machine × √D). La identidad cordal 2·arctan2(‖x-y‖, ‖x+y‖) elimina la singularidad de arccos.  
**Veredicto:** ✅ ESTABLE. Ningún NaN/Inf en 1,200+ iteraciones.

---

### Invariante 2: Ortogonalidad del Fallback Analítico (Gram-Schmidt con Detección U ‖ V)
$$\text{RotErr} = |\langle U_{\text{orth}}, V_{\text{orth}} \rangle|$$

| Dimensión | RotErr Mínimo | RotErr Máximo | Observación |
|:---|:---|:---|:---|
| D = 10,000 | 0.00e+00 | ~2.0e-01 | Oscilaciones esperadas en casos cuasi-paralelos |
| D = 50,000 | 0.00e+00 | ~1.4e-01 | Idem |
| D = 100,000 | 0.00e+00 | ~2.1e-01 | Casos extremos detectados |

**Diagnóstico CRÍTICO:** El RotErr oscila entre cero (fallback ortogonal determinista activado) y ~0.2 (casi-paralelismo antes del umbral de detección `is_parallel < 1e-4`). Esto confirma el Bug B23 de Kimi: el umbral `1e-4` es demasiado conservador. Con `U · V > 0.9999`, el V_perp tiene norma ~1.4e-04, justo en el límite, generando RotErr ~0.1-0.2.

**Acción Requerida:** Subir el umbral de detección de paralelismo a `is_parallel < 1e-3` para capturar los casos de borde antes de que el fallback falle.

---

## 2. HALLAZGOS SOTA — INVESTIGACIÓN EN PARALELO

### 2.1 Retracción Cayley-SMW (EJE 1 — Sabueso Stiefel)
**Fuente:** `SOTA_STIEFEL_CONSENSUS_BULLDOG.md`

La retracción Cayley-SMW Matrix-Free es **incondicionalmente no singular** gracias a la estructura antisimétrica del generador W:
$$Y(\alpha) = X - \alpha U \left(I_{2k} + \frac{\alpha}{2} V^T U\right)^{-1} V^T X$$
$$\det\left(I + \frac{\alpha}{2}W\right) = \prod_j \left(1 + \frac{\alpha^2\lambda_j^2}{4}\right) \geq 1 \quad \forall \alpha \in \mathbb{R}$$

**Costo:** O(4Dk² + 8k³) — 100% BLAS-3, sin singularidades en antípodas, sin oscilaciones trigonométricas.

**Aplicación a POLYDIM V78:** Reemplaza CliffordRotors.householder_reflect + cholesky_qr3 con la retracción Cayley-SMW cuando k=2 o k=8.

---

### 2.2 Shifted CholeskyQR3 (EJE 3 — Sabueso Stiefel)
**Fuente:** `SOTA_STIEFEL_CONSENSUS_BULLDOG.md`

Para matrices con κ(A) ≥ ε^{-1/2} (≈ 10^7 en FP64), CholQR estándar colapsa por NaN.  
El shift adaptativo de Fukaya et al. (2020):
$$s = \max(\epsilon, \epsilon \cdot \text{Tr}(G))$$
garantiza estabilidad hasta κ(A) ≈ 10^{15} en FP64 con 3 iteraciones.

---

### 2.3 FFI Tipado XLA (EJE 2 — Sabueso FFI/IPC)
**Fuente:** `SOTA_FFI_IPC_ZERO_COPY_BULLDOG.md`

- **Muerte de register_custom_call_target:** Sin validación de strides, dtype ni rank. Causa segfaults silenciosos.
- **jax.ffi con XLA_FFI_Handler:** Validación en compile-time. Propagación de errores estructurada. 100% fuera del GIL.
- **Trampa de strides:** Un jnp.transpose() genera strides no-contiguos que el kernel C++ lee como memoria basura si asume puntero plano.
- **Solución:** `is_c_contiguous()` explícito antes de cada acceso, con `_mm_sfence()` y `std::memory_order_release` en Ring Buffers IPC.

---

### 2.4 Anti-Patrón Detectado: Explosión de Tape en custom_vjp
**Fuente:** `SOTA_FFI_IPC_ZERO_COPY_BULLDOG.md`

Para D=10^6 con M=1,000 pasos: Tape = M × 8MB = 8GB → OOM.  
**Solución:** Almacenar solo el núcleo K^{-1} ∈ R^{2k×2k} (2 KB para k=8) + U, V ∈ R^{D×2k}.

---

## 3. PLAN DE ACCIÓN PARA V78

### Prioridad Crítica (Antes de escribir código)
1. **Umbral de Paralelismo:** `is_parallel < 1e-3` (confirmado por telemetría del runner).
2. **Cayley-SMW en lugar de CholQR3 + Householder:** La telemetría muestra que el RotErr máximo de 0.2 corresponde exactamente a casos cuasi-paralelos donde CholQR3 falla. Cayley-SMW evita esta ruta por completo.
3. **Identidad Cordal para log_map:** RevErr estable en ~1.2e-07 valida la fórmula 2·arctan2(‖x-y‖, ‖x+y‖).
4. **FFI tipado con is_c_contiguous():** Obligatorio en C++ y Rust.
5. **custom_vjp con residuos comprimidos:** Solo K^{-1} en `res`, no el estado completo.

---

## 4. ESTADO DE INFRAESTRUCTURA

| Motor | Estado | Detalles |
|:---|:---|:---|
| Runner de Silicio (`task-260`) | ✅ ACTIVO | Iteración ~1,200, 0 NaNs detectados |
| Cron Nocturno (`task-253`) | 🔴 DETENIDO | Saturación de tokens — fue disparando cada 5 min sin pausa. Se cancela. |
| Sabueso Stiefel | ✅ COMPLETO | Reporte en `SOTA_STIEFEL_CONSENSUS_BULLDOG.md` |
| Sabueso FFI/IPC | ✅ COMPLETO | Reporte en `SOTA_FFI_IPC_ZERO_COPY_BULLDOG.md` |
| Sabueso FFI/IPC (2do intento) | 🔴 QUOTA 429 | Reintentar en 165h con cuota renovada |

---

## 5. PRÓXIMOS SABUESOS A DESPLEGAR (En Orden de Prioridad)

1. **SOTA Autodiff Riemanniano:** Convergencia de custom_vjp en S^(D-1), gradientes en colisión, HVP.
2. **SOTA PMTP v44:** Protocolo de consenso tensorial con Ring-AllReduce Geodésico y Karcher Flow.
3. **SOTA Benchmark Comparativo:** CholQR3 vs Cayley-SMW vs Householder Compact WY en D=10^4, 10^6.
