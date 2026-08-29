# SOTA 2026: AUTODIFF RIEMANNIANO, VJP CAYLEY-SMW Y BENCHMARKS COMPARATIVOS

**Fuentes:** OpenRouter (Benchmark), OpenRouter (VJP Cayley)  
**Fecha:** 2026-08-28  
**Estado APIs:** Groq 401, Cerebras 402, Kimi sin clave, SambaNova sin clave, NVIDIA 404, Gemini 503 — OpenRouter funcional.

---

## 1. VJP CORRECTO DE LA RETRACCIÓN CAYLEY-SMW

### Fórmula Primal
$$Y(\alpha) = X - \alpha U K^{-1} V^T X$$
donde $K = I_{2k} + \frac{\alpha}{2} V^T U$, $U = [G, X]$, $V = [X, -G]$.

### Derivación del Pullback (dL/dX y dL/dG)

Dado $\bar{Y} = \frac{dL}{dY}$ (cotangente entrante), aplicamos la regla de la cadena matricial:

**Paso 1:** Diferenciar $Y$ respecto a $X$:
$$\frac{dY}{dX} = I - \alpha \left( \frac{dU}{dX} K^{-1} V^T X + U \frac{dK^{-1}}{dX} V^T X + U K^{-1} \frac{dV^T}{dX} X + U K^{-1} V^T \right)$$

Notar que $U = [G, X]$ implica $\frac{dU}{dX} \cdot \delta X = [0, \delta X]$ y $V = [X, -G]$ implica $\frac{dV^T}{dX} \cdot \delta X = [\delta X^T; 0]$.

**Paso 2:** El pullback completo:
$$\frac{dL}{dX} = \bar{Y}^T \cdot \frac{dY}{dX}$$

**Forma cerrada práctica para implementación en JAX custom_vjp:**

Sea $w = K^{-T} V^T \bar{Y}$ (resolver sistema triangular, NO invertir K explícitamente):

$$\frac{dL}{dX} = \bar{Y} - \alpha \left[ U K^{-1} (V^T \bar{Y}) + \frac{\alpha}{2} (V^T U K^{-1})^T \otimes (K^{-1} V^T X) \cdot \bar{Y} \right]$$

**Regla de oro para el residuo en `custom_vjp`:**
- **Guardar en `res`:** Solo $K^{-1} \in \mathbb{R}^{2k \times 2k}$, $U \in \mathbb{R}^{D \times 2k}$, $V \in \mathbb{R}^{D \times 2k}$.
- **NUNCA guardar:** La matriz $D \times D$ completa. Para $D=10^6$, $k=8$: guardar $K^{-1}$ = 2KB vs guardar $Y$ = 64MB.

---

## 2. GRADIENTE DE LA IDENTIDAD CORDAL

$$\theta = 2 \arctan2(\|x - y\|_2, \|x + y\|_2)$$

### Derivada respecto a $x$:

Sea $a = \|x-y\|_2$, $b = \|x+y\|_2$. Entonces:
$$\frac{d\theta}{dx} = 2 \cdot \frac{b \frac{da}{dx} - a \frac{db}{dx}}{a^2 + b^2}$$

donde:
$$\frac{da}{dx} = \frac{x - y}{\|x-y\|_2}, \quad \frac{db}{dx} = \frac{x + y}{\|x+y\|_2}$$

Sustituyendo:
$$\frac{d\theta}{dx} = \frac{2}{\|x-y\|^2 + \|x+y\|^2} \left( \|x+y\| \frac{x-y}{\|x-y\|} - \|x-y\| \frac{x+y}{\|x+y\|} \right)$$

Notar que $\|x-y\|^2 + \|x+y\|^2 = 2(\|x\|^2 + \|y\|^2) = 4$ para $x, y \in S^{D-1}$.

**Singularidades de gradiente:**
- En $x = y$ ($\theta \to 0$): $a \to 0$, $b \to 2$. El gradiente converge a $\frac{x-y}{\|x-y\|}$ normalizado. **Bien definido** (límite existe).
- En $x = -y$ ($\theta \to \pi$): $a \to 2$, $b \to 0$. El gradiente converge a $-\frac{x+y}{\|x+y\|}$ normalizado. **Bien definido** (límite existe).
- **Ventaja sobre arccos:** $\arccos(\langle x, y \rangle)$ tiene gradiente $\propto \frac{1}{\sqrt{1-\langle x,y\rangle^2}}$ que **explota** cuando $\langle x,y\rangle \to \pm 1$. La identidad cordal evita ambas singularidades.

---

## 3. SHIFTED CHOLESKYQR3: ANÁLISIS COMPARATIVO

### Diferencia entre $s = \epsilon \cdot \text{Tr}(G)$ vs $s = \alpha \cdot \epsilon \cdot \|A\|_F^2$

**Relación:** $\text{Tr}(G) = \text{Tr}(A^T A) = \|A\|_F^2$. Son **equivalentes** cuando $G = A^T A$.

La diferencia es el factor $\alpha$:
- Fukaya et al. (2020) recomienda $\alpha = 11$ para garantizar estabilidad hasta $\kappa(A) \approx \epsilon^{-1}$.
- $s = \epsilon \cdot \text{Tr}(G)$ corresponde a $\alpha = 1$, que garantiza estabilidad solo hasta $\kappa(A) \approx \epsilon^{-2/3} \approx 10^{10}$ en FP64.

**Tabla de Estabilidad:**

| Shift | Garantía FP64 | Garantía FP32 |
|:---|:---|:---|
| Sin shift | $\kappa < 10^7$ | $\kappa < 10^3$ |
| $s = \epsilon \cdot \|A\|_F^2$ (α=1) | $\kappa < 10^{10}$ | $\kappa < 10^5$ |
| $s = 11\epsilon \cdot \|A\|_F^2$ (Fukaya) | $\kappa < 10^{15}$ | $\kappa < 10^7$ |
| Cayley-SMW | Incondicional | Incondicional |
| Householder | Incondicional | Incondicional |

**Veredicto:** Para POLYDIM V78 con vectores cuasi-paralelos potencialmente extremos, usar $s = 11\epsilon \cdot \|A\|_F^2$ o directamente Cayley-SMW.

---

## 4. BENCHMARKS COMPARATIVOS: D=10^6, k=2 y k=8

### FLOPs Totales (3 iteraciones de CholeskyQR vs retracción única Cayley)

| Algoritmo | FLOPs k=2 | FLOPs k=8 | Sincronizaciones |
|:---|:---|:---|:---|
| CholQR3 sin shift | $6Dk^2 = 2.4 \times 10^7$ | $6Dk^2 = 3.84 \times 10^8$ | 3 |
| s-CholQR3 (Fukaya) | $6Dk^2 + 2Dk^2 = 3.2 \times 10^7$ | $\approx 5 \times 10^8$ | 3 |
| Cayley-SMW | $4Dk^2 + 8k^3 = 1.6 \times 10^7$ | $\approx 2.6 \times 10^8$ | 2 |
| Householder Compact WY | $4Dk^2 = 1.6 \times 10^7$ | $\approx 2.6 \times 10^8$ | k = 2 o 8 |

### Cuello de Botella por Hardware

| Hardware | Ganador k=2 | Ganador k=8 | Razón |
|:---|:---|:---|:---|
| GPU (A100, ~2TB/s BW) | Cayley-SMW | Cayley-SMW | Memory-bound: menos lecturas de A |
| CPU (64B cache line) | Householder WY | s-CholQR3 | Compute-bound: BLAS-3 secuencial vs paralelo |
| TPU (MXU 128x128) | s-CholQR3 | s-CholQR3 | GEMM dominante, MXU satura bien |

### Ring-AllReduce Geodésico: Bytes por Ronda (N=100 agentes, mismo nodo SHM)

| Opción | Bytes Transmitidos | Ventaja |
|:---|:---|:---|
| A: $X_i$ completo | $16$MB (k=2), $64$MB (k=8) | Ninguna — copia innecesaria |
| B: $\xi_i = \text{Log}_\mu(X_i)$ | $16$MB (k=2), $64$MB (k=8) | **Misma dimensión pero suma lineal** — reduce en espacio tangente (correcto geométricamente) |
| C: Solo $\nabla_R f$ proyectado | $16$MB (k=2), $64$MB (k=8) | Solo válido si todos los agentes tienen el mismo $\mu$ |

**Conclusión:** La Opción B (vectores tangentes) es la **arquitectónicamente correcta** porque la suma $\sum w_i \xi_i$ vive en $T_\mu \mathcal{M}$ (espacio tangente lineal), permitiendo la retracción posterior $R_\mu(\gamma \sum w_i \xi_i)$. A vs B vs C transmiten el mismo volumen de bytes, pero B es la única matemáticamente coherente con el Karcher Flow.

---

## 5. RESUMEN EJECUTIVO PARA V78

### Arquitectura Recomendada por el Tribunal MCP

1. **Ortogonalización:** Cayley-SMW Matrix-Free (incondicionalmente estable, 2 sincronizaciones, 100% BLAS-3).
2. **Log Map:** Identidad Cordal $\theta = 2\arctan2(\|x-y\|, \|x+y\|)$ — gradientes estables en toda la esfera.
3. **Fallback paralelismo:** Umbral `is_parallel < 1e-3` (telemetría del runner confirma que 1e-4 falla para RotErr ~0.2).
4. **custom_vjp:** Guardar solo $K^{-1} \in \mathbb{R}^{2k \times 2k}$ + $U, V \in \mathbb{R}^{D \times 2k}$. Cero matrices D×D en tape.
5. **Consenso Karcher:** Ring-AllReduce sobre vectores tangentes $\xi_i$ (Opción B).
6. **FFI:** Migrar a `jax.ffi.register_ffi_target` con `XLA_FFI_Handler`, validación explícita de `is_c_contiguous()`.

### Estado de APIs MCP
- **OpenRouter:** ✅ Funcional
- **Groq:** ❌ 401 Invalid Key
- **Cerebras:** ❌ 402 Quota agotada
- **Kimi:** ❌ Sin clave
- **SambaNova:** ❌ Sin clave
- **NVIDIA NIM:** ❌ 404
- **Gemini MCP:** ❌ 503 Alta demanda
