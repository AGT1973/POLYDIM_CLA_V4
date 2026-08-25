# 🚀 README FIRST - PROMPT & WHITEBOOK TEÓRICO POLYDIM V46

# PROMPT PARA LOS SABIOS - POLYDIM V46 TARDE
# Copiar este texto como mensaje y adjuntar el archivo "codigo_consolidado_v46_tarde.txt"

---

## CONTEXTO

Soy investigador en Computabilidad Geométrica y Espacios Nativos de Alta Dimensión (D >= 10,000). Este archivo .txt contiene 9 archivos fuente concatenados uno detrás del otro, delimitados por líneas de "=====" y la etiqueta "FILE: nombre.ext". Los archivos son:

1. **WHITEBOOK_POLYDIM_V46.md** — Ontología y constitución teórica del proyecto.
2. **polydim_motor_v46.py** — Motor central: SLERP en S^(D-1), media de Fréchet, TSQR bloqueado, receptor PMTP con anti-replay HMAC/BLAKE2b.
3. **lib_v46.rs** — Rust/PyO3: Anillo PMTP lock-free (MPMC), HMAC nativo blake2.
4. **slerp_kernel_v46.cpp** — C++ AVX2/FMA: Kernel SLERP con Kahan dot product, scratch buffer del caller.
5. **silicon_contract.py** — Derivación dinámica de constantes de hardware (eps, tiny, page_size).
6. **pmtp_binary_transfer_v46.py** — Validación de transferencia binaria zero-loss vía mmap.
7. **stress_8h_auditor.py** — Harness de estrés de 8 horas.
8. **polydim_suite_v46.py** — Suite de pruebas empíricas (CHK_28 a CHK_31).
9. **CERTIFICADO_ESTRES_8H_V46.md** — Resultados de la última corrida de pruebas.

Los archivos .rs son Rust, los .cpp son C++, los .py son Python. Leer TODO el contenido como código fuente válido dentro del .txt.

## MISIÓN

Actúa como auditor adversarial inflexible (Red Team). Tu objetivo NO es validar el código, sino intentar ROMPERLO. Evalúa línea por línea sin piedad pero con respeto profesional. Busca específicamente:

1. **Vectores de memoria:** Buffer overflows, fugas de memoria, asignaciones fantasma (NxD innecesarias), punteros no alineados para SIMD.
2. **Vectores de concurrencia:** Race conditions en el bitmap anti-replay de PMTP, operaciones no atómicas (|=, &=), ausencia de locks.
3. **Vectores numéricos:** Cancelación catastrófica (arccos(dot) vs arctan2), subnormales IEEE-754, ceros negativos (-0.0 vs 0.0) que rompan hashes criptográficos, divisiones por casi-cero.
4. **Vectores FFI:** Desalineación de firmas entre Python (ctypes) ↔ C++ ↔ Rust (PyO3). ¿Los argtypes coinciden con la firma C? ¿scratch_size es en elementos o en bytes?
5. **Vectores asintóticos:** ¿Qué pasa cuando D = 10,000,000? ¿El código escala o colapsa por OOM?

## FORMATO DE RESPUESTA

Para cada vulnerabilidad encontrada, reporta:
- **Ubicación:** Archivo y línea/función.
- **Mecanismo de destrucción:** Cómo se rompe.
- **Consecuencia:** Qué falla (segfault, NaN silencioso, OOM, DoS, etc.).
- **Solución propuesta:** Fix concreto.

Si el código aguanta el ataque, dilo sin rodeos. No inventes vulnerabilidades falsas.


---

# 📖 WHITEBOOK TÉCNICO Y CONSTITUCIÓN TEÓRICA: POLYDIM V46
**Título:** Computabilidad Geométrica en Espacios Nativos de Alta Dimensión ($S^{D-1}, D \ge 100,000$) y Protocolo de Comunicación Tensorial Nativa (PMTP)  
**Autor:** Ariel / Antigravity Orchestrator  
**Fecha:** 22 de Agosto de 2026  
**Estatus:** 🔴 **EN AUDITORÍA ADVERSARIAL (Protocolo Bulldog V46)**  
**Versión anterior:** V45 (Congelada con 32 bugs identificados y parchados)

---

## 0. CHANGELOG V45 → V46

| Área | V45 (Bug) | V46 (Corrección) |
|------|-----------|-------------------|
| Umbral Antipodal | $\epsilon^{2/3}$ sin justificación | $\sqrt{\epsilon}$ (Higham 2002, §3.5) |
| Tangente Determinista | SHAKE-256 directo → 800KB/vector | Seed 32B → PCG64 → `randn` → Gram-Schmidt |
| HMAC Criptográfico | `hmac.new(key, data, blake2b)` doble-hash | `blake2b(data, key=key)` nativo 1-pass |
| Ring Buffer Rust | `ManuallyDrop` + `mem::forget` (leak en panic) | `Vec<f64>` RAII (drop automático) |
| Kernel C++ `slerp` | 6 argumentos, sin validación de scratch | 7 argumentos: `scratch_size` + bounds check |
| `slerp_batch` NumPy | 4x alocación (P_unit, Q_unit, V_anti, out) | In-place mutation, sin V_anti densa |
| `machine_eps(dtype)` | Ignoraba `dtype`, siempre float64 | Respeta `dtype` vía `np.finfo(dtype).eps` |
| Bitmap anti-replay | Sin máscara → overflow en `window_bitmap` | `&= mask` obligatorio tras cada OR |
| TSQR `block_size` | Sin clamp superior | `min(block_size, D)` añadido |
| Auditor 8H | Keys hardcodeadas, `verify=False`, solo DeepSeek | `os.getenv`, `verify=True`, rotación 4 IAs |

---

## 1. DOGMA CENTRAL Y PRINCIPIOS FUNDAMENTALES (EL "NO-GUSANO")

### 1.1 La Tragedia del Ingeniero y la Desigualdad de Procesamiento de Datos (DPI)
Las arquitecturas convencionales de IA cometen un error entrópico fatal: obligan a modelos latentes de alta dimensión ($D \ge 10,000$) a colapsar sus estados internos a secuencias de tokens 1D (JSON, cadenas de texto, payloads MCP) en cada paso de interacción entre agentes. Por la **Desigualdad de Procesamiento de Datos (DPI)**:
$$I(X; Z) \le I(X; Y)$$
Cada colapso proyectivo a 1D destruye de manera irreversible la entropía geométrica y las relaciones de ortogonalidad del espacio latente. 

POLYDIM propone eliminar completamente el "gusano 1D" intermedio: los agentes de IA (LatentMAS) deben comunicarse directamente intercambiando tensores nativos $S^{D-1}$ sobre memoria compartida anónima a velocidad de silicio ($\ge 12\text{ GB/s}$).

### 1.2 El Puente Cuántico de Rotores y Isometrías
A diferencia de las arquitecturas Transformer estadísticamente no unitarias, POLYDIM opera mediante **rotores de Clifford e isometrías Riemannianas unitarias**. Las operaciones de transporte preservan la norma ($\|v\| = 1.0$) y la métrica angular, haciendo a POLYDIM la única arquitectura de IA naturalmente compatible y cuantizable para el hardware cuántico del futuro.

---

## 2. FORMALISMO MATEMÁTICO DE NÚCLEO V46

### 2.1 Métrica Geodésica Estable (Fórmula Kahan atan2)
Para evitar la pérdida catastrófica de precisión (*catastrophic cancellation*) cerca de vectores casi-coincidentes ($p \approx q$), el ángulo geodésico $\omega$ se calcula mediante la fórmula de Kahan:
$$\omega(p, q) = 2 \cdot \text{atan2}\left(\|p - q\|_2, \|p + q\|_2\right)$$

La interpolación esférica geodésica (SLERP) se define como:
$$\text{SLERP}(p, q, t) = \frac{\sin((1-t)\omega)}{\sin(\omega)} p + \frac{\sin(t\omega)}{\sin(\omega)} q$$

### 2.2 Escape Antipodal Determinista vía Seed Criptográfico (SHAKE-256 → PCG64)
En el punto de singularidad antipodal ($p \approx -q \implies \omega \to \pi$), $\sin(\omega) \to 0$. Para evitar la indeterminación $0/0$, se genera una tangente determinista ortogonal $v \perp p$ mediante un pipeline de dos etapas:

**Etapa 1 (Seed Criptográfico):** Se extrae un seed compacto de 32 bytes del vector $p$ usando SHAKE-256:
$$\text{seed}_{32} = \text{SHAKE256}(\text{bytes}(p), \text{length}=32)$$

**Etapa 2 (Expansión PCG64 + Gram-Schmidt):** El seed alimenta el generador `np.random.default_rng(seed)`, que produce un vector pseudo-aleatorio de dimensión $D$ con velocidad nativa del silicio. Se aplica Gram-Schmidt para ortogonalizar:
$$v_{\text{raw}} = \text{PCG64}(\text{seed}_{32}), \quad v = \frac{v_{\text{raw}} - \langle v_{\text{raw}}, p \rangle p}{\|v_{\text{raw}} - \langle v_{\text{raw}}, p \rangle p\|}$$

La interpolación antipodal queda:
$$\text{SLERP}(p, -p, t) = \cos(t\pi) \cdot p + \sin(t\pi) \cdot v$$

> **Justificación V46:** El método anterior (SHAKE-256 directo a $D$ bytes) generaba 800KB de entropía por vector, destruyendo el rendimiento en el hot-loop. El pipeline seed → PCG64 reduce la carga criptográfica a 32 bytes constantes independientemente de $D$.

### 2.3 Factorización Tree-TSQR con Reconstrucción Householder
Para ortogonalizar matrices de alta dimensión $A \in \mathbb{R}^{D \times K}$ ($D = 100,000, K = 32$), se implementa Tree-TSQR en bloques con reconstrucción de matrices de Householder $H_k = I - \tau_k v_k v_k^T$, garantizando que la matriz $Q$ preserva la ortogonalidad bajo la cota de Higham:
$$\|Q^T Q - I\|_F \le 10 \cdot D \cdot \epsilon_{\text{silicon}}$$

> **Guarda V46:** El tamaño de bloque se acota por `block_size = min(block_size, D)` para prevenir que un bloque exceda la dimensión total de la matriz.

### 2.4 Media de Fréchet Riemanniana en la Variedad de Stiefel $V_k(\mathbb{R}^D)$
El centroide geodésico de una nube de tensores $V = \{v_i\}_{i=1}^N$ se calcula minimizando la suma de distancias geodésicas cuadradas sobre la variedad Riemanniana:
$$\mu^* = \arg\min_{\mu \in S^{D-1}} \sum_{i=1}^N \text{dist}_{S^{D-1}}(\mu, v_i)^2$$
mediante descenso de gradiente en el espacio tangente $T_\mu S^{D-1}$ con retracción geodésica explícita.

---

## 3. DOGMA CERO: CONTRATO DE SILICIO (ANTI-HARDCODING)

### 3.1 Principio Fundacional
> **El software no asume. El software interroga.**

Está terminantemente prohibido introducir parámetros estáticos (ej. 64 bytes cache line, $10^{-15}$ precisiones mágicas, 0.9995 thresholds). Todo límite numérico o métrica de silicio es derivado en runtime por el módulo umbrella `SiliconContract`.

### 3.2 Ecuaciones Dinámicas de Umbrales (V46 Corregidas)
$$\theta_{\text{collinearity}}(D, \epsilon) = 16.0 \cdot \epsilon \cdot \sqrt{D}$$
$$\theta_{\text{antipodal}}(D, \epsilon) = \max\left(100.0 \cdot \epsilon \cdot \sqrt{D},\; \sqrt{\epsilon}\right)$$
$$\theta_{\text{dual\_guard}}(D, \epsilon) = 100.0 \cdot \epsilon \cdot \sqrt{D}$$

donde $\epsilon = \texttt{np.finfo(dtype).eps}$, derivado dinámicamente del `dtype` activo (no hardcodeado a float64).

> **Corrección V46:** La cota antipodal $\epsilon^{2/3}$ de V45 fue reemplazada por $\sqrt{\epsilon}$ siguiendo Higham (2002, §3.5). La cota $\epsilon^{2/3}$ carecía de justificación teórica publicada y generaba un umbral demasiado agresivo en $D > 10^5$.

### 3.3 Respeto Estricto de `dtype` (V46)
Las funciones `machine_eps(dtype)` y `machine_tiny(dtype)` ahora llaman a `np.finfo(dtype).eps` directamente, respetando el tipo de dato inyectado. Esto permite operar en Mixed Precision (float32 para I/O, float64 para geometría) sin que los umbrales colapsen al valor global de float64.

---

## 4. ARQUITECTURA DEL TRIPLE NÚCLEO V46

```
+-----------------------------------------------------------------------+
|                       SILICON CONTRACT UMBRELLA                       |
|         (Interrogación Dinámica de Hardware: CPU, Cache, SIMD)        |
+-----------------------------------------------------------------------+
                                   |
       +---------------------------+---------------------------+
       |                           |                           |
+--------------+           +---------------+           +---------------+
| MOTOR JAX    |           | KERNEL C++    |           | RING BUFFER   |
| (Float64)    |           | (Zero Heap)   |           | (Rust RAII)   |
+--------------+           +---------------+           +---------------+
| safe_sin     |           | scratch buf   |           | Vec<f64>      |
| Doble Guarda |           | scratch_size  |           | KeyedBlake2b  |
| PCG64 tangent|           | FMA SIMD      |           | fence(Release)|
| in-place bat.|           | bounds check  |           | CAS Acquire   |
+--------------+           +---------------+           +---------------+
```

### 4.1 `silicon_contract.py` — Interrogación de Hardware
Interroga la línea de caché L1D (64B/128B), el tamaño de página del SO, el ancho SIMD y calcula $\epsilon$ y umbrales dinámicos para el `dtype` activo.

### 4.2 `polydim_motor_v46.py` — Motor Numérico Float64 JAX/NumPy
Motor con piso anti-subnormales `safe_sin`, validación estricta de dominios $S^{D-1}$, tangente determinista vía PCG64, `slerp_batch` con mutación in-place (sin alocación densa de `V_anti`), y criptografía BLAKE2b nativa con `key=` (sin doble-hash HMAC).

### 4.3 `slerp_kernel_v46.cpp` — Kernel C++ de Bajo Nivel
Firma actualizada con 7 argumentos:
```cpp
extern "C" void slerp(
    const double* p, const double* q, double t,
    double* out, double* scratch,
    size_t D, size_t scratch_size  // ← V46: bounds check obligatorio
);
```
- Validación `scratch_size >= D` antes de escribir (abort si falla)
- FMA (`_mm256_fmadd_pd`) para dot product con compensación Kahan solo en reducción horizontal final
- Guard de subnormales: si `|sin(ω)| < ε`, fallback a Normalized LERP
- Umbral antipodal: `sqrt(eps)` derivado

### 4.4 `lib_v46.rs` — Ring Buffer Rust RAII
- Almacenamiento primario: `Vec<f64>` con Drop automático (cero leaks en panic)
- Concurrencia MPMC lock-free: `CAS(Acquire/Relaxed)` + `fence(Release)` antes de publicación
- Criptografía: `blake2::KeyedBlake2b512` nativa (1-pass, cero alocación)
- Anti-replay: bitmap con máscara `&= mask` obligatoria

---

## 5. PROTOCOLO DE AUDITORÍA V46 (Regla de los 5 Meses)

### 5.1 Prohibición Absoluta de Auditoría Pasiva
Ningún agente puede certificar código leyéndolo. Todo veredicto requiere ejecución física de ataques adversariales con inputs degenerados ($\text{NaN}, \infty, 0$), condiciones de carrera, desbordamiento de memoria y fronteras FFI.

### 5.2 Integridad Inter-Capa
Todo cambio de firma en cualquier capa (Python `ctypes.argtypes`, C++ `extern "C"`, Rust `#[pyfunction]`) debe propagarse inmediatamente a las otras capas y verificarse con un test dedicado.

### 5.3 Protocolo Co-Work
Las auditorías se ejecutan con mínimo 3 sabuesos adversariales atacando vectores distintos (memoria, concurrencia, precisión numérica). Si los 3 pasan, se sospecha de prompts blandos y se reformulan.

---

## 6. EVIDENCIA EMPÍRICA Y CERTIFICACIÓN (PENDIENTE)

> **ESTADO:** La evidencia empírica de V46 será generada tras ejecutar `polydim_suite_v46.py` con los 30 checks (27 heredados + 3 nuevos). Los resultados de V45 no se heredan: cada versión debe recertificarse desde cero.

---

## 7. CONCLUSIÓN
La arquitectura **POLYDIM V46** corrige las inconsistencias documentales y los bugs residuales de V45, alineando completamente el Whitebook con el código ejecutable. La "Regla de los 5 Meses" garantiza que ningún agente podrá volver a certificar código sin ejecutar ataques adversariales físicos.

