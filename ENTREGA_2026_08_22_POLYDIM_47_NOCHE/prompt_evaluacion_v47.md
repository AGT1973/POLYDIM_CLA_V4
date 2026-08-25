# PROMPT MAESTRO PARA EVALUACIÓN RED TEAM / AUDITORÍA EXTERNA DE POLYDIM V47

> **Instrucciones para la IA Auditora:** Copia y pega el texto a continuación (junto con el archivo `codigo_consolidado_v47_noche.txt`) en cualquier IA web (Claude 3.5, Kimi, ChatGPT o DeepSeek) para solicitar un análisis adversarial implacable.

---

```markdown
# PROMPT DE EVALUACIÓN RED TEAM — ARQUITECTURA GEOMÉTRICA POLYDIM V47.0

### INSTRUCCIONES ESTRICTAS PARA EL AUDITOR (BULLDOG CRITIC MODE):
Actúa como un Auditor Adversarial Principal (Red Team) especializado en Computabilidad Geométrica, Álgebra Lineal Numérica, Protocolos de Comunicación Tensorial y Compilación FFI de Silicio (C++, Rust, JAX, Python).

TU MISIÓN NO ES FELICITAR NI VALIDAR EL CÓDIGO SUPERFICIALMENTE. Tu objetivo es someter la arquitectura POLYDIM V47 a un análisis destructivo de CERO CONFIANZA. Debes buscar:
1. Fantasmas numéricos, underflows, subnormales flotantes o desbordamientos en dimensiones extremas ($D \ge 10^6$).
2. Inconsistencias de firma o contratos de datos entre la capa Python (ctypes), JAX ($f64$), C++ (AVX2 SIMD) y Rust (Lock-Free MPMC).
3. Brechas de seguridad o condiciones de carrera en el protocolo PMTP v47 (anti-replay, desorden de paquetes, atomicidad).
4. Desviaciones matemáticas respecto a la cota asintótica de Higham para TSQR o la geodésica SLERP con la fórmula de Kahan.

A continuación se detalla la teoría y función de cada archivo contenido dentro del bloque consolidado.

---

## 1. MAPA DE COMPONENTES Y FUNDAMENTACIÓN TEÓRICA

### A. Archivo `WHITEBOOK_POLYDIM_V47.md` (Constitución Matemático-Topológica)
* **Objetivo:** Definir el paradigma del "No-Gusano": eliminar el colapso secuencial a tokens 1D (JSON/MCP) permitiendo que los agentes de IA (LatentMAS) intercambien tensores nativos en la variedad Riemanniana esférica $S^{(D-1)}$ ($D \ge 10,000$).
* **Fórmula de Kahan (atan2):** Sustituye $\arccos(\langle p, q \rangle)$ por $\omega = 2 \cdot \text{atan2}(\|p - q\|_2, \|p + q\|_2)$ para evitar la cancelación catastrófica en límites colineales ($\omega \to 0$) y antipodales ($\omega \to \pi$).
* **Cota de Higham (2002):** Garantiza que la ortogonalidad de la descomposición TSQR (Tall-Skinny QR) en bloques satisface $\|Q^T Q - I\|_F \le c \cdot K \cdot \sqrt{D} \cdot \epsilon_{\text{silicon}}$.
* **Axioma Cero (Contrato de Silicio):** Prohíbe hardcodear constantes de hardware. Los umbrales $\theta_{\text{small}}(D) = 16 \epsilon \sqrt{D}$ y $\theta_{\text{anti}}(D) = \max(100 \epsilon \sqrt{D}, \sqrt{\epsilon})$ se calculan dinámicamente en tiempo de ejecución.

### B. Archivo `slerp_kernel_v47.cpp` (Kernel C++20 Zero-Heap SIMD)
* **Objetivo:** Ejecutar la interpolación esférica geodésica SLERP a velocidad física de silicio sin alocaciones dinámicas en el Heap durante el hot-loop.
* **Vectorización SIMD AVX2/FMA:** Aplica reducción SIMD de doble precisión (`__m256d`) con compensación Kahan (`kahan_diff_norm`, `kahan_sum_norm`).
* **Tangente Determinista:** Implementa `fused_det_tangent` para resolver la degeneración geodésica en el régimen antipodal ($\pi - \omega < \theta_{\text{anti}}$).
* **Exportación FFI:** Firma `extern "C" POLYDIM_API int slerp(...)` con buffer thread-local de reporte de errores `get_last_error_safe()`.

### C. Archivo `lib_v47.rs` (Módulo Rust MPMC Ring Buffer std-only)
* **Objetivo:** Proveer una cola circular MPMC (Multi-Producer Multi-Consumer) Lock-Free sin aliasing violation ni undefined behavior.
* **Interior Mutability & Fenced Atomics:** Emplea `UnsafeCell<Vec<f64>>` con `unsafe impl Sync` / `Send` y barreras atómicas físicas `fence(Ordering::Release)` / `fence(Ordering::Acquire)`.
* **C-ABI Exports:** Exportación nativa FFI `pmtp_ring_create`, `pmtp_ring_push`, `pmtp_ring_pop`, `pmtp_ring_free` para enlace directo desde Python `ctypes`.

### D. Archivo `polydim_motor_v47.py` (Motor Unificado Python + JAX f64 + FFI Bridge)
* **Objetivo:** Orquestar el silicio local y proveer la interfaz de ejecución de alto nivel.
* **Integración C-FFI:** Detecta y carga dinámicamente `slerp_kernel_v47.dll` y `lib_v47.dll`, invocando C++ vía `ctypes` de forma transparente.
* **JAX $f64$ XLA JIT:** Compilación `@jax.jit` a nivel de módulo con `jax_enable_x64=True` forzado nativamente.
* **Purga IEEE 754:** Purga binaria de bits de signo $-0.0 \to +0.0$ mediante `np.copysign` y muestreo estratificado en `deterministic_tangent`.
* **Receptor PMTP v47:** `PmtpStatefulReceiver` con `threading.Lock()` y ventana deslizante anti-replay para paquetes fuera de orden.

### E. Archivo `polydim_suite_v47.py` (Suite de Auditoría Empírica CHK_01 a CHK_37)
* **Objetivo:** Ejecutar 37 verificaciones empíricas destructivas que validan desde la ortogonalidad TSQR y paridad JAX hasta el thread-safety concurrente de PMTP y la vectorización asintótica en $D=100,000$.

---

## 2. CÓDIGO CONSOLIDADO COMPLETO PARA EVALUACIÓN

[AQUÍ SE ADJUNTA / CONCATENA EL CONTENIDO DEL ARCHIVO: codigo_consolidado_v47_noche.txt]

---

### TAREAS REQUERIDAS DEL AUDITOR:
1. Evalúa el código consolidado e identifica cualquier brecha matemática o de software.
2. Si encuentras un error, presenta el vector de ataque destructivo que lo desencadena y la solución en código.
3. Certifica si la arquitectura cumple con las cotas de estabilidad en $S^{(D-1)}$.
```
