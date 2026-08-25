# WHITEBOOK POLYDIM V47.0
## Protocolo y Arquitectura de Computabilidad Geométrica en Espacios de Alta Dimensión

### 1. DOGMA CENTRAL Y PRINCIPIOS DE DISEÑO

#### 1.1 El Principio No-Gusano
La inteligencia artificial moderna sufre de una atrofia infraestructural: forzar pensamientos representados en tensores de alta dimensión ($D \ge 10,000$) a colapsar secuencialmente a cadenas de texto 1D (JSON, MCP, tokens) en cada paso de razonamiento destruye la entropía geométrica por la **Desigualdad de Procesamiento de Datos (DPI)**.

POLYDIM v47 establece el Protocolo de Memoria Tensorial Protegida (PMTP v47), permitiendo a los agentes de IA (LatentMAS) comunicarse directamente intercambiando estados nativos en la esfera unitaria $S^{D-1}$ sobre memoria compartida anónima a velocidad de silicio ($\ge 12\text{ GB/s}$).

#### 1.2 Geometría Riemanniana y Rotores Clifford
POLYDIM utiliza geodésicas en la variedad Riemanniana esférica $S^{D-1}$ (interpolación esférica SLERP) y factorizaciones ortogonales en la variedad de Stiefel $V_k(\mathbb{R}^D)$. Estas operaciones preservan la norma euclídea y la métrica angular, siendo numéricamente unitarias e isométricas, lo que permite la cuantización directa a hardware cuántico futuro (QPUs) sin pérdida de coherencia.

---

### 2. FORMULACIÓN MATEMÁTICA RIGUROSA

#### 2.1 Métrica Angular y Fórmula de Kahan (atan2)
Para prevenir la cancelación catastrófica de floating-point cuando dos tensores son casi coincidentes ($\|p - q\| \to 0$) o casi antipodales ($\|p + q\| \to 0$), POLYDIM prohíbe el uso directo de $\arccos(\langle p, q \rangle)$.

El ángulo geodésico $\omega$ entre dos vectores unitarios $p, q \in S^{D-1}$ se calcula exclusivamente mediante la **fórmula estable de Kahan**:

$$\omega = 2 \cdot \text{atan2}\left(\|p - q\|_2, \|p + q\|_2\right)$$

#### 2.2 Tangente Determinista en la Frontera Antipodal
Cuando dos tensores son antipodales ($q \approx -p$, $\omega \approx \pi$), la geodésica SLERP no es única. Para garantizar un comportamiento determinista entre nodos distribuidos sin comunicación adicional, POLYDIM define una tangente ortogonal única derivando un vector pseudoaleatorio a partir de la huella SHAKE-256 del tensor $p$:

1. Muestreo estratificado de $p$ a lo largo de las $D$ dimensiones.
2. Normalización binaria IEEE 754 purgando el bit de signo $-0.0 \to +0.0$.
3. Hash SHAKE-256(128 bytes) $\to$ Semilla PCG64 / PRNGKey JAX.
4. Generación de vector Gaussiano $v_{\text{raw}} \sim \mathcal{N}(0, I_D)$.
5. Proyección Gram-Schmidt sobre $T_p S^{D-1}$:
   $$v_{\text{proj}} = v_{\text{raw}} - \langle v_{\text{raw}}, p \rangle p$$
   $$v_{\text{anti}} = \frac{v_{\text{proj}}}{\|v_{\text{proj}}\|_2}$$

#### 2.3 Cota Asintótica de Higham para TSQR
Para la ortogonalización de matrices altas $A \in \mathbb{R}^{N \times D}$ ($N \gg D$), POLYDIM emplea TSQR (Tall-Skinny QR). De acuerdo con Higham (2002), la desviación de ortogonalidad de la matriz $Q$ obtenida satisface la cota:

$$\|Q^T Q - I\|_F \le c \cdot K \cdot \sqrt{D} \cdot \epsilon_{\text{silicon}}$$

donde $c$ es una constante del alocador de silicio, $K$ es el número de bloques y $\epsilon_{\text{silicon}} = 2.2204 \times 10^{-16}$ para `float64`.

---

### 3. EL CONTRATO DE SILICIO (AXIOMA CERO)

#### 3.1 El Axioma Cero (Anti-Hardcoding)
*El software no asume. El software interroga.*  
Está terminantemente prohibido hardcodear parámetros de hardware (líneas de caché, tamaños SIMD, umbrales colineales fijados). Todo parámetro se calcula dinámicamente en tiempo de ejecución.

#### 3.2 Umbrales Asintóticos Dinámicos
Los umbrales de transición de régimen numérico para cualquier dtype y dimensión $D$ son:

$$\theta_{\text{small}}(D) = 16.0 \cdot \epsilon_{\text{silicon}} \cdot \sqrt{D}$$
$$\theta_{\text{antipodal}}(D) = \max\left(100.0 \cdot \epsilon_{\text{silicon}} \cdot \sqrt{D}, \sqrt{\epsilon_{\text{silicon}}}\right)$$

---

### 4. ESPECIFICACIÓN DEL PROTOCOLO PMTP V47

#### 4.1 Encabezado y Autenticación Criptográfica
Cada paquete de memoria tensorial contiene:
- `header`: Domain Separator `POLYDIM_PMTP_V47` + `epoch` (u64 LE) + `seq` (u64 LE).
- `tag`: BLAKE2b-512 keyed HMAC de 64 bytes sobre `(header + payload)`.
- Key Derivation: HKDF-BLAKE2b expandiendo `master_key` con contexto `POLYDIM_PMTP_V47_EPOCH_<epoch>`.

#### 4.2 Protección Anti-Replay Thread-Safe
El receptor `PmtpStatefulReceiver` implementa una ventana deslizante de 64 a 256 secuencias protegida por `threading.Lock()`. Mantiene sincronización atómica para paquetes en orden (`seq > last_seq`), cambio de época (`epoch > last_epoch`) y paquetes desordenados dentro de la ventana (`seq <= last_seq`).

---

### 5. HISTORIAL DE CAMBIOS (CHANGELOG V47.0)

| Módulo | Versión Previa (V46) | Versión SOTA (V47) | Justificación Térmica / Numérica |
| :--- | :--- | :--- | :--- |
| **PMTP Anti-Replay** | Bitmask no actualizado en `seq <= last_seq` | Lock total + actualización de bitmask en todas las ramas | Previene DoS por replay de paquetes fuera de orden |
| **TSQR Matrix Q** | `Q_final` hardcodeado a matriz de ceros | Reconstrucción completa $Q_{\text{loc}} @ Q_{\text{top}}$ | Preserva la ortogonalidad $A = QR$ |
| **Fréchet Mean** | `pass` en cut locus antipodal | Perturbación ortogonal determinista en $\omega \approx \pi$ | Garantiza convergencia fuera de puntos de silla |
| **JAX XLA JIT** | Recompilación en bucle / Truncamiento $f32$ | `@jax.jit` a nivel de módulo + `jax_enable_x64` | Cero recompilación + precisión $f64$ garantizada |
| **Rust MPMC** | `compiler_fence` / UB por alias `&self` | `UnsafeCell` + `fence(Release/Acquire)` físicos | Thread-safe real en arquitecturas ARM64 y x86 |
