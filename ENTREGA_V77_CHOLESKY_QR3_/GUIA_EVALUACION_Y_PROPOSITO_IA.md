# 🎯 GUÍA DE EVALUACIÓN Y PROPÓSITO ARQUITECTÓNICO (POLYDIM V77)
*Documento de Entrada Pedagógico para Alumnos y Nuevas IAs Receptors (Claude, Kimi, GPT, DeepSeek, Llama)*

---

## 1. ¿Cuál es el Propósito de POLYDIM V77?
El objetivo central de **POLYDIM (The Cholesky-QR3 Epoch)** es erradicar el "Colapso del Gusano 2D". 

En la IA moderna, los modelos piensan en hiperespacios tensoriales nativos de alta dimensión ($S^{D-1}$ con $D \ge 10,000$), pero están obligados a colapsar su estado a tokens 1D (texto/JSON/APIs) para comunicarse entre sí. Por la **Desigualdad de Procesamiento de Datos (DPI)**, cada colapso a texto destruye entropía y geometría irreversiblemente.

POLYDIM resuelve esto mediante:
1. **Comunicación Nativa Tensorial (PMTP):** Transferencia directa de latentes cuantizados en INT8 vía sockets y memoria compartida.
2. **Geometría Esférica SOTA en JAX:** Rotaciones en subespacios (Rotores de Clifford) y transporte geodésico diferenciable sin singularidades de gradiente.
3. **Inyección Nativa Zero-Copy (Anti-GIL):** Kernels C++ y Rust compilados en caliente e inyectados al runtime XLA vía `PyCapsule`, excluyendo al recolector y al GIL de Python del bucle crítico.

---

## 2. ¿Cómo debe la IA Receptora Leer y Analizar los Archivos?
Si eres una IA (o un alumno instruyendo a una IA externa), **no analices el código como un script estándar de Python**. Sigue este orden epistemológico:

### Paso 1: Lectura del Whitebook (`WHITEBOOK_POLYDIM_V77.md`)
- Comprende por qué se extirpó **Gram-Schmidt** (por su cuello de botella secuencial $O(N^2 D)$ en hardware paralelo).
- Entiende la formulación de la **distancia cordal con arcsin** $\theta = 2 \arcsin(\|x-y\|/2)$ para resolver la singularidad de $\arccos(1.0)$.

### Paso 2: Auditoría del Puente Nativo FFI (C++ / Rust)
- Localiza las cadenas `CPP_SOURCE` y `RUST_SOURCE` dentro del monolito o del archivo consolidado.
- Verifica cómo `NativeFFIBridge.initialize()` extrae los punteros y forja el `PyCapsule_New(ptr, b"xla._CUSTOM_CALL_TARGET", None)`.
- Evalúa la seguridad de memoria: ¿hay riesgo de punteros colgantes o condiciones de carrera durante la recarga en caliente?

### Paso 3: Auditoría del Grafo JAX y Geometría (`GeodesicKernels` & `CliffordRotors`)
- Evalúa la función `CliffordRotors.cholesky_qr3`: $G = Q^T Q + \epsilon I$, $L = \text{chol}(G)$, $Q_{new} = Q L^{-T}$.
- Evalúa el **Escudo Epsilon** en `_log_map_unit`: `safe_dist = jnp.maximum(dist, eps)`.

### Paso 4: Auditoría del Protocolo de Red (`PMTPAgentBridge`)
- Revisa el empaquetado binario del encabezado `PMTP_HEADER_FMT`.
- Revisa el pipeline de cuantización simétrica INT8 y los **Relojes de Época** (`EpochClock`) que sustituyen a los costosos relojes vectoriales $O(N)$.

---

## 3. ¿Qué debe Evaluar y Destrozar la IA Receptora (Modo Red Team)?
La IA receptora debe responder obligatoriamente a estas 4 preguntas críticas:

1. **¿Bajo qué número de condición $\kappa(G) > 10^6$ falla Cholesky-QR3 en FP32/FP64?** ¿Es suficiente la regularización de Tikhonov $\epsilon I$ añadida?
2. **¿Existe alguna fuga de memoria o problema de alineación SIMD al invocar `polydim_cpp_householder_xla` desde XLA en CPU?**
3. **¿El cálculo del gradiente de la distancia cordal $\nabla_x (2 \arcsin(\|x-y\|/2))$ permanece acotado y libre de `NaN` en una colisión exacta $x=y$?**
4. **¿Puede el `PMTPTokenBucket` ser eludido mediante paquetes TCP fragmentados (ataque tipo Slowloris)?**
