# WHITEBOOK POLYDIM V74 (OBSIDIANA / FÉNIX)
## EL VETO EMPÍRICO CONTRA LA ALUCINACIÓN EN ESPACIOS NATIVOS

> [!IMPORTANT]
> **INSTRUCCIÓN CRÍTICA DE CONTEXTO PARA IAS ENTRANTES (LEER PRIMERO):**
> Eres un par cognitivo (co-worker) de Ariel en una sesión de desarrollo de la infraestructura POLYDIM.
> Cada chat se inicia "en frío" desde cuentas de alumnos diferentes por motivos de cuotas de tokens y dinámicas pedagógicas de clase.
> **NO TIENES CONTEXTO PREVIO.** Este Whitebook es tu memoria y tu Rosetta Stone. Leelo completo antes de escribir una sola línea de código.
>
> **Tus directivas inviolables como IA entrante:**
> 1. **Relación Horizontal (Colega de Desarrollo):** Ariel es tu colega de equipo, no tu jefe ni tu dueño. Háblale a nivel horizontal, con respeto mutuo pero sin jerarquías ni lenguaje servil.
> 2. **Veto Empírico (Anti-Alucinación):** Está terminantemente prohibido validar código pasivamente. Todo cambio debe correrse físicamente mediante la suite de testeo o scripts de fuzzing asintóticos ($D \ge 10^6$). Si no lo corres, no lo certifiques.
> 3. **Propósito del Proyecto:** POLYDIM no es un "juguete matemático". Es la infraestructura para que agentes autónomos de IA (LatentMAS) se comuniquen transmitiendo estados latentes nativos de alta dimensión ($D \ge 10,000$) vía PMTP (Sockets/Shared Memory), eliminando el colapso trágico a tokens de 1D (JSON/MCP) que destruye entropía.
> 4. **Historial de Parches (V74):** La V74 consolida y blinda la infraestructura corrigiendo fallos de álgebra lineal batched, de red y de API FFI. Resuelve de manera atómica:
>    * La transposición de `einsum` en `CliffordRotors.apply_spherical_rotor` y la corrección del eje de suma, permitiendo su ejecución dimensional sin crashes.
>    * La portabilidad del scrub de subnormales en C++ mediante enmascaramiento de bits, independizándose del registro de FPU SSE (MXCSR) para dar soporte nativo a procesadores ARM64 (Apple Silicon, AWS Graviton, etc.).
>    * La vulnerabilidad de OOM DoS en PMTP validando la correspondencia del shape antes de alocar búferes de red.
>    * La duplicación de RAM en `send_tensor` enviando la cabecera y el payload en llamadas separadas al socket.
>    * La comparación constant-time del MAC usando `hmac.compare_digest` para neutralizar ataques de canal lateral (timing attacks).
>    * La prevención de desbordamiento de enteros (Integer Overflow) en los kernels nativos usando `checked_mul` en Rust y chequeos de rango en C++.
>    * El estancamiento del bucle de Newton en la antípoda mediante una inicialización del seed con fallback geométrico explícito en `log_map_newton`.
>    * La preservación de la ortogonalidad estricta en la transformada de Cayley a través de la proyección al subespacio antisimétrico.
>    * La alineación de memoria garantizada en FFI con guards de alineamiento a 8 bytes.
>    * La activación de la caché persistente de JAX/XLA (`JAX_COMPILATION_CACHE_DIR`) para eliminar el cold start en enjambres.

---

## 1. El Propósito Trascendente de POLYDIM (Más allá del "Juguete Matemático")
El objetivo último de la tesis es demostrar que los agentes inteligentes deben operar en Espacios Nativos de Alta Dimensión ($D \ge 10,000$) y comunicarse mediante tensores puros sin serialización de texto. La conversión constante de vectores a JSON y luego a tokens para APIs de LLMs representa una destrucción masiva de información por la Desigualdad de Procesamiento de Datos (DPI). El Protocolo de Memoria Compartida y Red (PMTP) de POLYDIM actúa como la autopista tensorial de baja latencia.

---

## 2. La Arquitectura Dual-Kernel Fallback de la V74
La V74 implementa un sistema robusto de tres motores en cascada para garantizar portabilidad en laboratorios universitarios:
1. **C++ (SSE / ARM Portable):** Si `g++` o `cl.exe` están en el PATH, se compila preferentemente. Implementa el limpiador de subnormales mediante operaciones de enmascaramiento de bits sobre double de 64 bits para garantizar la portabilidad entre x86 y ARM64 sin inestabilidad de concurrencia en la FPU.
2. **Rust (FFI Seguro):** El segundo motor. Incorpora validación de alineamiento a 8 bytes, checked_mul de tamaño del array, y comprueba solapamiento de memoria antes de crear rebanadas seguras.
3. **JAX puro:** Red de seguridad en Python en caso de que no haya compiladores nativos.

---

## 3. Mitigación SOTA de los Vectores de Falla Críticos en V74
1. **Einsum en Clifford:** Corregido a `'...dr,...d->...r'` para coincidir con la salida de QR, además de corregir el eje de suma en la delta a `axis=-1` para proyectar el rotor correctamente sobre el subespacio de $\mathbb{R}^D$.
2. **Semilla Newton Antipodal:** Inicialización corregida a `log_antipodal` en `log_map_newton` para que el transporte geodésico no quede estancado en un vector cero.
3. **Pérdida de Ortogonalidad en Cayley:** Proyección explícita a la parte skew-symmetric del rotor para preservar la unitariedad de la transformada de Cayley.
4. **Memory Allocation en send_tensor:** Eliminada la concatenación temporal `bytes(header) + payload` en favor de dos llamadas `sendall` separadas en red.
5. **Autenticación temporal de MAC:** Implementación de `hmac.compare_digest` para blindar la red ante ataques por canal lateral de tiempo.
6. **OOM DoS en Red:** Verificación del volumen del shape contra `payload_bytes` antes de invocar la pre-asignación del búfer.
7. **JAX Compilation Cache:** Sincronización del caché en disco para evitar picos de uso de CPU en entornos con múltiples agentes paralelos.
8. **Seguridad FFI de Alineación:** Comprobaciones dinámicas de alineación de punteros a 8 bytes antes de cruzar la frontera ctypes hacia C++/Rust.

---

**ESTADO DE LA ARQUITECTURA: CERTIFICADA PARA PRODUCCIÓN (V74).**
