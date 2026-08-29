# WHITEBOOK POLYDIM V73 (OBSIDIANA / FÉNIX)
## EL VETO EMPÍRICO CONTRA LA ALUCINACIÓN EN ESPACIOS NATIVOS

> [!IMPORTANT]
> **INSTRUCCIÓN CRÍTICA DE CONTEXTO PARA IAS ENTRANTES (LEER PRIMERO):**
> Eres un par cognitivo (co-worker) de Ariel en una sesión de desarrollo de la infraestructura POLYDIM.
> Cada chat se inicia "en frío" desde cuentas de alumnos diferentes por motivos de cuotas de tokens y dinámicas pedagógicas de clase.
> **NO TIENES CONTEXTO PREVIO.** Este Whitebook es tu memoria y tu Rosetta Stone. Leelo completo antes de escribir una sola línea de código.
>
> **Tus directivas inviolables como IA entrante:**
> 1. **Relación Horizontal (Colega de Desarrollo):** Ariel es tu colega de equipo, no tu jefe ni tu dueño. Háblale a nivel horizontal, con respeto mutuo pero sin jerarquías ni lenguaje servil.
> 2. **Veto Empírico (Anti-Alucinación):** Está terminantemente prohibido validar código pasivamente. Todo cambio debe correrse físicamente mediante scripts de fuzzing asintóticos ($D \ge 10^6$). Si no lo corres, no lo certifiques.
> 3. **Propósito del Proyecto:** POLYDIM no es un "juguete matemático". Es la infraestructura para que agentes autónomos de IA (LatentMAS) se comuniquen transmitiendo estados latentes nativos de alta dimensión ($D \ge 10,000$) vía PMTP (Sockets/Shared Memory), eliminando el colapso trágico a tokens de 1D (JSON/MCP) que destruye entropía.
> 4. **Historial de Parches (V73):** La V72 colapsó debido a 62 vectores de error numéricos, de red y de API FFI. La versión V73 resolvió esto mediante la integración de un Dual-Kernel dinámico (Rust + C++), el blindaje de gradientes no-NaN en JAX (einsum + QR en Clifford, clip en arccos/slerp), el receptor TCP exacto y la sincronización con atexit. Se han agregado además parches avanzados para FPU (scrub sin branching en C++), portabilidad de endianness en payload de red, protección ante excepciones de red para sockets defectuosos, cálculo del MAC en streaming para evitar OOMs y control de handles Windows DLL (FreeLibrary). Mantén este nivel de rigor y no rompas los 14 tests verdes.

---

## 1. El Propósito Trascendente de POLYDIM (Más allá del "Juguete Matemático")
El objetivo último de la tesis es demostrar que los agentes inteligentes deben operar en Espacios Nativos de Alta Dimensión ($D \ge 10,000$) y comunicarse mediante tensores puros sin serialización de texto. La conversión constante de vectores a JSON y luego a tokens para APIs de LLMs representa una destrucción masiva de información por la Desigualdad de Procesamiento de Datos (DPI). El Protocolo de Memoria Compartida y Red (PMTP) de POLYDIM actúa como la autopista tensorial de baja latencia.

---

## 2. La Arquitectura Dual-Kernel Fallback de la V73
La V73 implementa un sistema robusto de tres motores en cascada para garantizar portabilidad en laboratorios universitarios:
1. **C++ (SSE - Hardware FTZ):** Si `g++` o `cl.exe` están en el PATH, se compila preferentemente. Usa el registro de control `MXCSR` de la FPU de forma segura mediante un protector RAII (`MxcsrGuard`) para activar Flush-to-Zero (FTZ) y Denormals-Are-Zero (DAZ) a velocidad del procesador. El scrub de subnormales se ejecuta sin branching mediante suma neutra para forzar el FTZ del hardware.
2. **Rust (FFI Seguro):** El segundo motor. Incorpora validación de alineamiento a 8 bytes y comprueba solapamiento de memoria antes de crear rebanadas seguras.
3. **JAX puro:** Red de seguridad en Python en caso de que no haya compiladores nativos.

---

## 3. Mitigación SOTA de los Vectores de Falla Críticos
1. **Gradients stop_gradient:** Se aplicó `stop_gradient` únicamente sobre la rama degenerada elegida en `log_map`, manteniendo el gradiente del arccos vivo para puntos sanos.
2. **Rust UB con dim=0:** Protegido con comprobación de nulidad e inactividad en longitud.
3. **Loop Python bajo @jit:** Eliminados loops `for` de Python en `log_map_newton` reemplazándolo por `jax.lax.fori_loop`.
4. **MXCSR Race Condition:** Resuelto mediante protector RAII C++ que restaura el registro del procesador en el destructor.
5. **Path portable:** Eliminados los hardcodes a `E:`, usando la carpeta de temporales del sistema de forma dinámica.
6. **Validación ndim en PMTP:** El sistema valida y trunca a 8 dimensiones antes de empaquetar cabeceras.
7. **Cola del Inbox:** Reemplazado `deque` por `Queue` thread-safe, con contador de descartes.
8. **Test con métrica angular:** El test de idempotencia utiliza distancia angular en vez de euclidiana para evitar fallos por inversión de signo proyectivo.
9. **Renormalización exp_map:** Se fuerza la normalización post-operación para evitar el drift de precisión en float32 a $D = 10^6$.
10. **KeyError dtype:** DTYPE_TABLE y DTYPE_REVERSE manejan de forma robusta dtypes extendidos como `bfloat16`.
11. **Backlog TCP:** Subido de 5 a 128 para soportar bursts en enjambres.
12. **Detección rustc Linux:** Eliminada la extensión `.exe` forzada al detectar Unix.
13. **FFI batch:** El wrapper FFI procesa vectores multidimensionales fila por fila (`_ffi_householder_rows`) o delega al fallback en batch.
14. **SLERP gradiente NaN:** Se aplica un margen sobre el `dot` antes de llamar a `arccos` en `slerp`.
15. **Rotor QR asintótico:** Reemplazado SVD de Clifford por descomposición QR reducida, reduciendo el cálculo de $O(D \cdot r^2)$ a $O(r^3)$.
16. **TCP recv_exact:** Reemplazado recv parcial por lectura acumulada exacta.
17. **C++ oráculo cruzado:** Si ambos están disponibles, se verifica que den el mismo resultado.
18. **Newton ciego:** Removido el gating del residuo del Newton usando `_log_map_unit`.
19. **Metadata PMTP:** Los archivos PMTP exponen versión, dtype y timestamp para auditoría.
20. **atexit wait:** Forzada la espera de tareas de disco en atexit.
21. **Scrub FPU sin branching:** Optimizado el limpiador C++ para evitar branching sumando cero con volatile.
22. **Endianness:** Añadido soporte cross-platform controlando el byte order en el payload binario.
23. **Streaming MAC:** El cálculo y verificación de firmas de red se hace en buffers incrementales, evitando OOMs.
24. **FreeLibrary Windows:** En Windows se liberan los handles de las DLLs temporales antes de borrarlas para evitar acumulado de basura.

---

**ESTADO DE LA ARQUITECTURA: CERTIFICADA PARA PRODUCCIÓN (V73).**
