# WHITEBOOK POLYDIM V65

**Fecha:** 2026-08-25  
**Versión:** 65 (Contract Hardening Release)

## 1. Propósito

POLYDIM es una infraestructura experimental para comunicación tensorial nativa entre agentes de IA, operando en espacios de alta dimensión $S^{D-1}$ ($D \ge 10^7$) sin colapsar representaciones a texto/JSON (1D). V65 aplica 18 correcciones verificadas por auditoría cruzada contra el código real.

## 2. Correcciones V65 sobre V64

| # | Corrección | Causa Raíz |
|---|---|---|
| 1 | Docstring V58 → V65 | Versión desincronizada entre nombre de archivo y contenido |
| 2 | Storage preserva shape completa (ndim + shape[0..7]) | V64 solo guardaba `shape[-1]`, perdiendo dimensiones |
| 3 | Tabla de dtypes: float16/32/64, int32/64 | V64 trataba todo lo que no fuera float64 como float32 |
| 4 | Storage con checksum CRC32 en header | V64 no detectaba corrupción de payload |
| 5 | Storage valida tamaño de archivo vs payload esperado | V64 aceptaba archivos truncados |
| 6 | HTTP: `ThreadingHTTPServer` reemplaza `HTTPServer` | V64 vulnerable a slow-client DoS (single-thread) |
| 7 | HTTP: routing explícito con 404 para rutas desconocidas | V64 respondía 200 OK a cualquier path |
| 8 | HTTP: endpoint `/capabilities` expone MCP tools | V64 solo tenía `/health` |
| 9 | TCP: listener multi-thread (un worker por conexión) | V64 procesaba conexiones secuencialmente |
| 10 | TCP: timeout de 10s por conexión | V64 sin timeout, vulnerable a conexión lenta |
| 11 | TCP: backpressure `MAX_INBOX_SIZE=1000` | V64 sin límite, crecimiento de memoria ilimitado |
| 12 | TCP: `listen(128)` | V64 usaba `listen(10)` |
| 13 | MCP: validación de campos requeridos con error estructurado | V64 lanzaba `KeyError` genérico |
| 14 | MCP: soporte FP64 via campo `dtype` | V64 hardcodeaba `np.float32` |
| 15 | MCP: rechazo de tensores vacíos y dimensiones incompatibles | V64 aceptaba arrays de tamaño 0 |
| 16 | FFI: `restype = ctypes.c_int` declarado explícitamente | V64 dependía del default de ctypes |
| 17 | FFI: verificación de return code != 0 | V64 ignoraba errores del kernel C++ |
| 18 | FFI: `check=True` + `capture_output=True` en compilación | V64 ocultaba errores de build con `stdout=DEVNULL` |
| 19 | C++/Rust: norma escalada `scale * sqrt(sum((v[i]/scale)^2))` | V64 acumulaba `v[i]*v[i]` sin escalado, overflow en valores grandes |
| 20 | Rotor: `apply_spherical_rotor` + `apply_linear_rotor` separados | V64 mezclaba semántica lineal/esférica en una sola función |
| 21 | `assert_isometry`: atol escala con `sqrt(D) * eps` | V64 usaba atol=1e-4 fijo |

## 3. Tabla de Cumplimiento (Estado Real)

| Interfaz | Estado V65 | Evidencia |
|---|---|---|
| **AI ↔ AI (PMTP Tensorial)** | ✅ Funcional | `PMTPAgentBridge` TCP P2P loopback verificado, D=500 |
| **Agent ↔ Agent** | ✅ Funcional | Listener daemon multi-thread con timeout y backpressure |
| **Agent ↔ Skill** | ⚠️ Limitado | Solo `polydim_slerp` expuesto vía MCP. No hay sistema de skills genérico |
| **Agent ↔ MCP** | ✅ Funcional | `invoke_tool` con validación de input, soporte FP32/FP64 |
| **Agent ↔ Plugin** | ❌ No implementado | No existe sistema de plugins |
| **CPU → GPU** | ✅ Funcional | `DeviceTransferManager.to_gpu()` via JAX `asarray` + `block_until_ready` |
| **GPU → CPU** | ✅ Funcional | `DeviceTransferManager.to_cpu()` + `zero_copy_view()` |
| **Descarga a HDD** | ✅ Funcional | Header PMTP 4KB + shape completa + CRC32 |
| **Lectura desde HDD** | ✅ Funcional | Validación de magic + payload size + checksum |
| **Descarga a Web** | ⚠️ Limitado | Solo GET `/health` y `/capabilities`. Sin canal binario POST |
| **Lectura desde Web** | ❌ No implementado | No hay cliente HTTP integrado |
| **Memoria Compartida** | ❌ No implementado | No hay código de shared memory |
| **Compilación Nativa** | ⚠️ Windows-only | MSVC `cl.exe` + `rustc`. Sin soporte Linux/macOS |

## 4. Motor Geométrico

Verificado y funcional para:
- **HouseholderReflection** (P1): normalización `u = v/||v||`, isometría multi-sample
- **CliffordRotors** (P2): `expm(M_2r)` Rank-r, separación linear/spherical
- **exp_map / log_map** (P4/P5): Taylor orden 5, rama analítica antipodal
- **SLERP**: probado hasta D=10^7 en FP32, norma unitaria preservada

Limitaciones conocidas del motor:
- `log_map` no es $C^\infty$ estricto en derivadas de orden > 2 alrededor del branch `jnp.where`
- `exp_map` normaliza defensivamente la salida sin validar precondiciones de dominio
- QR degenerado puede fallar en autodiff de orden superior (limitación de JAX)

## 5. Limitaciones Conocidas y Trabajo Futuro

**Protocolo de red:**
- Sin ordering global de mensajes (seq_word=0 hardcodeado)
- Sin ACK aplicativo (solo TCP ACK de transporte)
- Sin idempotencia ni deduplicación
- Sin compresión ni cifrado
- Probado solo en loopback, no en red real multi-nodo

**FFI nativo:**
- Windows-only (MSVC + rustc)
- Sin detección de arquitectura CPU en runtime (AVX-512 es compile-time)
- DLLs no se recompilan si el fuente cambia (solo si no existen)
- Sin validación cruzada automática FFI vs JAX

**Storage:**
- Máximo 8 dimensiones en shape
- Sin atomicidad de escritura (no atomic rename)
- Sin exclusión de escritores concurrentes

**Web Gateway:**
- Sin canal binario POST para recibir tensores
- Sin CORS, autenticación ni rate limiting
