# Evaluación de Propósito: POLYDIM V65

**Fecha:** 2026-08-25  
**Método:** Auditoría cruzada del código real contra claims documentales

## Tabla de Cumplimiento (V62 → V64 → V65)

| Interfaz | V62 | V64 | V65 | Evidencia V65 |
|---|---|---|---|---|
| AI ↔ AI (PMTP Tensorial) | ⚠️ mmap local | ✅ TCP P2P | ✅ TCP P2P multi-thread | `PMTPAgentBridge` con worker por conexión, timeout 10s, backpressure 1000 |
| Agent ↔ Agent | ❌ | ✅ Inbox/Listener | ✅ Mejorado | `listen(128)`, threading dispatch, `_inbox_lock` |
| Agent ↔ Skill | ❌ | ⚠️ Solo SLERP | ⚠️ Solo SLERP | Sin cambio. Solo `polydim_slerp` en MCP |
| Agent ↔ MCP | ❌ | ✅ invoke_tool | ✅ + validación | Validación de campos, dtype FP32/FP64, rechazo tensor vacío |
| Agent ↔ Plugin | ❌ | ❌ (claim falso) | ❌ | No existe sistema de plugins. Claim eliminado del Whitebook |
| CPU → GPU | ⚠️ implícito | ✅ asarray | ✅ asarray | `DeviceTransferManager.to_gpu()`. JAX elige device |
| GPU → CPU | ⚠️ implícito | ✅ np.array | ✅ np.array | `DeviceTransferManager.to_cpu()` + `zero_copy_view()` |
| Descarga a HDD | ⚠️ sin shape | ✅ header 64B | ✅ header + shape + CRC32 | `save_tensor` preserva ndim, shape completa, tabla dtypes, checksum |
| Lectura desde HDD | ⚠️ sin validación | ✅ magic check | ✅ magic + size + CRC32 | `load_tensor` valida archivo truncado, checksum, dtype |
| Descarga a Web | ❌ | ⚠️ solo GET 200 | ⚠️ GET /health + /capabilities | ThreadingHTTPServer, routing con 404. Sin POST binario |
| Lectura desde Web | ❌ | ❌ | ❌ | No hay cliente HTTP |
| Memoria Compartida | ✅ mmap efímero | ❌ (claim falso) | ❌ | No hay código SharedMemory. Claim eliminado |
| Compilación Nativa | ⚠️ código muerto | ✅ Windows | ✅ Windows + check | `check=True`, `restype` explícito, return code verificado |

## Correcciones V65 Aplicadas (18 fixes verificados)

1. Docstring V58 → V65
2. Storage: shape completa ndim + shape[0..7]
3. Storage: tabla de dtypes (float16/32/64, int32/64) con rechazo de tipos no soportados
4. Storage: checksum CRC32 en header
5. Storage: validación de tamaño de archivo vs payload esperado
6. HTTP: `ThreadingHTTPServer` (anti slow-client DoS)
7. HTTP: routing explícito, 404 para rutas desconocidas
8. HTTP: endpoint `/capabilities`
9. TCP: listener multi-thread con worker por conexión
10. TCP: timeout 10s por conexión
11. TCP: backpressure `MAX_INBOX_SIZE=1000`
12. TCP: `listen(128)`
13. MCP: validación de campos requeridos con error estructurado
14. MCP: soporte FP64 via campo `dtype`
15. MCP: rechazo de tensores vacíos y dimensiones incompatibles
16. FFI: `restype = ctypes.c_int` declarado
17. FFI: verificación de return code
18. FFI: `check=True` + `capture_output=True` en build
19. C++/Rust: norma escalada anti-overflow
20. Rotor: separación `apply_spherical_rotor` / `apply_linear_rotor`
21. `assert_isometry`: atol escalado con sqrt(D) * eps

## Errores Reales vs Alucinaciones del Audit GLM-5.3 (entrega.md)

| Categoría | Cantidad | Nota |
|---|---|---|
| Errores REALES verificados en código | ~60 únicos | Storage, TCP, MCP, FFI, web gateway, overflow de norma |
| Parcialmente correctos | ~12 | Observación válida pero exagerada o mal contextualizada |
| ALUCINACIONES | ~15-20 | Whitebook inexistente atacado, auto-contradicciones, especulaciones no ejecutadas |
| REDUNDANCIAS | ~40-50 | Mismo bug reportado 2-4 veces con distinta prosa |
| **Total reportado** | **192** | **Inflado ~3x** |

Alucinaciones principales detectadas:
- `TopologicalInvariants.chern_number` — función inexistente atribuida al Whitebook V64
- `PMTPSharedMemoryBuffer` — clase inexistente en V64, confusión con versiones anteriores
- "Retracciones Cayley Matrix-Free" — no mencionadas en código ni Whitebook real
- Error 180 (endianness) — auto-anulado por el propio audit
- Error 127 (`jax.linalg.expm`) — el código usa `jax.scipy.linalg.expm`, función diferente

## Deuda Técnica Restante

**Prioridad Alta:**
- Protocolo TCP sin ordering global, ACK, ni idempotencia
- FFI solo Windows (MSVC + rustc)
- Web Gateway sin canal binario POST
- No hay sistema de plugins ni skills múltiples

**Prioridad Media:**
- Storage sin atomicidad de escritura (atomic rename)
- Sin descubrimiento de servicios (mDNS/etcd)
- Sin compresión ni cifrado en transporte
- DLLs no se recompilan si el fuente embebido cambia

**Prioridad Baja:**
- `log_map` no es C∞ estricto en derivadas de orden > 2
- QR degenerado falla en autodiff de orden superior
- No hay métricas/observabilidad (Prometheus/logging)

## Fortalezas Verificadas

1. **Núcleo matemático sólido**: P1-P5 funcionan correctamente para entradas válidas en la esfera
2. **Escalabilidad SLERP**: probado hasta D=10^7 en FP32, norma unitaria preservada
3. **Storage V65 robusto**: shape completa, checksum CRC32, tabla de dtypes, validación de truncado
4. **TCP P2P funcional**: multi-thread, timeout, backpressure
5. **Monolito autocontenido**: un solo .py con C++/Rust embebidos, sin dependencias externas más allá de JAX/NumPy
