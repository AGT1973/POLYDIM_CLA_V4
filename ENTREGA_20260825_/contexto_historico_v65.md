# Contexto Histórico V65 — Cierre de Sesión 2026-08-25 16:55

## Estado del Proyecto

### GitHub
- **Repo:** `github.com/AGT1973/POLYDIM_CLA_V4.git`
- **PAT:** `[REDACTED_PAT]` (no caduca)
- **Rama remota:** `origin/main` → commit `940ae3d` (V64 clean release, 1129 archivos)
- **Rama local activa:** `main_clean` (tracking `origin/main`)
- **PENDIENTE:** Subir V65 a GitHub (commit incremental con los 3 archivos nuevos V65)

### Archivos V65 generados y probados
Todos en `E:\POLYDIM_EINSOF\ENTREGA_20260825_\`:

| Archivo | Tamaño | Estado |
|---|---|---|
| `polydim_v65_monolito.py` | 41.0 KB | ✅ Ejecutado, 6/7 PASS (FFI warn esperado: no hay MSVC en PATH) |
| `WHITEBOOK_POLYDIM_V65.md` | 5.4 KB | ✅ Reescrito con honestidad radical |
| `evaluacion_proposito_polydim.md` | 5.1 KB | ✅ Tabla V62→V64→V65 + análisis alucinaciones |
| `codigo_consolidado_v65.txt` | 52.2 KB | ✅ Combina Whitebook + Monolito + Evaluación |

### 21 correcciones V65 aplicadas y verificadas por ejecución
1. Docstring V58→V65
2. Storage: shape completa ndim + shape[0..7]
3. Storage: tabla dtypes (float16/32/64, int32/64) con rechazo
4. Storage: checksum CRC32
5. Storage: validación tamaño archivo vs payload
6. HTTP: ThreadingHTTPServer (anti DoS)
7. HTTP: routing con 404
8. HTTP: endpoint /capabilities
9. TCP: listener multi-thread
10. TCP: timeout 10s
11. TCP: backpressure MAX_INBOX_SIZE=1000
12. TCP: listen(128)
13. MCP: validación campos requeridos
14. MCP: soporte FP64
15. MCP: rechazo tensores vacíos
16. FFI: restype declarado
17. FFI: return code check
18. FFI: check=True en build
19. C++/Rust: norma escalada anti-overflow
20. Rotor: apply_spherical_rotor / apply_linear_rotor separados
21. assert_isometry: atol escalado sqrt(D)*eps

### Auditoría cruzada entrega.md (GLM-5.3)
- 192 errores reportados por GLM-5.3
- ~60 reales verificados, ~15-20 alucinaciones, ~40-50 redundancias
- Inflación ~3x del conteo
- Auditoría completa en: `C:\Users\eluithi\.gemini\antigravity\brain\fced2a9c-7f94-4a5f-9aa1-3abc1ae1830d\auditoria_entrega_md.md`

### Cuotas agotadas
- **Modelo `pro`:** 429 RESOURCE_EXHAUSTED, reset en ~138h (~6 días desde 2026-08-25 15:34)
- El modelo actual usado es Claude Opus 4.6

### Google Drive backup
- `I:\Mi unidad\POLYDIM_EINSOF_RESPALDO_V64\` — contiene V64, NO V65 aún

### Reglas clave recordar
- Regla 18: máximo 4 archivos para subir a plataforma (LEEME no cuenta)
- Regla 5: NUNCA borrar _HISTORICO
- Regla 11: NUNCA ejecutar desde Google Drive
- Regla 17: NUNCA certificar sin ejecutar
- Git: archivos >100MB excluidos via .gitignore (_HISTORICO/*.dat, *.npy, *.pt, *.bin, *.dll)

## Próximos pasos para nueva sesión
1. `git add` los 4 archivos V65 + `git commit` + `git push origin main_clean:main`
2. Copiar V65 a Google Drive como respaldo pasivo
3. Considerar probar FFI con MSVC si se configura el PATH de Visual Studio
