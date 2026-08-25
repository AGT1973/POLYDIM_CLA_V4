# REGLAS DE ENTREGA Y PROTOCOLO DE CONSOLIDADOS PARA IAs (LEY ARIEL)

## 📌 PROTOCOLO OBLIGATORIO DE ENTREGAS PARA POLYDIM

1. **PROHIBICIÓN ABSOLUTA DE BINARIOS A IAs:**
   - NUNCA adjuntar o pasar binarios compilados (`.dll`, `.so`, `.exe`) a IAs web o subagentes.
   - Las IAs no pueden descompilar ni evaluar binarios en tiempo de ejecución.
   - Toda evaluación de errores debe hacerse sobre el CÓDIGO FUENTE COMPLETO UTF-8 (`.cpp`, `.rs`, `.py`).

2. **CONSOLIDACIÓN EN ARCHIVO `.txt` / MONOLITO `.py` DOCUMENTADO:**
   - NUNCA entregar fuentes `.cpp` o `.rs` como archivos sueltos descontextualizados.
   - Todos los fuentes de todos los lenguajes (C++, Rust, Python) y la documentación (Whitebook) se consolidan dentro de un único archivo `.txt` (ej. `codigo_consolidado_v47_noche.txt`) o monolito `.py`.
   - Cada sección debe estar claramente delimitada con encabezados Markdown y comentarios explicativos para auditoría rápida.

3. **UBICACIÓN AUTORITATIVA EN DISCO `E:` (ZERO-WASTE):**
   - El espacio de trabajo primario de ejecución y compilación física es el disco local `E:\POLYDIM_EINSOF\ENTREGA_<FECHA>_\`.
   - Google Drive (`I:\`) se utiliza estrictamente para sincronización/respaldo pasivo sin ejecutar nada desde allí.

4. **COMPILACIÓN NATIVA EN CALIENTE:**
   - El monolito `.py` extrae los fuentes de C++ y Rust, invoca los compiladores locales (`cl.exe`, `rustc.EXE`), genera las DLLs en caliente en disco `E:` y las enlaza mediante `ctypes`.
