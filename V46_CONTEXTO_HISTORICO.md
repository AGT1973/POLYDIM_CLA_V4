# ESTADO DE TRANSFERENCIA: POLYDIM V46 (Cierre de Contexto)

## 1. Lo que ya está completado y sellado
- **Bugs V45 (1 al 32) parcheados físicamente:** Rust RAII lock-free implementado, Kernel C++ con chequeos de memoria y vectorización FMA, Motor NumPy sin cuellos de botella de memoria y SHAKE-256 optimizado.
- **Ley Ariel (Regla 17) Inyectada:** La constitución global en `C:\Users\eluithi\.gemini\config\AGENTS.md` ya tiene la orden estricta de prohibir auditorías pasivas y exigir ataques físicos.
- **Estructura V46 Creada:** La carpeta `E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_46` ya existe y contiene los archivos base.
- **Whitebook V46:** Ya fue redactado y alinea todas las ecuaciones y firmas con el código empírico que corregimos (Ej. `sqrt(eps)` para antipodal).

## 2. Tareas Críticas Pendientes para el NUEVO CHAT
El próximo agente deberá leer este archivo y ejecutar inmediatamente lo siguiente en la carpeta V46:

1. **Reparar Frontera FFI (Python ↔ C++):** Abrir `polydim_motor_v46.py`, buscar las líneas de `ctypes` (`slerp_kernel.slerp.argtypes`) y añadir el séptimo argumento obligatorio `ctypes.c_size_t` (`scratch_size`), enviando `len(scratch_array)` en la llamada real. Si esto no se hace, el C++ hará segfault.
2. **Nuevos Tests Empíricos:** Añadir CHK_28 (ataque de scratch size en C++), CHK_29 (bitmap overflow) y CHK_30 (respeto de dtype) a `polydim_suite_v46.py`.
3. **Ejecutar Suite:** Correr `python polydim_suite_v46.py` y registrar los resultados en `CERTIFICADO_ESTRES_8H_V46.md`.
4. **Empaquetar V46:** Actualizar `codigo_consolidado_v46.txt` y crear `mañana.zip` para la entrega final de V46.

## Instrucciones para iniciar el nuevo chat:
Copia y pega este prompt al abrir la nueva conversación:
`Carga el archivo V46_CONTEXTO_HISTORICO.md, asume el protocolo Bulldog Critic, y ejecuta la tarea #1 (Reparar Frontera FFI en V46).`
