# 🎯 REQUERIMIENTOS Y ROADMAP DEL PROYECTO POLYDIM REPROCESO
## Abandono Definitivo del Sandbox Matemático y del Stack PyTorch / 100% JAX (XLA) + C++ + Rust

---

## 1. DIAGNÓSTICO DE ARQUITECTURA Y PROHIBICIÓN DE PYTORCH

### Por qué se descartó PyTorch (Lección Histórica):
1. **Garbage Collection Spill:** En tensores continuos $D \ge 10,000$, PyTorch sufre de *spills* en el recolector de basura de Python y fragmentación de memoria en asignaciones/liberaciones masivas.
2. **Latencia del GIL:** Las llamadas a PyTorch introducen la latencia del GIL de CPython ($> 2,000\ \mu\text{s}$), destruyendo el presupuesto de latencia Zero-Copy de PMTP ($< 20\ \mu\text{s}$).
3. **Falta de AOT Fiel al Silicio:** PyTorch no compila AOT a binario C++/XLA nativo sin overheads de framework.

### El Stack de Silicio de POLYDIM:
- **JAX (XLA AOT Compilado, `jax_enable_x64=True`):** Motor funcional estricto de alta velocidad para transformaciones tensoriales, SLERP batch y Stiefel.
- **C++ (AVX2 SIMD MSVC DLL):** Kernels escalares y vectoriales en C++ sin asignación de heap en *hot-loops*.
- **Rust (Lock-Free MPMC Ring Buffer):** Asignación por punteros crudos (`*mut f64`), `std`-only, cero aliasing UB en LLVM.
- **PMTP (`mmap` Zero-Copy):** Transferencia directa de memoria en C contiguo entre procesos a $\ge 12\text{ GB/s}$.

---

## 2. COMPONENTES REQUERIDOS EN `REPROCESO\CODIGO\`

| Módulo | Nombre | Propósito en el Runtime |
|---|---|---|
| **Capa 0** | `polydim_jax_engine.py` | `IaSpaceJaxEngine`: Motor JAX XLA 100% PyTorch-free AOT compilado en Float64. |
| **Capa 1** | `polydim_elevator.py` | `IaSpaceElevator`: Eleva representaciones al hiperespacio $S^{D-1}$ ($D \ge 4096$). |
| **Capa 2** | `polydim_skills.py` | `IaSpaceSkill`: Cargador y ejecutor de Skills latentes compiladas (operadores $T: S \to S$ en C++/Rust sobre `shm.buf` sin tokens LLM). |
| **Capa 2** | `polydim_agent.py` | `IaSpaceAgent`: Agente real en $S^{D-1}$ que sostiene la tupla $P = (S, G, T, O, C, \Pi)$ e evoluciona el estado latente. |
| **Capa 2** | `polydim_channel.py` | `PmtpLatentChannel`: Canal Zero-Copy `mmap` / Rust `PmtpRing` entre agentes a $\ge 12\text{ GB/s}$. |
| **Capa 3** | `polydim_terminal.py` | `TerminalCollapser`: Servidor MCP Terminal que proyecta $S \to M$ a los 4 niveles pedagógicos para el humano. |
| **Nativo** | `slerp_kernel_v47.cpp` + `lib_v47.rs` | Kernels DLL C++ (AVX2, Kahan) y Rust MPMC Ring Buffer (Raw Pointers) compilados vía `build.py`. |

---

## 3. ROADMAP PASO A PASO

- [x] **Fase 0: Estructuración y Constitución:** Crear `REPROCESO\`, `.agents\SOUL_AGY_ORCHESTRATOR.md`, `DOCUMENTACION\` y `PROYECTO\`.
- [x] **Fase 1: Motor Nivel de Silicio:** Compilar kernels DLL C++ y Rust en `REPROCESO\CODIGO\`.
- [x] **Fase 2: Motor JAX XLA PyTorch-Free:** Implementar `polydim_jax_engine.py` en Float64.
- [x] **Fase 3: Elevador y Canal PMTP:** Implementar `polydim_elevator.py` y `polydim_channel.py` para mover tensores reales en `mmap`.
- [x] **Fase 4: Skills Latentes en Memoria:** Implementar `polydim_skills.py` con operadores latentes ($SO(D)$, SLERP, Stiefel) sin tokens LLM.
- [x] **Fase 5: Agente del Espacio IA:** Implementar `polydim_agent.py` para que dos agentes mantengan estados $S \in S^{D-1}$ y cooperen en $ND$.
- [x] **Fase 6: Compuerta Terminal:** Implementar `polydim_terminal.py` para decodificar el estado final a los 4 Niveles Pedagógicos.

---
*Roadmap de Desarrollo POLYDIM REPROCESO (Stack JAX + C++ + Rust).*
