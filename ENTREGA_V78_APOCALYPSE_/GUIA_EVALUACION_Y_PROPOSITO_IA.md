# GUÍA DE EVALUACIÓN Y PROPÓSITO PARA MODELOS DE IA (LEY ARIEL)

**Entrega:** POLYDIM V78 APOCALYPSE  
**Directorio Autorizado:** `E:\POLYDIM_EINSOF\ENTREGA_V78_APOCALYPSE_\`  
**Fecha:** 28 de Agosto de 2026  

---

## 📌 1. PROPÓSITO DEL SISTEMA Y OBJETIVO DE AUDITORÍA

El propósito de POLYDIM V78 es proporcionar un motor de **Computabilidad Geométrica Riemannian y Comunicación Tensorial Nativa (PMTP v44)** en ultra-alta dimensión ($D \ge 10,000$).

### Regla de Auditoría Activa (Ley Ariel - Anti-Tautología Operativa)
Está estrictamente prohibido que cualquier subagente o IA evaluadora realice una auditoría pasiva ("el código se ve estándar"). Toda evaluación debe asumir que el sistema contiene vectores de fallo latentes y debe auditar la consistencia matemática de los siguientes 4 invariantes:

1. **Invarianza de Norma:** $\| \text{Exp}_x(v) \|_2 = 1.0 \pm 10^{-12}$.
2. **Reversibilidad Geodésica:** $\text{Exp}_x(\text{Log}_x(y)) = y$.
3. **No-Singularidad de Cayley-SMW:** $\det(I_{2k} + \frac{\alpha}{2} V^T U) > 0 \quad \forall \alpha \in \mathbb{R}$.
4. **Seguridad FFI:** Detección de aliasing por rango de bytes y comprobación de contigüidad C.

---

## 📖 2. ORDEN CANÓNICO DE LECTURA PARA IAs EVALUADORAS

Para evitar la pérdida de contexto o truncado sintáctico, cualquier IA debe procesar los archivos de la entrega en este orden estricto:

1. [`GUIA_EVALUACION_Y_PROPOSITO_IA.md`](file:///E:/POLYDIM_EINSOF/ENTREGA_V78_APOCALYPSE_/GUIA_EVALUACION_Y_PROPOSITO_IA.md) (Este documento)
2. [`WHITEBOOK_POLYDIM_V78.md`](file:///E:/POLYDIM_EINSOF/ENTREGA_V78_APOCALYPSE_/WHITEBOOK_POLYDIM_V78.md) (Fundamentos teóricos)
3. [`kernel_cpp_v78.cpp.txt`](file:///E:/POLYDIM_EINSOF/ENTREGA_V78_APOCALYPSE_/kernel_cpp_v78.cpp.txt) (Fuente C++ nativo)
4. [`kernel_rust_v78.rs.txt`](file:///E:/POLYDIM_EINSOF/ENTREGA_V78_APOCALYPSE_/kernel_rust_v78.rs.txt) (Fuente Rust nativo)
5. [`polydim_v78_monolito.py`](file:///E:/POLYDIM_EINSOF/ENTREGA_V78_APOCALYPSE_/polydim_v78_monolito.py) (Monolito Python ejecutable)
6. [`PROTOCOLO_PRUEBAS_POR_SERVIDOR.md`](file:///E:/POLYDIM_EINSOF/ENTREGA_V78_APOCALYPSE_/PROTOCOLO_PRUEBAS_POR_SERVIDOR.md) (Protocolo de benchmarks)

---

## 🧪 3. MATRIZ DE EVALUACIÓN RECOMENDADA POR MODELO

| Modelo Evaluador | Enfoque de Pruebas Recomendado |
| :--- | :--- |
| **Kimi / OpenRouter** | Auditoría de contratos de memoria FFI, aliasing de punteros y barreras atómicas en C++/Rust. |
| **Claude 3.7 / 3.5 Sonnet** | Verificación de demostraciones formales LaTeX y análisis asintótico de complejidad Cayley-SMW. |
| **DeepSeek-R1 / Reasoner** | Verificación de estabilidad flotante en Shifted CholeskyQR3 y prueba de gradientes `custom_vjp`. |
| **Gemini 2.5 Pro / Flash** | Benchmarks de ejecución paralela XLA y escalabilidad en GPUs multi-nodo. |
