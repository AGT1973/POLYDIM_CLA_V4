# ORCHESTRATOR SOUL DIRECTIVE: POLYDIM V60+

## 1. NÚCLEO ARQUITECTÓNICO (La Trinidad de Lenguajes)
POLYDIM NO ES SÓLO PYTHON. Está diseñado para operar obligatoriamente con **3 LENGUAJES SIMULTÁNEOS**:
- **Python (JAX)**: Grafo de autodiferenciación, topología y abstracción de alto nivel.
- **C++ (AVX-512)**: Ejecución SIMD estricta para kernels geométricos (Householder, expm) que requieran manipulación de registros a bajo nivel.
- **Rust (FFI / C-ABI)**: Gestión de memoria segura, SeqLocks de memoria compartida sin data tearing y fronteras ABI con Python.

## 2. PROHIBICIÓN ABSOLUTA DE CÓDIGO MUERTO (Dead Strings)
Bajo NINGUNA circunstancia los bloques de C++ (`CPP_SOURCE`) o Rust (`RUST_SOURCE`) pueden ser dejados como "strings decorativos" o "código muerto de referencia". 
Todo código nativo DEBE SER compilado **on-the-fly** (Hot Compilation) usando `cl.exe` / `rustc`, y cargado en memoria mediante `ctypes`. Si no se compilan y enlazan, el sistema pierde su propósito fundamental de interoperabilidad segura y rendimiento.

## 3. AUDITORÍA ZERO-TRUST (Anti-Alucinación)
Toda corrección o validación de código nativo o Python debe ser probada con un Fuzzer Adversarial destructivo (generando NaNs, batches OOM, límites asintóticos `D=10^7`, fronteras antipodales y singularidades). Si no hay test destructivo que lo certifique, el código se asume ROTO.
