# DOCUMENTO MAESTRO SOTA: ARQUITECTURA ZERO-COPY JAX-FFI Y MITIGACIÓN TOCTOU (MODO BULLDOG)

**ESTADO:** CRÍTICO / VETO ACTIVO A ENFOQUES LEGACY.
**OBJETIVO:** Erradicar overhead de serialización en interoperabilidad Python/Nativo y destruir condiciones de carrera en compilación en caliente.

---

## 1. DESTRUCCIÓN DEL ENFOQUE CTYPES / PYBIND11 (EL CUELLO DE BOTELLA 1D)

El uso de `ctypes` o `pybind11` para pasar matrices ND desde Python hacia C++/Rust es un crimen asintótico. Cada llamada serializa metadatos, cruza el GIL y, en el peor de los casos, induce copias ocultas si los strides no son perfectos. Esto es inaceptable para $D \ge 10,000$.

### LA SOLUCIÓN SOTA: `jax.ffi.ffi_call` (XLA Custom Calls)

El nuevo estándar abandona Python como intermediario en tiempo de ejecución. `jax.ffi` inyecta el puntero de la función C/Rust **directamente en el grafo XLA**. Durante la ejecución (JIT), el runtime de XLA invoca el binario nativo pasando los punteros crudos (Device Pointers) sin que Python se entere.

#### Arquitectura de Memoria (Zero-Copy Puro)
1. **Firma C/Rust:** La función nativa ya no toma `PyObject*` ni estructuras de `ctypes`. Toma `XlaCustomCallStatus*` y arreglos opacos que representan los buffers de entrada/salida.
2. **Cero Marshalling:** XLA pasa el puntero físico a la memoria de GPU/TPU/CPU. No hay conversión de tipos. Los tensores son arreglos unidimensionales en memoria, y el kernel nativo reconstruye los strides.
3. **Asintótica:** 
   - Overhead de llamada: $O(1)$ absoluto.
   - Copia de memoria: $0$ bytes.
   - Costo de transferencia: Limitado únicamente por el bus PCIe/NVLink, no por el runtime de software.

**Ejemplo de firma nativa obligatoria (Rust):**
```rust
#[no_mangle]
pub extern "C" fn my_kernel_ffi(
    stream: *mut core::ffi::c_void,
    buffers: *mut *mut core::ffi::c_void,
    opaque: *const core::ffi::c_char,
    opaque_len: usize,
    status: *mut XlaCustomCallStatus,
) {
    // 1. Cero copias. 'buffers' contiene los punteros crudos a las entradas y salidas de XLA.
    // 2. Operación vectorial masiva ND directamente sobre la memoria física.
}
```
*Si tu código C++/Rust copia un solo byte de esos buffers para procesarlo o iterarlo fuera de SIMD, tu arquitectura está rota.*

---

## 2. ERRADICACIÓN DE TOCTOU EN COMPILACIÓN EN CALIENTE (HOT-RELOAD)

**El Problema:** La compilación Just-In-Time (JIT) de C++/Rust y su carga dinámica (`.dll`/`.so`) está plagada de condiciones de carrera (Time-of-Check to Time-of-Use).
*Falsa solución complaciente:* Usar `threading.Lock()` o `multiprocessing.Lock()` en Python. **VETO.** Eso no detiene a otro proceso independiente (ej. un worker the pytest, o un Sabueso) que intente leer el `.dll` mientras `rustc`/`g++` lo está escribiendo, resultando en un `Segfault` por binario truncado o un error `ETXTBSY` / `Sharing Violation`.

### EL PROTOCOLO ESTRICTO DE SISTEMA (FS ATOMICITY)

La única fuente de verdad concurrente confiable es el Sistema de Archivos (POSIX/NTFS) mediante **Renombrados Atómicos**.
En NTFS (Windows), existe un agravante extremo: un `.dll` cargado en memoria está bloqueado por el kernel del SO. **No puedes sobreescribirlo ni borrarlo.**

#### Flujo SOTA (Inmutabilidad por Hashing + UUID)

Para compilar y cargar un kernel dinámicamente sin colisiones, debes tratar los binarios generados como artefactos inmutables.

1. **Fingerprinting (Hash):** Calcula el SHA-256 del código fuente exacto (y flags de compilación). 
   - Objetivo: `kernel_<HASH>.dll`.
2. **Time-of-Check Seguro:**
   - ¿Existe `kernel_<HASH>.dll`? Si existe, otra hebra/proceso ya lo compiló. Cárgalo y finaliza.
3. **Aislamiento en Compilación:**
   - Genera un `UUID` criptográficamente seguro (v4).
   - Compila el artefacto hacia un archivo temporal único: `kernel_<HASH>_<UUID>.tmp`.
   - *Nota: Nadie más en el multiverso conoce este UUID, por lo que la escritura es 100% thread-safe y process-safe.*
4. **Renombrado Atómico (El Puente Crítico):**
   - Ejecuta un rename del OS: `os.replace()` en Python (que invoca `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)` en Windows, o `rename()` en POSIX).
   - Transacción atómica: `kernel_<HASH>_<UUID>.tmp` $\rightarrow$ `kernel_<HASH>.dll`.
   - **Mitigación de colisión en Rename:** Si dos procesos terminan de compilar el mismo hash al mismo tiempo, ambos intentarán el rename. Uno ganará. El segundo puede fallar (en NTFS, porque el primer hilo ya cargó el DLL y lo bloqueó, o en POSIX simplemente lo sobreescribirá limpiamente porque el FS lo garantiza).
   - **Manejo del fallo:** Si el rename falla, capturamos la excepción y verificamos si `kernel_<HASH>.dll` ya existe. Si existe, borramos nuestro `.tmp` sobrante silenciosamente y cargamos la versión que el otro hilo instauró.
5. **Carga Inmutable:** Carga `kernel_<HASH>.dll` vía `ctypes.CDLL` o la API de XLA. Como el nombre está anclado al hash del contenido, cualquier modificación al código fuente en caliente producirá un NUEVO hash, saltando la memoria bloqueada del DLL anterior y creando un nuevo artefacto aislado.

### REGLA DE ORO DE LOS 10,000D
Nunca confíes en el estado del software por encima del estado físico de los transistores y el disco duro. El silicio no miente. La sincronización basada en memoria de aplicación (Mutex) es una alucinación cuando compites por descriptores de archivos de sistema operativo. Renombrado atómico o Segfault. Elegir.
