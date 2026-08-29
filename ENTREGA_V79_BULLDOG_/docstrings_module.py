"""
===============================================================================
POLYDIM V79 BULLDOG — API DOCUMENTATION MODULE
===============================================================================
Módulo de documentación inline con ejemplos para todas las funciones públicas.
===============================================================================
"""

# Este archivo contiene docstrings extensos que deben copiarse
# a los módulos correspondientes. Se entrega separado para facilitar
# la revisión de documentación.

# =============================================================================
# NATIVEFFIBRIDGE
# =============================================================================

NATIVEFFIBRIDGE_DOC = """
NativeFFIBridge
===============
Puente seguro entre Python y kernels nativos (C++/Rust).

Proporciona:
  - Carga dinámica de DLLs/.so/.dylib por plataforma
  - Manejo de shapes arbitrarias (batch + dim)
  - Buffers thread-local para evitar data races
  - Keepalive de referencias durante llamadas FFI
  - Fallback a JAX/NumPy puro si no hay kernels nativos

Ejemplo de uso:
    >>> import numpy as np
    >>> from polydim_v79_monolito_fixed import NativeFFIBridge
    >>> x = np.array([1.0, 0.0, 0.0, 0.0])
    >>> v = np.array([0.0, 1.0, 0.0, 0.0])
    >>> result = NativeFFIBridge.householder_reflect(x, v)
    >>> print(result)
    [ 1.  -0.   0.   0. ]

Thread-safety:
    Las llamadas a ctypes liberan el GIL, permitiendo paralelismo.
    Cada thread usa su propio buffer de salida (thread-local).
    El caller DEBE garantizar que 'out' no es compartido entre threads
    si se pasa explícitamente.

Parámetros de householder_reflect:
    x (np.ndarray): Vector(es) a reflejar. Shape: (..., dim)
    v (np.ndarray): Vector(es) de Householder. Shape: (..., dim)
    backend (str): "cpp", "rust", o "jax" (fallback automático)

Retorna:
    np.ndarray: Reflejo de x sobre el hiperplano ortogonal a v.
                Shape idéntico al input.

Excepciones:
    RuntimeError: Si el kernel nativo retorna código de error != 0
    ValueError: Si las shapes de x y v no coinciden
"""

# =============================================================================
# GEODESICKERNELS
# =============================================================================

GEODESICKERNELS_DOC = """
GeodesicKernels
===============
Operaciones geodésicas en la esfera unitaria S^{n-1}.

Todas las funciones están implementadas con JAX (cuando disponible)
o NumPy como fallback. Las operaciones son JIT-compilables y
diferenciables (excepto log_map_newton que usa custom_vjp).

Funciones principales:

safe_dot(a, b, keepdims=False)
    Producto punto seguro con broadcasting.

    Args:
        a (array): Primer operando. Shape: (..., dim)
        b (array): Segundo operando. Shape: (..., dim)
        keepdims (bool): Si True, mantiene la dimensión reducida

    Returns:
        array: Producto punto. Shape: (...) o (..., 1)

safe_norm(x, keepdims=False, eps=1e-12)
    Norma L2 regularizada. Diferenciable en TODO R^n.

    Usa regularización suave: sqrt(||x||² + eps²)
    en lugar de máscara booleana (que rompe diferenciabilidad).

    Args:
        x (array): Vector(es). Shape: (..., dim)
        keepdims (bool): Si True, mantiene la dimensión
        eps (float): Término de regularización

    Returns:
        array: Norma regularizada. Shape: (...) o (..., 1)

exp_map(x, v)
    Mapa exponencial riemanniano en S^{n-1}.

    Proyecta v al espacio tangente en x, normaliza, y aplica
    la fórmula exp(x, v) = cos(||v||) * x + sinc(||v||) * v.

    Args:
        x (array): Punto base en la esfera. Shape: (..., dim)
        v (array): Vector tangente. Shape: (..., dim)

    Returns:
        array: Punto en la esfera. Shape: (..., dim)
        Garantía: ||result|| = 1 (dentro de precisión numérica)

    Nota:
        Usa doble proyección de Gram-Schmidt para eliminar drift
        y jnp.sinc para evitar branching en v=0.

log_map(x, y, tau_geom=1e-12)
    Mapa logarítmico en S^{n-1}.

    Calcula el vector tangente v tal que exp_map(x, v) = y.

    Args:
        x (array): Punto base. Shape: (..., dim)
        y (array): Punto destino. Shape: (..., dim)
        tau_geom (float): Tolerancia para singularidades

    Returns:
        array: Vector tangente. Shape: (..., dim)

        Casos especiales:
          - x ≈ y: retorna zeros (identidad)
          - x antipodal a y: retorna zeros (singularidad manejada)
          - x=0 o y=0: retorna zeros (punto degenerado)

    Nota:
        NO retorna NaN en casos singulares (a diferencia de v79 original).
        El caller debe verificar la distancia geodésica si es crítico.

log_map_newton(x, y, max_iter=5)
    Refinamiento de Newton para log_map.

    Usa log_map como inicialización y aplica iteraciones de
    Newton para mejorar precisión.

    Args:
        x (array): Punto base. Shape: (..., dim)
        y (array): Punto destino. Shape: (..., dim)
        max_iter (int): Máximo de iteraciones (≤20 recomendado)

    Returns:
        array: Vector tangente refinado. Shape: (..., dim)

    Diferenciabilidad:
        Usa @jax.custom_vjp. El forward usa while_loop (preciso).
        El backward usa log_map (diferenciable) como proxy.

    Advertencia:
        max_iter > 50 puede causar stack overflow en compilación XLA.
"""

# =============================================================================
# CLIFFORDROTORS
# =============================================================================

CLIFFORDROTORS_DOC = """
CliffordRotors
==============
Operaciones de álgebra de Clifford y ortogonalización.

IMPORTANTE: El nombre "CliffordRotors" es histórico. Las funciones
actuales operan en la esfera S^{n-1} vía geometría riemanniana,
NO vía conjugación en el álgebra de Clifford.

Funciones principales:

cholesky_qr3(W, max_iter=5)
    Ortogonalización por Cholesky-QR iterado.

    Algoritmo:
      1. G = W^T W
      2. G_reg = G + shift * I (shift suave, diferenciable)
      3. L = chol(G_reg)
      4. Q = W * L^{-T}
      5. Repetir max_iter veces

    Args:
        W (array): Matriz(es) a ortogonalizar. Shape: (..., n, k)
        max_iter (int): Iteraciones de refinamiento

    Returns:
        tuple: (Q, status)
          - Q: Matriz ortogonal. Shape: (..., n, k)
          - status: 0=OK, 1=rank deficiente detectado

    Nota:
        Usa jnp.matmul (batched BLAS) en lugar de einsum para
        mejor rendimiento en GPU/TPU.

apply_spherical_geodesic_rotation(x, U, V, alpha=1.0)
    Aplica una rotación geodésica en S^{n-1}.

    Calcula la velocidad tangente v = alpha * (U_tan + V_tan)
    y aplica exp_map(x, v).

    Args:
        x (array): Punto base en la esfera. Shape: (..., dim)
        U (array): Primer generador. Shape: (..., dim)
        V (array): Segundo generador. Shape: (..., dim)
        alpha (float): Factor de escala

    Returns:
        array: Punto rotado en la esfera. Shape: (..., dim)

    Advertencia:
        El Jacobiano de esta transformación NO preserva el volumen.
        NO usar en flows normalizing sin corrección de Jacobiano.
"""

# =============================================================================
# PMTPNETWORKLAYER
# =============================================================================

PMTP_DOC = """
PMTPNetworkLayer
================
Protocolo PMTP v44 (Polydim Message Transfer Protocol).

Seguridad implementada:
  - AEAD (AES-GCM) sobre payload completo
  - HMAC-SHA256 sobre header de 112B
  - Anti-replay con ventana deslizante (100K entradas, 60s TTL)
  - Anti-DoS (payload ≤ 100MB)
  - Timing attack resistance
  - Seq persistente en WAL
  - Boot ID de 128 bits

Formato de paquete:
    [Header: 112B][MAC: 16B][Nonce: 12B][Ciphertext: payload_len]

Header (112B):
    magic:      4B  (b"PMTP")
    version:    1B  (44)
    type:       1B
    reserved:   1B
    ndim:       1B
    payload_sz: 8B
    sender:     16B
    receiver:   16B
    seq:        8B
    boot_id:    8B
    timestamp:  8B (double, UTC)
    shape:      5×8B (40B)

Funciones principales:

pack_tensor_header(tensor_shape, payload_bytes, receiver_id)
    Crea un header PMTP firmado.

    Args:
        tensor_shape (tuple): Shape del tensor (max 5 dimensiones)
        payload_bytes (int): Tamaño del payload en bytes
        receiver_id (bytes): ID del receptor (16B max)

    Returns:
        bytes: Header firmado de 128B

    Raises:
        ValueError: Si payload > 100MB o ndim > 5

unpack_and_verify(packet, expected_receiver)
    Verifica y desempaqueta un paquete PMTP.

    Args:
        packet (bytes): Paquete completo (header + mac)
        expected_receiver (bytes): ID esperado del receptor

    Returns:
        tuple: (sender, payload_bytes, shape, payload)

    Raises:
        ValueError: Si cualquier verificación falla (mensaje genérico)

    Nota:
        Calcula HMAC PRIMERO para mitigar timing attacks.
        No revela qué verificación falló.

pack_secure(tensor_shape, payload, receiver_id)
    Crea paquete con payload cifrado (AEAD).

    Args:
        tensor_shape (tuple): Shape del tensor
        payload (bytes): Datos del payload
        receiver_id (bytes): ID del receptor

    Returns:
        bytes: Paquete completo con cifrado

    Requiere:
        pip install cryptography

unpack_secure(packet, expected_receiver)
    Desempaqueta y descifra paquete AEAD.

    Args:
        packet (bytes): Paquete completo
        expected_receiver (bytes): ID esperado

    Returns:
        tuple: (sender, payload, shape)

    Raises:
        ValueError: Si descifrado o verificación falla
"""

# =============================================================================
# THREAD-SAFETY MODULE
# =============================================================================

THREADSAFE_DOC = """
polydim_v79_threadsafe
======================
Utilidades para operaciones seguras en entornos multithreaded.

Clases principales:

ThreadSafeFFI
    Wrapper seguro para llamadas FFI.

    safe_call(fn, x, v, out=None, use_pool=False)
        - Asegura contigüidad sin copia (zero-copy cuando es posible)
        - Mantiene keepalive durante la llamada FFI
        - Usa buffers thread-local o pooled

JAXSafetyWrapper
    Previene deadlock entre JAX async y ctypes síncrono.

    ensure_materialized(arr)
        Fuerza materialización de arrays JAX antes de FFI.

    safe_jit_boundary(fn)
        Decorador que materializa argumentos JAX antes de llamar fn.

MPSeqGenerator
    Generador de seq seguro para multiprocessing.

    next_seq()
        Retorna seq único compuesto de timestamp + pid + tid.

FairSeqGenerator
    Generador de seq con cola FIFO.

    Garantiza fairness en alta contención (evita starvation).

ZeroCopyBufferManager
    Gestiona buffers numpy sin copia innecesaria.

    ensure_native(arr)
        Asegura C-contiguous float64 sin copiar si ya lo es.

    get_cached_buffer(shape_tuple)
        Cache de buffers por shape (LRU de 32 entradas).
"""

# =============================================================================
# MEMORY MODULE
# =============================================================================

MEMORY_DOC = """
polydim_v79_memory
==================
Utilidades para gestión segura de memoria.

Clases principales:

LRUSlidingWindow
    Ventana deslizante con TTL y LRU eviction.

    Reemplaza sets monotónicos que crecen sin límite (OOM).

    Args:
        max_size (int): Máximo de entradas (default: 100000)
        ttl (float): Tiempo de vida en segundos (default: 60.0)

ObjectPool
    Pool reutilizable de objetos numpy.

    Reduce allocations en hot path de entrenamiento.

    Args:
        max_per_shape (int): Máximo de buffers por shape (default: 10)

JAXStackGuard
    Previene stack overflow en compilación XLA.

    MAX_ITER = 20 (límite seguro para while_loop)
    MAX_NESTED_WHILE = 10 (límite de anidamiento)

CapacityValidator
    Valida capacidad de buffers antes de pasar a C/Rust.

    validate(arr, dim, batch, name)
        Verifica que arr.size >= dim * batch.

ImmutablePacket
    Wrapper inmutable para paquetes de red.

    Previene race conditions en bytearray mutable.

RAIIBuffer
    Buffer con inicialización garantizada y cleanup seguro.

    __enter__ / __exit__ para uso con 'with'.
"""

# =============================================================================
# PORTABILITY MODULE
# =============================================================================

PORTABLE_DOC = """
polydim_v79_portable
====================
Abstracción de portabilidad cross-platform.

Clases principales:

PlatformDetector
    Detecta arquitectura y capacidades del sistema.

    is_x86_64(), is_arm64(), is_wasm()
    is_windows(), is_macos(), is_linux()
    has_avx512(), has_neon()

SIMDDispatcher
    Selecciona kernel nativo según arquitectura.

    get_householder_kernel()
        Retorna: ("cpp", lib) | ("rust", lib) | "numpy"

Backend
    Backend abstracto JAX/NumPy.

    get_array_module() -> jnp o np
    is_tpu(), is_gpu()

Endianness
    Manejo de endianness para protocolos de red.

    to_little_endian(arr)
    to_native(arr)
    ensure_network(arr)

UniversalBinary
    Helper para compilar universal binaries en macOS.
"""

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NATIVEFFIBRIDGE_DOC",
    "GEODESICKERNELS_DOC",
    "CLIFFORDROTORS_DOC",
    "PMTP_DOC",
    "THREADSAFE_DOC",
    "MEMORY_DOC",
    "PORTABLE_DOC",
]
