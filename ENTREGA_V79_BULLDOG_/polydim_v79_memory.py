"""
===============================================================================
POLYDIM V79 BULLDOG — MEMORY SAFETY MODULE
===============================================================================
Soluciona errores:
  2.1  Buffer overflow por batch*dim overflow
  2.2  Use-after-free en ctypes callback
  2.3  Memory leak en PMTPNetworkLayer (sets monotónicos)
  2.4  Stack overflow en JAX (log_map_newton compilación)
  2.5  Uninitialized memory en C++ (branch no-tomada)
  2.6  Double-free en Rust (from_raw_parts + drop)
  2.7  Heap fragmentation en JAX (muchos small arrays)
  2.8  Stack buffer overflow en PMTP (header mutable)
===============================================================================
"""

import os
import time
import threading
import numpy as np
from collections import OrderedDict
from pathlib import Path

# =============================================================================
# 2.3: LRU SLIDING WINDOW (reemplaza sets monotónicos)
# =============================================================================

class LRUSlidingWindow:
    """
    Ventana deslizante con TTL y LRU eviction.
    Reemplaza los sets monotónicos que crecen sin límite.
    """

    def __init__(self, max_size=100000, ttl=60.0):
        self._data = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def add(self, key):
        with self._lock:
            now = time.monotonic()
            # Evict expired (front of OrderedDict)
            expired = []
            for k, ts in self._data.items():
                if now - ts > self._ttl:
                    expired.append(k)
                else:
                    break  # OrderedDict mantiene orden temporal
            for k in expired:
                del self._data[k]

            # Evict oldest if full (LRU)
            while len(self._data) >= self._max_size:
                self._data.popitem(last=False)

            self._data[key] = now

    def __contains__(self, key):
        with self._lock:
            now = time.monotonic()
            # Cleanup lazy
            expired = [k for k, ts in self._data.items() if now - ts > self._ttl]
            for k in expired:
                del self._data[k]

            if key in self._data:
                # Touch para LRU
                self._data.move_to_end(key)
                self._hits += 1
                return True
            self._misses += 1
            return False

    def stats(self):
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# =============================================================================
# 2.7: BUFFER POOL (previene heap fragmentation)
# =============================================================================

class ObjectPool:
    """
    Pool reutilizable de objetos numpy.
    Reduce allocations en hot path de entrenamiento.
    """

    def __init__(self, max_per_shape=10):
        self._pools = {}
        self._max = max_per_shape
        self._lock = threading.Lock()
        self._allocations = 0
        self._reuses = 0

    def acquire(self, shape, dtype=np.float64):
        key = (shape, dtype)
        with self._lock:
            if key in self._pools and self._pools[key]:
                self._reuses += 1
                return self._pools[key].pop()
        self._allocations += 1
        return np.empty(shape, dtype=dtype)

    def release(self, buf):
        if buf is None:
            return
        key = (buf.shape, buf.dtype)
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            if len(self._pools[key]) < self._max:
                self._pools[key].append(buf)

    def stats(self):
        with self._lock:
            total = self._allocations + self._reuses
            reuse_rate = self._reuses / total if total > 0 else 0.0
            return {
                "allocations": self._allocations,
                "reuses": self._reuses,
                "reuse_rate": reuse_rate,
                "pools": {str(k): len(v) for k, v in self._pools.items()},
            }


# =============================================================================
# 2.4: STACK GUARD PARA JAX (previene stack overflow en compilación)
# =============================================================================

class JAXStackGuard:
    """
    Previene stack overflow en compilación XLA por while_loop anidado.
    Limita max_iter y detecta grafos profundos.
    """

    MAX_ITER = 20
    MAX_NESTED_WHILE = 10

    @classmethod
    def validate_max_iter(cls, max_iter):
        if max_iter > cls.MAX_ITER:
            raise ValueError(
                f"max_iter={max_iter} excede límite de compilación segura "
                f"({cls.MAX_ITER}). Reducir o usar log_map directo."
            )
        return max_iter

    @classmethod
    def validate_nesting(cls, depth):
        if depth > cls.MAX_NESTED_WHILE:
            raise ValueError(
                f"Anidamiento de while_loop={depth} excede límite "
                f"({cls.MAX_NESTED_WHILE}). Inlinear o refactorizar."
            )


# =============================================================================
# 2.1: CAPACITY VALIDATOR (previene buffer overflow)
# =============================================================================

class CapacityValidator:
    """
    Valida que los buffers tienen capacidad suficiente
    antes de pasar a C/Rust.
    """

    @staticmethod
    def validate(arr, dim, batch, name="buffer"):
        required = dim * batch
        actual = arr.size
        if actual < required:
            raise ValueError(
                f"{name}: capacidad insuficiente. "
                f"Requiere {required} elementos, tiene {actual}."
            )
        return True

    @staticmethod
    def validate_shape(arr, expected_shape, name="buffer"):
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name}: shape mismatch. "
                f"Esperado {expected_shape}, obtenido {arr.shape}."
            )
        return True


# =============================================================================
# 2.8: IMMUTABLE PACKET (previene race condition en bytearray)
# =============================================================================

class ImmutablePacket:
    """
    Wrapper inmutable para paquetes de red.
    Previene modificación concurrente entre len() y slicing.
    """

    __slots__ = ('_data', '_hash')

    def __init__(self, data):
        if isinstance(data, (bytes, bytearray)):
            self._data = bytes(data)
        else:
            raise TypeError("Data debe ser bytes o bytearray")
        self._hash = hash(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, ImmutablePacket):
            return self._data == other._data
        return self._data == other

    def raw(self):
        return self._data

    def hex(self):
        return self._data.hex()


# =============================================================================
# 2.5 + 2.6: RAII BUFFER (previene uninitialized memory y double-free)
# =============================================================================

class RAIIBuffer:
    """
    Buffer con inicialización garantizada y cleanup seguro.
    """

    def __init__(self, shape, dtype=np.float64, zero_init=True):
        self._buf = np.empty(shape, dtype=dtype)
        if zero_init:
            self._buf.fill(0)
        self._disposed = False

    @property
    def data(self):
        if self._disposed:
            raise RuntimeError("Buffer ya fue liberado")
        return self._buf

    def dispose(self):
        """Liberación explícita (no es necesario en Python, pero documenta intento)."""
        self._disposed = True
        self._buf = None

    def __enter__(self):
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()
        return False

    def __del__(self):
        if not self._disposed:
            self.dispose()


# =============================================================================
# MEMORY PROFILER (debugging)
# =============================================================================

class MemoryProfiler:
    """Profiler simple de uso de memoria para debugging."""

    @staticmethod
    def numpy_memory_usage():
        """Estima memoria usada por arrays numpy activos."""
        import gc
        total = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray):
                    total += obj.nbytes
            except ReferenceError:
                pass
        return total

    @staticmethod
    def format_bytes(n):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if n < 1024.0:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} PB"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LRUSlidingWindow",
    "ObjectPool",
    "JAXStackGuard",
    "CapacityValidator",
    "ImmutablePacket",
    "RAIIBuffer",
    "MemoryProfiler",
]
