"""
===============================================================================
POLYDIM V79 BULLDOG — THREAD-SAFETY MODULE
===============================================================================
Soluciona errores:
  1.1  GIL + ctypes = serialización implícita (buffers compartidos)
  1.2  JAX + threading = deadlock potencial
  1.3  _seq_lock NO protege multiprocessing
  1.4  ctypes pointer lifetime (use-after-free)
  1.5  Rust + multithreading + panic (ya en kernel_rust_fixed.rs)
  1.6  JAX async dispatch + ctypes
  1.7  threading.Lock NO es fair (starvation)
  1.8  GIL + numpy array creation en hot path
===============================================================================
"""

import os
import sys
import threading
import queue
import concurrent.futures
import numpy as np
import ctypes
import warnings
from collections import deque
from functools import lru_cache

# =============================================================================
# 1.1 + 1.4: SAFE FFI CALL CON KEEPALIVE + THREAD-LOCAL BUFFERS
# =============================================================================

class ThreadSafeFFI:
    """Wrapper thread-safe para llamadas FFI con keepalive y buffers locales."""

    _local = threading.local()
    _buffer_pool = {}
    _pool_lock = threading.Lock()

    @classmethod
    def _get_thread_buffer(cls, shape):
        """Buffer thread-local para evitar data races en 'out'."""
        if not hasattr(cls._local, 'buffers'):
            cls._local.buffers = {}
        key = shape
        if key not in cls._local.buffers:
            cls._local.buffers[key] = np.empty(shape, dtype=np.float64)
        return cls._local.buffers[key]

    @classmethod
    def _get_pooled_buffer(cls, shape):
        """Buffer del pool global (thread-safe via lock)."""
        with cls._pool_lock:
            key = shape
            if key in cls._buffer_pool and cls._buffer_pool[key]:
                return cls._buffer_pool[key].pop()
        return np.empty(shape, dtype=np.float64)

    @classmethod
    def _return_pooled_buffer(cls, buf):
        """Devolver buffer al pool."""
        with cls._pool_lock:
            key = buf.shape
            if key not in cls._buffer_pool:
                cls._buffer_pool[key] = []
            if len(cls._buffer_pool[key]) < 10:  # Max 10 por shape
                cls._buffer_pool[key].append(buf)

    @classmethod
    def safe_call(cls, fn, x, v, out=None, use_pool=False):
        """
        Llamada FFI segura:
          - Asegura contigüidad sin copia si ya lo es.
          - Mantiene referencias vivas durante la llamada.
          - Usa buffer thread-local o pooled si out=None.
        """
        # 1.8: Evitar copia si ya es contiguo y float64
        if not x.flags['C_CONTIGUOUS'] or x.dtype != np.float64:
            x = np.ascontiguousarray(x, dtype=np.float64)
        if not v.flags['C_CONTIGUOUS'] or v.dtype != np.float64:
            v = np.ascontiguousarray(v, dtype=np.float64)

        if out is None:
            if use_pool:
                out = cls._get_pooled_buffer(x.shape)
                return_pooled = True
            else:
                out = cls._get_thread_buffer(x.shape)
                return_pooled = False
        else:
            return_pooled = False
            if not out.flags['C_CONTIGUOUS'] or out.dtype != np.float64:
                out = np.ascontiguousarray(out, dtype=np.float64)

        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        v_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # 1.4: Keepalive explícito para prevenir GC durante FFI
        _keepalive = (x, v, out)
        try:
            dim = x.shape[-1]
            batch = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
            ret = fn(x_ptr, v_ptr, out_ptr, 
                     ctypes.c_uint64(dim), ctypes.c_uint64(batch))
        finally:
            del _keepalive

        if return_pooled:
            cls._return_pooled_buffer(out)

        return ret, out


# =============================================================================
# 1.2 + 1.6: JAX SAFETY WRAPPER
# =============================================================================

class JAXSafetyWrapper:
    """Previene deadlock entre JAX async y ctypes síncrono."""

    @staticmethod
    def ensure_materialized(arr):
        """Fuerza materialización de array JAX antes de pasar a ctypes."""
        try:
            import jax
            import jax.numpy as jnp
            if isinstance(arr, jnp.ndarray):
                # 1.6: Block until ready + convert to numpy
                arr = np.array(arr.block_until_ready())
        except ImportError:
            pass
        return arr

    @staticmethod
    def safe_jit_boundary(fn):
        """Decorador: nunca llama ctypes desde dentro de @jax.jit."""
        def wrapper(*args, **kwargs):
            # Materializar TODOS los argumentos JAX
            args = tuple(JAXSafetyWrapper.ensure_materialized(a) for a in args)
            kwargs = {k: JAXSafetyWrapper.ensure_materialized(v) 
                      for k, v in kwargs.items()}
            return fn(*args, **kwargs)
        return wrapper


# =============================================================================
# 1.3: MULTIPROCESSING-SAFE SEQUENCE GENERATOR
# =============================================================================

class MPSeqGenerator:
    """
    Generador de seq seguro para multiprocessing.
    Usa timestamp + pid + tid para evitar colisiones entre procesos.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def next_seq(self):
        with self._lock:
            pid = os.getpid()
            tid = threading.current_thread().ident or 0
            ts = int(time.time() * 1e6)  # microsegundos
            # Componer: 32 bits TS | 16 bits PID | 16 bits TID
            return (ts << 32) | ((pid & 0xFFFF) << 16) | (tid & 0xFFFF)


# =============================================================================
# 1.7: FAIR LOCK / FIFO SEQUENCE GENERATOR
# =============================================================================

class FairSeqGenerator:
    """
    Generador de seq con cola FIFO.
    Garantiza que ningún thread se queda sin seq en alta contención.
    """

    def __init__(self):
        self._seq = 0
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def next_seq(self):
        fut = concurrent.futures.Future()
        self._queue.put(fut)
        return fut.result()

    def _worker(self):
        while True:
            try:
                fut = self._queue.get(timeout=1.0)
                self._seq += 1
                fut.set_result(self._seq)
            except queue.Empty:
                continue


# =============================================================================
# 1.8: ZERO-COPY BUFFER MANAGER
# =============================================================================

class ZeroCopyBufferManager:
    """
    Gestiona buffers numpy sin copia innecesaria.
    Reduce presión de GIL en hot path.
    """

    @staticmethod
    def ensure_native(arr):
        """Asegura que el array es C-contiguous y float64 SIN copiar si ya lo es."""
        if isinstance(arr, np.ndarray):
            if arr.dtype == np.float64 and arr.flags['C_CONTIGUOUS']:
                return arr  # Zero copy
        return np.ascontiguousarray(arr, dtype=np.float64)

    @staticmethod
    @lru_cache(maxsize=32)
    def get_cached_buffer(shape_tuple):
        """Cache de buffers por shape (útil para shapes repetidos en training)."""
        return np.empty(shape_tuple, dtype=np.float64)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ThreadSafeFFI",
    "JAXSafetyWrapper",
    "MPSeqGenerator",
    "FairSeqGenerator",
    "ZeroCopyBufferManager",
]
