"""
POLYDIM V57 MEMORY & IPC PROTOCOL (PMTP SWMR)
- Single-Writer / Multi-Reader (SWMR) Protocol con Sequence Word en offset 0
- Dynamic Temporary Path via tempfile.gettempdir()
- Auto-cleanup seguro (unlink() solo en modo writer al salir)
- Silicon-aware LRU Arena Allocator con OrderedDict
"""

import os
import sys
import time
import struct
import mmap
import tempfile
import re
from collections import OrderedDict

import numpy as np
import jax
import jax.numpy as jnp

class PMTPConsistencyError(Exception):
    """Excepción para incoherencia o desincronización en la lectura Seqlock."""
    pass

class PMTPProtocolError(Exception):
    """Excepción para errores de cabecera o protocolo binario PMTP."""
    pass

class PMTPHeader:
    HEADER_SIZE = 64
    MAGIC = 0x504F4C5944494D34  # "POLYDIM4" en Little Endian
    PROTOCOL_VERSION = 57

    @staticmethod
    def pack(seq_word: int, dim: int, dtype_code: int = 1, payload_bytes: int = 0, timestamp: int = 0, generation: int = 1) -> bytes:
        if payload_bytes == 0:
            payload_bytes = dim * 4
        if timestamp == 0:
            timestamp = int(time.time_ns())
        reserved = b'\x00' * 16
        return struct.pack(
            "<QQIIIIQQ16s",
            seq_word,
            PMTPHeader.MAGIC,
            PMTPHeader.PROTOCOL_VERSION,
            dim,
            dtype_code,
            payload_bytes,
            timestamp,
            generation,
            reserved
        )

    @staticmethod
    def unpack(header_bytes: bytes) -> dict:
        if len(header_bytes) < PMTPHeader.HEADER_SIZE:
            raise PMTPProtocolError(f"Cabecera binaria incompleta: {len(header_bytes)} bytes < {PMTPHeader.HEADER_SIZE}")
        
        fields = struct.unpack("<QQIIIIQQ16s", header_bytes[:PMTPHeader.HEADER_SIZE])
        magic = fields[1]
        if magic != PMTPHeader.MAGIC:
            raise PMTPProtocolError(f"Magic Number binario inválido: {hex(magic)} != {hex(PMTPHeader.MAGIC)}")
        
        version = fields[2]
        if version != PMTPHeader.PROTOCOL_VERSION:
            raise PMTPProtocolError(f"Versión de protocolo no soportada: {version} != {PMTPHeader.PROTOCOL_VERSION}")
        
        return {
            "seq_word": fields[0],
            "magic": fields[1],
            "version": fields[2],
            "dim": fields[3],
            "dtype_code": fields[4],
            "payload_bytes": fields[5],
            "timestamp": fields[6],
            "generation": fields[7],
        }


class PMTPSharedMemoryBuffer:
    _DTYPE_MAP = {
        1: (np.float32, 4),
        2: (np.float64, 8)
    }

    def __init__(self, name: str, dim: int = 10000, mode: str = 'writer', dtype_code: int = 1):
        if not re.match(r'^[a-zA-Z0-9_\-]+$', name):
            raise ValueError(f"Nombre de canal SHM inválido: {name}")
        if dtype_code not in self._DTYPE_MAP:
            raise ValueError(f"dtype_code C-ABI no soportado: {dtype_code}")

        self.name = name
        self.dim = dim
        self.mode = mode.lower()
        self.dtype_code = dtype_code
        self.numpy_dtype, self.dtype_size = self._DTYPE_MAP[dtype_code]
        self.data_bytes = dim * self.dtype_size
        self.total_size = PMTPHeader.HEADER_SIZE + self.data_bytes
        self.fd = None
        self.buf = None
        self._seq = 0

        temp_dir = tempfile.gettempdir()
        self.path = os.path.join(temp_dir, f"{name}.shm")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        if self.mode == 'writer':
            if not os.path.exists(self.path) or os.path.getsize(self.path) != self.total_size:
                with open(self.path, "wb") as f_init:
                    f_init.write(b'\x00' * self.total_size)
                    f_init.flush()
            self.fd = open(self.path, "r+b")
            self._seq = 0
            self.buf = mmap.mmap(self.fd.fileno(), self.total_size)
            header_bytes = PMTPHeader.pack(seq_word=0, dim=dim, dtype_code=dtype_code, generation=1)
            self.buf[0:PMTPHeader.HEADER_SIZE] = header_bytes
        elif self.mode == 'reader':
            wait_time = 0.0
            while not os.path.exists(self.path) and wait_time < 5.0:
                time.sleep(0.01)
                wait_time += 0.01

            if not os.path.exists(self.path):
                raise PMTPProtocolError(f"Canal SHM de lectura no existe tras timeout: {self.path}")
                
            self.fd = open(self.path, "rb")
            self.buf = mmap.mmap(self.fd.fileno(), self.total_size, access=mmap.ACCESS_READ)
            hdr = PMTPHeader.unpack(bytes(self.buf[0:PMTPHeader.HEADER_SIZE]))
            if hdr["dim"] != self.dim:
                raise PMTPProtocolError(f"Dimensión de canal mismatch: esperada {self.dim}, canal tiene {hdr['dim']}")
        else:
            raise ValueError("Modo debe ser 'writer' o 'reader'")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self.mode == 'writer':
            self.unlink()
        return False

    def write_latent_bytes(self, raw_bytes: bytes):
        if self.mode != 'writer':
            raise PMTPProtocolError("Solo el propietario writer puede escribir en el canal PMTP SWMR")
            
        if len(raw_bytes) != self.data_bytes:
            raise ValueError(f"Tamaño de payload incorrecto: {len(raw_bytes)} vs {self.data_bytes}")

        self._seq += 1
        seq_start = self._seq * 2 + 1
        
        struct.pack_into("<Q", self.buf, 0, seq_start)
        self.buf.flush()  # Barrera de memoria (msync) - memory_order_acquire equivalente
        
        self.buf[PMTPHeader.HEADER_SIZE:self.total_size] = raw_bytes
        self.buf.flush()  # Barrera de memoria (msync) - memory_order_release equivalente
        
        seq_end = self._seq * 2
        struct.pack_into("<Q", self.buf, 0, seq_end)
        self.buf.flush()  # Commit final

    def read_snapshot(self, max_retries: int = 100) -> np.ndarray:
        for _ in range(max_retries):
            # Intento de barrera de lectura (membarrier/msync).
            # En Windows ACCESS_READ no soporta flush(), así que usamos try-except
            try:
                self.buf.flush()
            except Exception:
                pass
                
            seq1 = struct.unpack_from("<Q", self.buf, 0)[0]
            if seq1 % 2 != 0:
                time.sleep(0.0001)
                continue
                
            payload_raw = self.buf[PMTPHeader.HEADER_SIZE:self.total_size]
            
            try:
                self.buf.flush()
            except Exception:
                pass
                
            seq2 = struct.unpack_from("<Q", self.buf, 0)[0]
            
            if seq1 == seq2:
                np_dtype_str = '<f4' if self.dtype_code == 1 else '<f8'
                return np.frombuffer(payload_raw, dtype=np_dtype_str)
            time.sleep(0.0001)

        raise PMTPConsistencyError("Seqlock read timeout: Incoherencia por escrituras concurrentes intensivas")

    def close(self):
        if self.buf is not None:
            try:
                self.buf.close()
            except Exception:
                pass
            self.buf = None
            
        if self.fd is not None:
            try:
                self.fd.close()
            except Exception:
                pass
            self.fd = None

    def unlink(self):
        self.close()
        if os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except Exception:
                pass


class PolydimArenaAllocator:
    def __init__(self, capacity: int = 16, max_dim: int = 100000000):
        self.capacity = capacity
        self.max_dim = max_dim
        self.backend = str(jax.default_backend()).lower()
        self._pool = OrderedDict()

    def get_scratch_buffer(self, shape: tuple, dtype=jnp.float32) -> jnp.ndarray:
        key = (shape, dtype)
        size = 1
        for s in shape:
            size *= s
            
        if size > self.max_dim:
            return jnp.zeros(shape, dtype=dtype)

        if key in self._pool:
            self._pool.move_to_end(key)
            return self._pool[key]

        new_buf = jnp.zeros(shape, dtype=dtype)
        if len(self._pool) >= self.capacity:
            self._pool.popitem(last=False)

        self._pool[key] = new_buf
        return new_buf

    def clear(self):
        self._pool.clear()
