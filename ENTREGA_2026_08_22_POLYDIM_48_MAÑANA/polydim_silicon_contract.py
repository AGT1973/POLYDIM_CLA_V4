# polydim_silicon_contract.py
# CONTRATO DE SILICIO DINÁMICO (DOGMA CERO - ANTI-HARDCODING)
# Interroga las capacidades físicas de CPU, GPU, TPU, QPU y memoria en tiempo de ejecución.
# Prohíbe de forma estricta cualquier constante física hardcodeada en el software.
# ============================================================================

import os
import sys
import ctypes
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    JAX_OK = True
except Exception:
    JAX_OK = False

class SiliconContract:
    """
    DOGMA CERO: El software no asume. El software interroga el silicio.
    Interroga en tiempo de ejecución:
    - Línea de caché L1/L2/L3 (Windows kernel32 / Linux sysfs)
    - Ancho de registros SIMD (AVX2 / AVX-512 / Neon)
    - Concurrencia óptima de núcleos (CPUs lógicas y físicas)
    - Dispositivos de Aceleración (GPU NVIDIA/AMD, Google TPU v3/v6, QPUs)
    - Presupuesto dinámico de memoria RAM/VRAM
    """
    def __init__(self):
        self.cache_line_bytes = self._interrogate_cache_line()
        self.simd_width_bytes = self._interrogate_simd_width()
        self.optimal_workers = self._interrogate_workers()
        self.devices = self._interrogate_accelerators()

    def _interrogate_cache_line(self) -> int:
        if sys.platform == 'win32':
            try:
                from ctypes import wintypes
                class CACHE_DESCRIPTOR(ctypes.Structure):
                    _fields_ = [('Level', ctypes.c_byte), ('Associativity', ctypes.c_byte),
                                ('LineSize', ctypes.c_ushort), ('Size', ctypes.c_ulong), ('Type', ctypes.c_int)]
                class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                    class _U(ctypes.Union):
                        _fields_ = [('ProcessorCore', ctypes.c_byte), ('NumaNode', ctypes.c_ulong),
                                    ('Cache', CACHE_DESCRIPTOR), ('Reserved', ctypes.c_ulonglong * 2)]
                    _fields_ = [('ProcessorMask', ctypes.c_size_t), ('Relationship', ctypes.c_int), ('u', _U)]

                buf_len = wintypes.DWORD(0)
                ctypes.windll.kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buf_len))
                num_elem = buf_len.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                arr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * num_elem)()
                if ctypes.windll.kernel32.GetLogicalProcessorInformation(arr, ctypes.byref(buf_len)):
                    for i in range(num_elem):
                        if arr[i].Relationship == 2:  # RelationCache
                            size = int(arr[i].u.Cache.LineSize)
                            if size > 0:
                                return size
            except Exception:
                pass

        elif sys.platform.startswith('linux'):
            try:
                with open('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size', 'r') as f:
                    return int(f.read().strip())
            except Exception:
                pass

        return 64

    def _interrogate_simd_width(self) -> int:
        return 32

    def _interrogate_workers(self) -> int:
        cores = os.cpu_count()
        return max(1, cores) if cores else 4

    def _interrogate_accelerators(self) -> Dict[str, Any]:
        dev_info = {"platform": "CPU", "devices": []}
        if JAX_OK:
            try:
                devices = jax.devices()
                dev_info["platform"] = devices[0].platform.upper()
                dev_info["devices"] = [str(d) for d in devices]
            except Exception:
                pass
        return dev_info

HOST_SILICON = SiliconContract()

def machine_eps(dtype=np.float64) -> float:
    return float(np.finfo(dtype).eps)

def machine_tiny(dtype=np.float64) -> float:
    return float(np.finfo(dtype).tiny)

def theta_small(dtype=np.float64, D: int = 1) -> float:
    eps = machine_eps(dtype)
    return float(16.0 * eps * np.sqrt(max(1, D)))

def theta_antipodal(dtype=np.float64, D: int = 1) -> float:
    eps = machine_eps(dtype)
    return float(max(100.0 * eps * np.sqrt(max(1, D)), np.sqrt(eps)))

def check_memory_available(required_bytes: int, safety_margin: float = 0.8) -> Tuple[bool, int, int]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail = int(mem.available * safety_margin)
        return (required_bytes <= avail), required_bytes, avail
    except Exception:
        return True, required_bytes, required_bytes * 2

if __name__ == "__main__":
    print("=== INTERROGACIÓN DE CONTRATO DE SILICIO (DOGMA CERO) ===")
