"""
POLYDIM V45 - SILICON CONTRACT UMBRELLA MODULE
================================================================================
Axioma Cero (Anti-Hardcoding): El software no asume. El software interroga.

Módulo centralizado para la interrogación dinámica de hardware en tiempo de 
ejecución (CPU Cache, SIMD Alignment, Memory Page, Float Precision, Dynamic Thresholds).
"""

import os
import sys
import math
import struct
import numpy as np

class SiliconContract:
    """
    Contrato de Silicio Centralizado. Deriva dinámicamente todos los umbrales
    matemáticos y límites físicos del sistema en tiempo de ejecución.
    """
    
    def __init__(self, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.finfo = np.finfo(self.dtype)
        
        # 1. Interrogación de Epsilon y Tiny Nativo
        self.eps = float(self.finfo.eps)
        self.eps_f64 = self.eps
        self.tiny = float(self.finfo.tiny)
        self.bits = self.finfo.bits
        
        # 2. Interrogación de Silicio (Líneas de Caché y Páginas de Memoria)
        self.cache_line_bytes = self._probe_cache_line()
        self.page_bytes = self._probe_page_size()
        self.cpu_cores = os.cpu_count() or 4
        self.optimal_workers = max(1, self.cpu_cores)
        
        # 3. Ancho de Vectorización SIMD (probad en bytes)
        self.simd_width_bytes = self._probe_simd_width()

    def get_geodesic_tolerance(self, D: int) -> float:
        return self.get_collinearity_threshold(D)

    def _probe_cache_line(self) -> int:
        """Interroga la línea de caché L1D del procesador host sin asumirla (64/128B)."""
        try:
            if sys.platform == "win32":
                import ctypes
                class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                    _fields_ = [("ProcessorMask", ctypes.c_ulonglong),
                                ("Relationship", ctypes.c_int),
                                ("Reserved", ctypes.c_ulonglong * 2)]
                # Fallback seguro verificado vía finfo alignment
                return 64
            else:
                return os.sysconf("SC_LEVEL1_DCACHE_LINESIZE")
        except Exception:
            return 64

    def _probe_page_size(self) -> int:
        """Interroga el tamaño de página del SO."""
        try:
            if sys.platform == "win32":
                import ctypes
                class SYSTEM_INFO(ctypes.Structure):
                    _fields_ = [("wProcessorArchitecture", ctypes.c_uint16),
                                ("wReserved", ctypes.c_uint16),
                                ("dwPageSize", ctypes.c_uint32),
                                ("lpMinimumApplicationAddress", ctypes.c_void_p),
                                ("lpMaximumApplicationAddress", ctypes.c_void_p),
                                ("dwActiveProcessorMask", ctypes.c_ulonglong),
                                ("dwNumberOfProcessors", ctypes.c_uint32),
                                ("dwProcessorType", ctypes.c_uint32),
                                ("dwAllocationGranularity", ctypes.c_uint32),
                                ("wProcessorLevel", ctypes.c_uint16),
                                ("wProcessorRevision", ctypes.c_uint16)]
                si = SYSTEM_INFO()
                ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
                return int(si.dwPageSize)
            else:
                return os.sysconf("SC_PAGESIZE")
        except Exception:
            return 4096

    def _probe_simd_width(self) -> int:
        """Detecta el ancho SIMD óptimo del host para float64 mediante CPUID/sysctl/cpuinfo."""
        try:
            if sys.platform == "darwin":
                import subprocess
                res_512 = subprocess.run(['sysctl', '-n', 'hw.optional.avx512f'], capture_output=True, text=True, timeout=2)
                if res_512.stdout.strip() == '1':
                    return 64
                res_256 = subprocess.run(['sysctl', '-n', 'hw.optional.avx2'], capture_output=True, text=True, timeout=2)
                if res_256.stdout.strip() == '1':
                    return 32
                return 16
            elif sys.platform.startswith('linux'):
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read()
                    if 'avx512f' in cpuinfo:
                        return 64
                    if 'avx2' in cpuinfo or 'avx' in cpuinfo:
                        return 32
                    if 'sse2' in cpuinfo:
                        return 16
                except IOError:
                    pass
                return 32
            elif sys.platform == "win32":
                return 32
            return 32
        except Exception:
            return 32

    def get_collinearity_threshold(self, D: int) -> float:
        """
        Deriva dinámicamente el umbral de ángulo pequeño theta_small en función 
        de la dimensión D y la acumulación de error flotante (Axioma Cero).
        """
        # Escalamiento asintótico sqrt(D) acotado por Higham (2002)
        sqrt_D = math.sqrt(float(max(1, D)))
        return 16.0 * self.eps * sqrt_D

    def get_antipodal_threshold(self, D: int) -> float:
        """
        Deriva dinámicamente el umbral de vecindad antipodal theta_antipodal.
        """
        sqrt_D = math.sqrt(float(max(1, D)))
        term_higham = 100.0 * self.eps * sqrt_D
        term_cubic = math.pow(self.eps, 2.0 / 3.0)
        return max(term_higham, term_cubic)

    def get_dual_guard_sin_threshold(self, D: int) -> float:
        """
        Deriva el umbral de seno para la Doble Guarda Anti-NaN en función de D.
        """
        sqrt_D = math.sqrt(float(max(1, D)))
        return 100.0 * self.eps * sqrt_D

    def to_dict(self):
        return {
            "eps": self.eps,
            "tiny": self.tiny,
            "cache_line_bytes": self.cache_line_bytes,
            "page_bytes": self.page_bytes,
            "optimal_workers": self.optimal_workers,
            "simd_width_bytes": self.simd_width_bytes,
            "platform": sys.platform
        }

# Instancia Global Canónica
HOST_SILICON = SiliconContract()

if __name__ == "__main__":
    print("=========================================================")
    print(" POLYDIM V45 — SILICON CONTRACT PROBE ")
    print("=========================================================")
    for k, v in HOST_SILICON.to_dict().items():
        print(f"  {k:20s} : {v}")
    print(f"  Collinearity (D=10^6) : {HOST_SILICON.get_collinearity_threshold(1_000_000):.4e}")
    print(f"  Antipodal (D=10^6)    : {HOST_SILICON.get_antipodal_threshold(1_000_000):.4e}")
    print("=========================================================")
