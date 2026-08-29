"""
===============================================================================
POLYDIM V79 BULLDOG — PORTABILITY MODULE
===============================================================================
Soluciona errores:
  3.1  AVX-512 no existe en ARM64
  3.2  Rust sin feature flags para arch
  3.3  ctypes CDLL no funciona en WASM
  3.4  JAX no disponible en Windows nativamente
  3.5  TPU no soporta ctypes
  3.6  macOS ARM64 Rosetta 2 incompatibilidad
  3.7  Linux glibc version mismatch
  3.8  Windows DLL hell (MSVCRT vs UCRT)
  3.9  NumPy endianness en red
===============================================================================
"""

import os
import sys
import platform
import struct
import warnings
import numpy as np

# =============================================================================
# DETECCIÓN DE PLATAFORMA
# =============================================================================

class PlatformDetector:
    """Detecta plataforma y capacidades del sistema."""

    ARCH = platform.machine().lower()
    SYSTEM = platform.system().lower()
    IS_64BIT = sys.maxsize > 2**32

    @classmethod
    def is_x86_64(cls):
        return cls.ARCH in ('x86_64', 'amd64', 'i386', 'i686')

    @classmethod
    def is_arm64(cls):
        return cls.ARCH in ('arm64', 'aarch64', 'armv8')

    @classmethod
    def is_wasm(cls):
        return cls.ARCH in ('wasm32', 'wasm64')

    @classmethod
    def is_windows(cls):
        return cls.SYSTEM == 'windows'

    @classmethod
    def is_macos(cls):
        return cls.SYSTEM == 'darwin'

    @classmethod
    def is_linux(cls):
        return cls.SYSTEM == 'linux'

    @classmethod
    def has_avx512(cls):
        """Detecta AVX-512 vía CPUID (solo x86_64)."""
        if not cls.is_x86_64():
            return False
        try:
            # Verificar flags de CPU en Linux
            if cls.is_linux():
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    return 'avx512' in cpuinfo.lower()
            # macOS
            elif cls.is_macos():
                import subprocess
                result = subprocess.run(['sysctl', '-a'], capture_output=True, text=True)
                return 'avx512' in result.stdout.lower()
            # Windows
            elif cls.is_windows():
                import subprocess
                result = subprocess.run(['wmic', 'cpu', 'get', 'Caption'], 
                                       capture_output=True, text=True)
                return 'avx512' in result.stdout.lower()
        except Exception:
            pass
        return False

    @classmethod
    def has_neon(cls):
        """Detecta NEON (ARM64)."""
        if not cls.is_arm64():
            return False
        try:
            if cls.is_linux():
                with open('/proc/cpuinfo', 'r') as f:
                    return 'neon' in f.read().lower()
        except Exception:
            pass
        return True  # ARM64 siempre tiene NEON

    @classmethod
    def summary(cls):
        return {
            "arch": cls.ARCH,
            "system": cls.SYSTEM,
            "is_64bit": cls.IS_64BIT,
            "has_avx512": cls.has_avx512(),
            "has_neon": cls.has_neon(),
            "is_wasm": cls.is_wasm(),
        }


# =============================================================================
# 3.1 + 3.6: DISPATCH SIMD POR ARQUITECTURA
# =============================================================================

class SIMDDispatcher:
    """
    Selecciona el kernel nativo correcto según la arquitectura.
    Fallback a NumPy/JAX si no hay kernel nativo disponible.
    """

    _kernel_cache = {}

    @classmethod
    def get_householder_kernel(cls):
        """Retorna la mejor implementación disponible."""
        cache_key = (PlatformDetector.ARCH, PlatformDetector.SYSTEM)
        if cache_key in cls._kernel_cache:
            return cls._kernel_cache[cache_key]

        # 1. Intentar C++ AVX-512 (x86_64)
        if PlatformDetector.is_x86_64() and PlatformDetector.has_avx512():
            kernel = cls._try_load_cpp()
            if kernel:
                cls._kernel_cache[cache_key] = kernel
                return kernel

        # 2. Intentar C++ genérico (x86_64 sin AVX-512)
        if PlatformDetector.is_x86_64():
            kernel = cls._try_load_cpp()
            if kernel:
                cls._kernel_cache[cache_key] = kernel
                return kernel

        # 3. Intentar Rust (ARM64, x86_64)
        if PlatformDetector.is_arm64() or PlatformDetector.is_x86_64():
            kernel = cls._try_load_rust()
            if kernel:
                cls._kernel_cache[cache_key] = kernel
                return kernel

        # 4. Fallback a NumPy/JAX
        warnings.warn("Ningún kernel nativo disponible. Usando fallback NumPy.")
        cls._kernel_cache[cache_key] = "numpy"
        return "numpy"

    @classmethod
    def _try_load_cpp(cls):
        import ctypes
        candidates = []
        if PlatformDetector.is_windows():
            candidates.append("polydim_kernel_cpp_v79.dll")
        elif PlatformDetector.is_macos():
            candidates.append("libpolydim_kernel_cpp_v79.dylib")
        else:
            candidates.append("libpolydim_kernel_cpp_v79.so")

        for name in candidates:
            path = os.path.join(os.path.dirname(__file__), name)
            if os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    lib.polydim_householder_reflect_cpp.argtypes = [
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.c_uint64,
                        ctypes.c_uint64,
                    ]
                    lib.polydim_householder_reflect_cpp.restype = ctypes.c_int
                    return ("cpp", lib)
                except OSError:
                    pass
        return None

    @classmethod
    def _try_load_rust(cls):
        import ctypes
        candidates = []
        if PlatformDetector.is_windows():
            candidates.append("polydim_kernel_rust_v79.dll")
        elif PlatformDetector.is_macos():
            candidates.append("libpolydim_kernel_rust_v79.dylib")
        else:
            candidates.append("libpolydim_kernel_rust_v79.so")

        for name in candidates:
            path = os.path.join(os.path.dirname(__file__), name)
            if os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    lib.polydim_householder_reflect_rust.argtypes = [
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.c_uint64,
                        ctypes.c_uint64,
                    ]
                    lib.polydim_householder_reflect_rust.restype = ctypes.c_int
                    return ("rust", lib)
                except OSError:
                    pass
        return None


# =============================================================================
# 3.3: WASM FALLBACK
# =============================================================================

class WASMBackend:
    """
    Backend para WebAssembly (Pyodide).
    No usa ctypes ni DLLs dinámicas.
    """

    @staticmethod
    def is_wasm():
        return PlatformDetector.is_wasm()

    @staticmethod
    def householder_reflect(x, v):
        """Implementación pura NumPy para WASM."""
        v_max = np.max(np.abs(v), axis=-1, keepdims=True)
        safe_v_max = np.where(v_max < 1e-30, 1.0, v_max)
        v_norm = v / safe_v_max
        v_sq = np.sum(v_norm * v_norm, axis=-1, keepdims=True)
        is_zero = v_sq < 1e-30
        safe_v_sq = np.where(is_zero, 1.0, v_sq)
        factor = 2.0 * np.sum(x * v_norm, axis=-1, keepdims=True) / safe_v_sq
        reflect = x - factor * v_norm
        return np.where(is_zero, x, reflect)


# =============================================================================
# 3.4 + 3.5: BACKEND ABSTRACTO (JAX opcional, TPU-aware)
# =============================================================================

class Backend:
    """
    Backend abstracto que selecciona automáticamente entre:
      - JAX (GPU/TPU/CPU)
      - NumPy (CPU fallback)
      - WASM (Pyodide)
    """

    _jax_available = None
    _jax_backend = None

    @classmethod
    def has_jax(cls):
        if cls._jax_available is None:
            try:
                import jax
                cls._jax_available = True
                cls._jax_backend = jax.lib.xla_bridge.get_backend().platform
            except ImportError:
                cls._jax_available = False
                cls._jax_backend = None
        return cls._jax_available

    @classmethod
    def get_backend_name(cls):
        if PlatformDetector.is_wasm():
            return "wasm"
        if cls.has_jax():
            return f"jax_{cls._jax_backend}"
        return "numpy"

    @classmethod
    def get_array_module(cls):
        """Retorna el módulo de arrays apropiado (jnp o np)."""
        if PlatformDetector.is_wasm():
            return np
        if cls.has_jax():
            import jax.numpy as jnp
            return jnp
        return np

    @classmethod
    def is_tpu(cls):
        return cls.has_jax() and cls._jax_backend == "tpu"

    @classmethod
    def is_gpu(cls):
        return cls.has_jax() and cls._jax_backend in ("gpu", "cuda", "rocm")


# =============================================================================
# 3.7: GLIBC COMPATIBILITY CHECK
# =============================================================================

class GlibcChecker:
    """Verifica compatibilidad de glibc en Linux."""

    @staticmethod
    def get_glibc_version():
        if not PlatformDetector.is_linux():
            return None
        try:
            import subprocess
            result = subprocess.run(['ldd', '--version'], capture_output=True, text=True)
            version_line = result.stdout.split('\n')[0]
            # "ldd (GNU libc) 2.35"
            parts = version_line.split()
            return parts[-1] if parts else None
        except Exception:
            return None

    @staticmethod
    def check_compatibility(required="2.17"):
        """Verifica si la versión de glibc es suficiente."""
        current = GlibcChecker.get_glibc_version()
        if current is None:
            return True  # No es Linux, no aplica

        def parse(v):
            return tuple(int(x) for x in v.split('.'))

        if parse(current) < parse(required):
            warnings.warn(
                f"glibc {current} < {required}. "
                f"Compilar con manylinux o usar musl libc."
            )
            return False
        return True


# =============================================================================
# 3.8: WINDOWS RUNTIME CHECK
# =============================================================================

class WindowsRuntimeChecker:
    """Verifica runtime de Windows (UCRT vs MSVCRT)."""

    @staticmethod
    def check_runtime():
        if not PlatformDetector.is_windows():
            return True

        # Python 3.5+ usa UCRT
        if sys.version_info >= (3, 5):
            # Verificar que vcredist está instalado
            try:
                import ctypes
                ctypes.CDLL("vcruntime140.dll")
                return True
            except OSError:
                warnings.warn(
                    "Visual C++ Redistributable no encontrado. "
                    "Instalar vcredist_x64.exe"
                )
                return False
        return True


# =============================================================================
# 3.9: ENDIANNESS UTILITIES
# =============================================================================

class Endianness:
    """Manejo de endianness para protocolos de red."""

    NATIVE = sys.byteorder  # 'little' o 'big'
    NETWORK = 'big'

    @classmethod
    def to_little_endian(cls, arr):
        """Convierte array numpy a little-endian."""
        if arr.dtype.byteorder in ('=', cls.NATIVE):
            return arr.astype('<f8') if arr.dtype == np.float64 else arr
        elif arr.dtype.byteorder == '>':
            return arr.byteswap().newbyteorder('<')
        return arr

    @classmethod
    def to_native(cls, arr):
        """Convierte array little-endian a native."""
        if arr.dtype.byteorder == '<':
            return arr.byteswap().newbyteorder('=')
        return arr

    @classmethod
    def ensure_network(cls, arr):
        """Asegura little-endian para transmisión de red."""
        return cls.to_little_endian(arr)


# =============================================================================
# UNIVERSAL BINARY HELPER (macOS)
# =============================================================================

class UniversalBinary:
    """Helper para compilar y cargar universal binaries en macOS."""

    @staticmethod
    def get_dylib_name():
        if not PlatformDetector.is_macos():
            return None

        arch = PlatformDetector.ARCH
        if arch == 'arm64':
            return "polydim_kernel_arm64.dylib"
        elif arch == 'x86_64':
            return "polydim_kernel_x86_64.dylib"
        else:
            return "polydim_kernel_universal.dylib"

    @staticmethod
    def compile_universal(src, output):
        """Compila binary universal (x86_64 + arm64) en macOS."""
        if not PlatformDetector.is_macos():
            return False
        import subprocess
        cmd = [
            'clang++', '-O3', '-shared', '-fPIC',
            '-arch', 'x86_64', '-arch', 'arm64',
            src, '-o', output
        ]
        try:
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PlatformDetector",
    "SIMDDispatcher",
    "WASMBackend",
    "Backend",
    "GlibcChecker",
    "WindowsRuntimeChecker",
    "Endianness",
    "UniversalBinary",
]
