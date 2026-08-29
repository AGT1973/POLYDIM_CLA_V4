"""
===============================================================================
POLYDIM V79 BULLDOG — MONOLITO CORREGIDO (LEY ARIEL COMPLIANT)
===============================================================================
Fixes aplicados:
  - PMTP: struct 112B consistente, HMAC sobre 112B, time.time(), ventana
    deslizante, AEAD sobre payload, seq persistente, anti-replay real.
  - NativeFFIBridge: FFI REAL con ctypes, manejo de shapes arbitrarias,
    keepalive de referencias, fallback JAX limpio.
  - GeodesicKernels: exp_map con doble proyección + jnp.sinc,
    log_map con arccos estable + NaN-safe, log_map_newton con custom_vjp.
  - CliffordRotors: cholesky_qr3 con shift suave + jnp.matmul + status real,
    rename apply_spherical_geodesic_rotation.
  - JAX: x64 opcional, NO side-effect global.
===============================================================================
"""

import os
import sys
import time
import ctypes
import struct
import warnings
import threading
import hmac
import hashlib
import secrets
from pathlib import Path
from collections import OrderedDict
from functools import partial

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# JAX es opcional — NO side-effect global
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # Fallback
    lax = None

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Clave PMTP: archivo privado en home, NO env var
_PMTP_KEY_FILE = Path.home() / ".polydim" / "pmtp.key"

def _load_pmtp_key():
    if _PMTP_KEY_FILE.exists():
        key = _PMTP_KEY_FILE.read_bytes()
        if len(key) == 32:
            return key
    key = secrets.token_bytes(32)
    _PMTP_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PMTP_KEY_FILE.write_bytes(key)
    _PMTP_KEY_FILE.chmod(0o600)
    return key

PMTP_NET_KEY = _load_pmtp_key()

# =============================================================================
# NATIVE FFI BRIDGE — IMPLEMENTACIÓN REAL
# =============================================================================

class NativeFFIBridge:
    _cpp_lib = None
    _rust_lib = None
    _is_initialized = False
    _lock = threading.Lock()
    _local = threading.local()

    @staticmethod
    def initialize():
        with NativeFFIBridge._lock:
            if NativeFFIBridge._is_initialized:
                return
            curr_dir = os.path.dirname(os.path.abspath(__file__))

            candidates_cpp = [
                os.path.join(curr_dir, "polydim_kernel_cpp_v79.dll"),
                os.path.join(curr_dir, "libpolydim_kernel_cpp_v79.so"),
                os.path.join(curr_dir, "libpolydim_kernel_cpp_v79.dylib"),
            ]
            candidates_rust = [
                os.path.join(curr_dir, "polydim_kernel_rust_v79.dll"),
                os.path.join(curr_dir, "libpolydim_kernel_rust_v79.so"),
                os.path.join(curr_dir, "libpolydim_kernel_rust_v79.dylib"),
            ]

            for cpp_dll in candidates_cpp:
                if os.path.exists(cpp_dll):
                    try:
                        NativeFFIBridge._cpp_lib = ctypes.CDLL(cpp_dll)
                        fn = NativeFFIBridge._cpp_lib.polydim_householder_reflect_cpp
                        fn.argtypes = [
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.c_uint64,
                            ctypes.c_uint64,
                        ]
                        fn.restype = ctypes.c_int
                        break
                    except OSError:
                        pass

            for rust_dll in candidates_rust:
                if os.path.exists(rust_dll):
                    try:
                        NativeFFIBridge._rust_lib = ctypes.CDLL(rust_dll)
                        fn = NativeFFIBridge._rust_lib.polydim_householder_reflect_rust
                        fn.argtypes = [
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.c_uint64,
                            ctypes.c_uint64,
                        ]
                        fn.restype = ctypes.c_int
                        break
                    except OSError:
                        pass

            if NativeFFIBridge._cpp_lib is None and NativeFFIBridge._rust_lib is None:
                warnings.warn("Ninguna DLL nativa encontrada. Usando fallback JAX/NumPy.")

            NativeFFIBridge._is_initialized = True

    @staticmethod
    def _get_thread_local_buffer(shape):
        if not hasattr(NativeFFIBridge._local, 'buf') or NativeFFIBridge._local.buf.shape != shape:
            NativeFFIBridge._local.buf = np.empty(shape, dtype=np.float64)
        return NativeFFIBridge._local.buf

    @staticmethod
    def householder_reflect(x, v, backend="cpp"):
        NativeFFIBridge.initialize()
        lib = NativeFFIBridge._cpp_lib if backend == "cpp" else NativeFFIBridge._rust_lib

        if lib is None:
            return NativeFFIBridge._householder_jax_fallback(x, v)

        # Asegurar contigüidad y dtype
        x_np = np.ascontiguousarray(x, dtype=np.float64)
        v_np = np.ascontiguousarray(v, dtype=np.float64)
        out_np = np.empty_like(x_np)

        # Manejo de shapes arbitrarias
        orig_shape = x_np.shape
        dim = orig_shape[-1]
        batch = int(np.prod(orig_shape[:-1])) if x_np.ndim > 1 else 1

        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        v_ptr = v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        fn = lib.polydim_householder_reflect_cpp if backend == "cpp" else lib.polydim_householder_reflect_rust

        # Keepalive para prevenir GC durante FFI
        _keepalive = (x_np, v_np, out_np)
        ret = fn(x_ptr, v_ptr, out_ptr, ctypes.c_uint64(dim), ctypes.c_uint64(batch))
        del _keepalive

        if ret != 0:
            raise RuntimeError(f"Householder native error code: {ret}")

        return out_np.reshape(orig_shape) if HAS_JAX else out_np.reshape(orig_shape)

    @staticmethod
    def _householder_jax_fallback(x, v):
        """Fallback JAX/NumPy puro, sin side-effects globales."""
        xp = jnp if HAS_JAX else np
        v_max = xp.max(xp.abs(v), axis=-1, keepdims=True)
        safe_v_max = xp.where(v_max < 1e-30, 1.0, v_max)
        v_norm = v / safe_v_max
        v_sq = xp.sum(v_norm * v_norm, axis=-1, keepdims=True)
        is_zero = v_sq < 1e-30
        safe_v_sq = xp.where(is_zero, 1.0, v_sq)
        factor = 2.0 * xp.sum(x * v_norm, axis=-1, keepdims=True) / safe_v_sq
        reflect = x - factor * v_norm
        return xp.where(is_zero, x, reflect)


# =============================================================================
# GEODESIC KERNELS — CORREGIDOS
# =============================================================================

class GeodesicKernels:
    _EPS = 1e-12

    @staticmethod
    def _get_eps(x):
        if HAS_JAX:
            return jnp.finfo(x.dtype).eps
        return np.finfo(x.dtype).eps

    @staticmethod
    def safe_dot(a, b, keepdims=False):
        xp = jnp if HAS_JAX else np
        return xp.sum(a * b, axis=-1, keepdims=keepdims)

    @staticmethod
    def safe_norm(x, keepdims=False, eps=None):
        xp = jnp if HAS_JAX else np
        if eps is None:
            eps = GeodesicKernels._EPS
        # Regularización suave: diferenciable en TODO R^n
        sq_norm = xp.sum(x * x, axis=-1, keepdims=True)
        norm = xp.sqrt(sq_norm + eps * eps)
        return norm if keepdims else xp.squeeze(norm, axis=-1)

    @staticmethod
    def exp_map(x, v):
        xp = jnp if HAS_JAX else np
        eps = GeodesicKernels._get_eps(x)
        x_norm = GeodesicKernels.safe_norm(x, keepdims=True, eps=eps)
        x_u = x / xp.maximum(x_norm, eps)

        # Proyección ortogonal EXACTA (doble Gram-Schmidt)
        v_tan = v - GeodesicKernels.safe_dot(v, x_u, keepdims=True) * x_u
        v_tan = v_tan - GeodesicKernels.safe_dot(v_tan, x_u, keepdims=True) * x_u

        v_norm = GeodesicKernels.safe_norm(v_tan, keepdims=True, eps=eps)

        # jnp.sinc es robusto para v_norm → 0 (sin branching)
        sinc = xp.sinc(v_norm / xp.pi)  # sin(v)/v exacto en C
        exp = xp.cos(v_norm) * x_u + sinc * v_tan

        # Normalización final para garantizar estar en la variedad
        exp_norm = GeodesicKernels.safe_norm(exp, keepdims=True, eps=eps)
        return exp / xp.maximum(exp_norm, eps)

    @staticmethod
    def log_map(x, y, tau_geom=1e-12):
        xp = jnp if HAS_JAX else np
        eps = GeodesicKernels._get_eps(x)
        xn = GeodesicKernels.safe_norm(x, keepdims=True, eps=eps)
        yn = GeodesicKernels.safe_norm(y, keepdims=True, eps=eps)
        xu = x / xp.maximum(xn, eps)
        yu = y / xp.maximum(yn, eps)

        c = GeodesicKernels.safe_dot(xu, yu, keepdims=True)
        c = xp.clip(c, -1.0, 1.0)
        theta = xp.arccos(c)

        # Proyección ortogonal estable
        y_perp = yu - c * xu
        y_perp_norm = GeodesicKernels.safe_norm(y_perp, keepdims=True, eps=eps)

        safe_perp = xp.where(y_perp_norm < eps, 1.0, y_perp_norm)
        u_tangent = y_perp / safe_perp
        log_normal = theta * u_tangent

        is_zero = (xn < eps) | (yn < eps)
        is_antipodal = (c < -1.0 + tau_geom) & (y_perp_norm < tau_geom)
        is_identity = theta < tau_geom

        # NaN-safe: usar zeros en lugar de NaN para no poisonar gradientes
        res = xp.where(is_antipodal, xp.zeros_like(log_normal), log_normal)
        res = xp.where(is_identity, xp.zeros_like(res), res)
        res = xp.where(is_zero, xp.zeros_like(res), res)

        # Marcar singularidades con una máscara separada (NO en el valor)
        return res  # El caller maneja is_antipodal si es necesario

    @staticmethod
    def log_map_newton(x, y, max_iter=5):
        if not HAS_JAX:
            # En NumPy puro, usar log_map directo
            return GeodesicKernels.log_map(x, y)

        # JAX: custom_vjp para diferenciabilidad
        return _log_map_newton_jax(x, y, max_iter)


# =============================================================================
# LOG_MAP_NEWTON CON CUSTOM_VJP (JAX ONLY)
# =============================================================================

if HAS_JAX:
    @jax.custom_vjp
    def _log_map_newton_jax(x, y, max_iter=5):
        eps = jnp.finfo(x.dtype).eps
        xu = x / jnp.maximum(GeodesicKernels.safe_norm(x, keepdims=True, eps=eps), eps)
        yu = y / jnp.maximum(GeodesicKernels.safe_norm(y, keepdims=True, eps=eps), eps)

        def cond_fun(state):
            i, v, err = state
            valid = jnp.isfinite(err) & jnp.all(jnp.isfinite(v))
            return jnp.logical_and(jnp.logical_and(i < max_iter, err > 1e-12), valid)

        def body_fun(state):
            i, v, _ = state
            y_approx = GeodesicKernels.exp_map(xu, v)
            res = GeodesicKernels.log_map(y_approx, yu)
            # Reemplazar NaN por zeros para evitar poison
            res = jnp.where(jnp.isnan(res), jnp.zeros_like(res), res)
            c = GeodesicKernels.safe_dot(y_approx, xu, keepdims=True)
            safe_denom = jnp.where(jnp.abs(1.0 + c) < eps, 1.0, 1.0 + c)
            trans_res = res - (GeodesicKernels.safe_dot(res, y_approx + xu, keepdims=True) / safe_denom) * (y_approx + xu)
            v_new = v + trans_res
            v_new = v_new - GeodesicKernels.safe_dot(v_new, xu, keepdims=True) * xu
            err = jnp.max(GeodesicKernels.safe_norm(res, keepdims=False, eps=eps))
            return i + 1, v_new, err

        v_init = GeodesicKernels.log_map(xu, yu)
        v_init = jnp.where(jnp.isnan(v_init), jnp.zeros_like(v_init), v_init)
        err_init = 1.0
        _, v_final, _ = lax.while_loop(cond_fun, body_fun, (0, v_init, err_init))
        return v_final

    def _log_map_newton_fwd(x, y, max_iter):
        v = _log_map_newton_jax(x, y, max_iter)
        return v, (x, y)

    def _log_map_newton_bwd(residuals, g):
        x, y = residuals
        # Backward: usar log_map (cerrado, diferenciable) como proxy
        _, f_vjp = jax.vjp(lambda yy: GeodesicKernels.log_map(x, yy), y)
        return (None, f_vjp(g)[0], None)

    _log_map_newton_jax.defvjp(_log_map_newton_fwd, _log_map_newton_bwd)
else:
    _log_map_newton_jax = None


# =============================================================================
# CLIFFORD ROTORS — CORREGIDOS
# =============================================================================

class CliffordRotors:
    @staticmethod
    def cholesky_qr3(W, max_iter=5):
        xp = jnp if HAS_JAX else np
        eps = GeodesicKernels._get_eps(W)
        K = W.shape[-1]
        I = xp.eye(K, dtype=W.dtype)
        Q = W
        for _ in range(max_iter):
            # Usar jnp.matmul en lugar de einsum (batched BLAS)
            G = xp.matmul(Q.swapaxes(-1, -2), Q)
            # Shift suave SIN maximum (diferenciable)
            trace_G = xp.trace(G, axis1=-2, axis2=-1)[..., None, None]
            shift = (eps + 10.0 * eps * trace_G / K) * I
            G_reg = G + shift
            L = xp.linalg.cholesky(G_reg)
            L_invT = jax.lax.linalg.triangular_solve(L.swapaxes(-1, -2), I, lower=False) if HAS_JAX else \
                     np.linalg.solve(L.swapaxes(-1, -2).T, I).T  # Fallback numpy
            Q = xp.matmul(Q, L_invT)

        # Detectar rank deficiency real
        orth_err = xp.max(xp.abs(xp.matmul(Q.swapaxes(-1, -2), Q) - I))
        status = xp.where(orth_err > 1e-10, 1, 0)
        return Q, status

    @staticmethod
    def apply_spherical_geodesic_rotation(x, U, V, alpha=1.0):
        """
        Aplica una rotación geodésica en S^{n-1} generada por dos vectores tangentes.
        NOTA: Esto NO es un rotor Clifford (que actuaría por conjugación en el álgebra).
        Es un mapa exponencial riemanniano en la esfera.
        """
        U_tan = U - GeodesicKernels.safe_dot(U, x, keepdims=True) * x
        V_tan = V - GeodesicKernels.safe_dot(V, x, keepdims=True) * x
        v_vel = alpha * (U_tan + V_tan)
        return GeodesicKernels.exp_map(x, v_vel)


# =============================================================================
# PMTP NETWORK LAYER — REESCRITURA SEGURA
# =============================================================================

PMTP_MAGIC = b"PMTP"
# 112 bytes exactos (sin dummy MAC)
PMTP_HEADER_FMT = "<4s B B B B Q 16s 16s Q Q d Q Q Q Q Q"
PMTP_HEADER_SIZE = struct.calcsize(PMTP_HEADER_FMT)  # 112
PMTP_PACKET_SIZE = PMTP_HEADER_SIZE + 16  # 128

class SlidingWindowSet:
    """Ventana deslizante con TTL y tamaño máximo."""
    def __init__(self, max_size=100000, ttl=60.0):
        self._data = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()

    def add(self, key):
        with self._lock:
            now = time.monotonic()
            # Evict expired
            while self._data and now - next(iter(self._data.values())) > self._ttl:
                self._data.popitem(last=False)
            # Evict oldest if full
            if len(self._data) >= self._max_size:
                self._data.popitem(last=False)
            self._data[key] = now

    def __contains__(self, key):
        with self._lock:
            self.add(key)  # Trigger cleanup
            return key in self._data


class PMTPNetworkLayer:
    def __init__(self, node_id: str):
        self.node_id = node_id.encode("utf-8").ljust(16, b"\x00")[:16]
        self._seq_file = Path.home() / ".polydim" / f"seq_{node_id}.wal"
        self._seq_num = self._load_seq()
        self._boot_id = os.urandom(16)  # 128 bits para evitar collisiones
        self._seq_lock = threading.Lock()
        self._seen_seqs = SlidingWindowSet(max_size=100000, ttl=60.0)
        self._seen_boot_ids = SlidingWindowSet(max_size=1000, ttl=3600.0)

    def _load_seq(self):
        try:
            return int(self._seq_file.read_text())
        except (FileNotFoundError, ValueError):
            return 0

    def _save_seq(self):
        self._seq_file.parent.mkdir(parents=True, exist_ok=True)
        self._seq_file.write_text(str(self._seq_num))

    def pack_tensor_header(self, tensor_shape: tuple, payload_bytes: int, receiver_id: bytes) -> bytes:
        if payload_bytes > 100 * 1024 * 1024:
            raise ValueError("Anti-DoS: Payload > 100MB")

        with self._seq_lock:
            self._seq_num += 1
            seq = self._seq_num
            self._save_seq()

        shape_len = len(tensor_shape)
        if shape_len > 5:
            raise ValueError("PMTP Error: Máximo 5 dimensiones.")

        padded_shape = list(tensor_shape) + [0] * (5 - shape_len)
        ts = time.time()  # UTC absoluto, comparable entre nodos

        header_raw = struct.pack(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, 44, 1, 1, shape_len, payload_bytes,
            self.node_id, receiver_id.ljust(16, b"\x00")[:16], seq,
            int.from_bytes(self._boot_id, "little"), ts,
            *padded_shape
        )

        # HMAC sobre los 112B del header (consistente con unpack)
        mac = hmac.new(PMTP_NET_KEY, header_raw, hashlib.sha256).digest()[:16]
        return header_raw + mac

    def unpack_and_verify(self, packet: bytes, expected_receiver: bytes) -> tuple:
        # Copia inmutable para prevenir race conditions
        packet = bytes(packet)

        if len(packet) < PMTP_PACKET_SIZE:
            raise ValueError(f"Longitud de paquete PMTP inválida: {len(packet)}")

        header_raw = packet[:PMTP_HEADER_SIZE]
        mac_received = packet[PMTP_HEADER_SIZE:PMTP_PACKET_SIZE]
        payload = packet[PMTP_PACKET_SIZE:]

        # HMAC PRIMERO (timing attack mitigation)
        mac_calc = hmac.new(PMTP_NET_KEY, header_raw, hashlib.sha256).digest()[:16]
        mac_ok = hmac.compare_digest(mac_calc, mac_received)

        unpacked = struct.unpack(PMTP_HEADER_FMT, header_raw)
        magic, ver, tipo, res, ndim, p_bytes, sender, rec, seq, boot, ts = unpacked[:11]

        # Verificaciones TODAS antes de reportar error (no branching temprano)
        len_ok = len(packet) >= PMTP_PACKET_SIZE
        magic_ok2 = magic == PMTP_MAGIC
        ver_ok = ver == 44
        time_ok = abs(time.time() - ts) < 60.0
        size_ok = p_bytes <= 100 * 1024 * 1024
        receiver_ok = rec == expected_receiver.ljust(16, b"\x00")[:16]

        if not all([mac_ok, len_ok, magic_ok2, ver_ok, time_ok, size_ok, receiver_ok]):
            raise ValueError("PMTP verification failed")

        # Anti-replay
        sender_clean = sender  # NO strip zeros para evitar impersonación
        boot_bytes = struct.pack("Q", boot)

        if boot_bytes not in self._seen_boot_ids:
            self._seen_boot_ids.add(boot_bytes)

        if seq in self._seen_seqs:
            raise ValueError("Replay: seq duplicado")
        self._seen_seqs.add(seq)

        shape = unpacked[11:11 + ndim]
        return sender_clean, p_bytes, shape, payload

    def pack_secure(self, tensor_shape, payload, receiver_id):
        """Pack con AEAD sobre payload completo."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise RuntimeError("Instalar 'cryptography' para AEAD: pip install cryptography")

        header = self.pack_tensor_header(tensor_shape, len(payload), receiver_id)
        nonce = os.urandom(12)
        aesgcm = AESGCM(PMTP_NET_KEY)
        ciphertext = aesgcm.encrypt(nonce, payload, header)
        return header + nonce + ciphertext

    def unpack_secure(self, packet, expected_receiver):
        """Unpack con AEAD sobre payload completo."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise RuntimeError("Instalar 'cryptography' para AEAD")

        if len(packet) < PMTP_PACKET_SIZE + 12:
            raise ValueError("Paquete AEAD demasiado corto")

        header = packet[:PMTP_PACKET_SIZE]
        nonce = packet[PMTP_PACKET_SIZE:PMTP_PACKET_SIZE + 12]
        ciphertext = packet[PMTP_PACKET_SIZE + 12:]

        sender, p_bytes, shape, _ = self.unpack_and_verify(header, expected_receiver)

        aesgcm = AESGCM(PMTP_NET_KEY)
        payload = aesgcm.decrypt(nonce, ciphertext, header)

        if len(payload) != p_bytes:
            raise ValueError("Payload size mismatch")

        return sender, payload, shape


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NativeFFIBridge",
    "GeodesicKernels",
    "CliffordRotors",
    "PMTPNetworkLayer",
    "HAS_JAX",
]
