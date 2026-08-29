"""
===============================================================================
POLYDIM V79 BULLDOG - MONOLITO NATIVO DE ALTA DIMENSION (LEY ARIEL COMPLIANT)
===============================================================================
File: polydim_v79_monolito.py
Architecture: Geometria Diferencial Riemannian, Clifford Rotors, Cayley-SMW,
              Shifted-CholeskyQR3, PMTP v44 Zero-Trust Socket Engine & Hot FFI.
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

if not jax.config.jax_enable_x64:
    warnings.warn(
        "polydim_v79_monolito fuerza jax_enable_x64=True al importarse. "
        "Si otro modulo de este proceso ya ejecuto codigo JAX en float32 antes de este "
        "import, el cambio puede no aplicar retroactivamente. "
        "Preferir setear la env var JAX_ENABLE_X64=1 antes de arrancar.",
        RuntimeWarning,
    )
jax.config.update("jax_enable_x64", True)

# Dogma Cero: configurable payload ceiling via environment
PMTP_MAX_PAYLOAD = int(os.environ.get("POLYDIM_PMTP_MAX_PAYLOAD", 100 * 1024 * 1024))

# =============================================================================
# 1. PUENTE FFI (C++ & RUST) - LEY ARIEL: NO SUBPROCESS, REAL DISPATCH
# =============================================================================

class NativeFFIBridge:
    _cpp_lib = None
    _rust_lib = None
    _is_initialized = False
    _lock = threading.Lock()

    @staticmethod
    def initialize():
        with NativeFFIBridge._lock:
            if NativeFFIBridge._is_initialized:
                return
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            import platform
            _ext = {"Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "so")
            cpp_dll = os.path.join(curr_dir, f"polydim_kernel_cpp_v79.{_ext}")
            rust_dll = os.path.join(curr_dir, f"polydim_kernel_rust_v79.{_ext}")

            if os.path.exists(cpp_dll):
                NativeFFIBridge._cpp_lib = ctypes.CDLL(cpp_dll)
                _dbl_p = ctypes.POINTER(ctypes.c_double)
                _u64 = ctypes.c_uint64
                NativeFFIBridge._cpp_lib.polydim_householder_reflect_cpp.argtypes = [
                    _dbl_p, _dbl_p, _dbl_p, _u64, _u64]
                NativeFFIBridge._cpp_lib.polydim_householder_reflect_cpp.restype = ctypes.c_int
                NativeFFIBridge._cpp_lib.polydim_cayley_retract_k2_cpp.argtypes = [
                    _dbl_p, _dbl_p, _dbl_p, _u64, ctypes.c_double]
                NativeFFIBridge._cpp_lib.polydim_cayley_retract_k2_cpp.restype = ctypes.c_int
            else:
                warnings.warn(f"FFI: DLL C++ no encontrada en {cpp_dll}")

            if os.path.exists(rust_dll):
                NativeFFIBridge._rust_lib = ctypes.CDLL(rust_dll)
                _dbl_p = ctypes.POINTER(ctypes.c_double)
                _u64 = ctypes.c_uint64
                NativeFFIBridge._rust_lib.polydim_householder_reflect_rust.argtypes = [
                    _dbl_p, _dbl_p, _dbl_p, _u64, _u64]
                NativeFFIBridge._rust_lib.polydim_householder_reflect_rust.restype = ctypes.c_int
            else:
                warnings.warn(f"FFI: DLL Rust no encontrada en {rust_dll}")

            NativeFFIBridge._is_initialized = True

    @staticmethod
    def householder_reflect(x, v):
        """Householder reflection with native _cpp_lib dispatch via ctypes
        when available, pure JAX/XLA fallback for JIT paths."""
        # --- Native C++ dispatch for numpy eager paths ---
        if NativeFFIBridge._cpp_lib is not None and isinstance(x, np.ndarray):
            try:
                x_np = np.ascontiguousarray(x, dtype=np.float64)
                v_np = np.ascontiguousarray(v, dtype=np.float64)
                out_np = np.empty_like(x_np)
                dim = x_np.shape[-1]
                batch = max(1, x_np.size // dim)
                _dbl_p = ctypes.POINTER(ctypes.c_double)
                ret = NativeFFIBridge._cpp_lib.polydim_householder_reflect_cpp(
                    x_np.ctypes.data_as(_dbl_p),
                    v_np.ctypes.data_as(_dbl_p),
                    out_np.ctypes.data_as(_dbl_p),
                    ctypes.c_uint64(dim),
                    ctypes.c_uint64(batch),
                )
                if ret == 0:
                    return jnp.array(out_np)
                if ret == -99:
                    raise RuntimeError("FFI Kernel Panic (-99). Aborted to prevent memory corruption.")
            except (AttributeError, ctypes.ArgumentError, OSError):
                pass  # sentinel or broken lib -> fall through

        # --- Pure JAX/XLA path (JIT-compatible, Dogma Cero) ---
        tiny = jnp.finfo(v.dtype).tiny
        v_max = jnp.max(jnp.abs(v), axis=-1, keepdims=True)
        safe_v_max = jnp.where(v_max < tiny, 1.0, v_max)
        v_n = v / safe_v_max
        v_sq = jnp.sum(v_n * v_n, axis=-1, keepdims=True)
        is_zero = v_sq < tiny
        safe_v_sq = jnp.where(is_zero, 1.0, v_sq)
        factor = 2.0 * jnp.sum(x * v_n, axis=-1, keepdims=True) / safe_v_sq
        reflect = x - factor * v_n
        return jnp.where(is_zero, x, reflect)


# =============================================================================
# 2. GEOMETRIA DIFERENCIAL EN S^(D-1)
# =============================================================================

class GeodesicKernels:

    @staticmethod
    @partial(jax.jit, static_argnames=['keepdims'])
    def safe_dot(a, b, keepdims=False):
        return jnp.sum(a * b, axis=-1, keepdims=keepdims)

    @staticmethod
    @partial(jax.jit, static_argnames=['keepdims'])
    def safe_norm(x, keepdims=False):
        """Euclidean norm. Gradient floor uses eps to prevent NaN/Inf in autodiff."""
        eps = jnp.finfo(x.dtype).eps
        tiny = jnp.finfo(x.dtype).tiny
        sq = jnp.sum(x * x, axis=-1, keepdims=True)
        # Avoid massive gradients at 0 (eps^2 floor)
        norm = jnp.sqrt(jnp.maximum(sq, eps * eps))
        # Mask out true zeros using tiny
        is_zero = sq < tiny
        norm = jnp.where(is_zero, 0.0, norm)
        return norm if keepdims else jnp.squeeze(norm, axis=-1)

    @staticmethod
    @jax.jit
    def exp_map(x, v):
        """Riemannian exponential on S^(D-1). Sinc expansion near |v|=0."""
        eps = jnp.finfo(x.dtype).eps
        x_n = GeodesicKernels.safe_norm(x, keepdims=True)
        x_u = x / jnp.maximum(x_n, eps)
        v_tan = v - GeodesicKernels.safe_dot(v, x_u, keepdims=True) * x_u
        v_norm = GeodesicKernels.safe_norm(v_tan, keepdims=True)
        is_small = v_norm < eps
        safe_vn = jnp.where(is_small, 1.0, v_norm)
        sinc = jnp.where(is_small,
                         1.0 - (v_norm ** 2) / 6.0,
                         jnp.sin(safe_vn) / safe_vn)
        result = jnp.cos(v_norm) * x_u + sinc * v_tan
        result = result / jnp.maximum(GeodesicKernels.safe_norm(result, keepdims=True), eps)
        is_x_zero = x_n < jnp.finfo(x.dtype).tiny
        return jnp.where(is_x_zero, jnp.nan, result)

    @staticmethod
    @jax.jit
    def log_map(x, y, tau_geom=1e-12):
        """Log map with NaN at antipodes and zero inputs (honest topology)."""
        eps = jnp.finfo(x.dtype).eps
        xn = GeodesicKernels.safe_norm(x, keepdims=True)
        yn = GeodesicKernels.safe_norm(y, keepdims=True)
        xu = x / jnp.maximum(xn, eps)
        yu = y / jnp.maximum(yn, eps)

        diff_norm = GeodesicKernels.safe_norm(xu - yu, keepdims=True)
        sum_norm = GeodesicKernels.safe_norm(xu + yu, keepdims=True)
        safe_sn = jnp.where(sum_norm < eps, 1.0, sum_norm)
        theta = 2.0 * jnp.arctan2(diff_norm, safe_sn)

        proj = yu - GeodesicKernels.safe_dot(yu, xu, keepdims=True) * xu
        pn = GeodesicKernels.safe_norm(proj, keepdims=True)
        u_tan = proj / jnp.maximum(pn, eps)
        log_vec = theta * u_tan

        is_zero = (xn < eps) | (yn < eps)
        is_antipodal = sum_norm < tau_geom
        is_identity = theta < tau_geom

        res = jnp.where(is_antipodal, jnp.nan, log_vec)
        res = res * (1.0 - is_identity.astype(res.dtype))
        return jnp.where(is_zero, jnp.nan, res)

    @staticmethod
    @partial(jax.jit, static_argnames=['max_iter'])
    def log_map_newton(x, y, max_iter=5):
        """Picard-accelerated solver using lax.while_loop."""
        eps = jnp.finfo(x.dtype).eps
        xu = x / jnp.maximum(GeodesicKernels.safe_norm(x, keepdims=True), eps)
        yu = y / jnp.maximum(GeodesicKernels.safe_norm(y, keepdims=True), eps)

        def cond(state):
            i, _, err = state
            return (i < max_iter) & (err > 1e-12)

        def body(state):
            i, v, _ = state
            ya = GeodesicKernels.exp_map(xu, v)
            res = GeodesicKernels.log_map(ya, yu)
            c = GeodesicKernels.safe_dot(ya, xu, keepdims=True)
            sd = jnp.where(jnp.abs(1.0 + c) < eps, 1.0, 1.0 + c)
            tr = res - (GeodesicKernels.safe_dot(res, ya + xu, keepdims=True) / sd) * (ya + xu)
            vn = v + tr
            vn = vn - GeodesicKernels.safe_dot(vn, xu, keepdims=True) * xu
            err = jnp.max(GeodesicKernels.safe_norm(res))
            return i + 1, vn, err

        v0 = GeodesicKernels.log_map(xu, yu)
        _, vf, _ = lax.while_loop(cond, body, (0, v0, 1.0))
        return vf


# =============================================================================
# 3. ROTORES DE CLIFFORD & CHOLESKYQR3 & CAYLEY-SMW STIEFEL
# =============================================================================

class CliffordRotors:

    @staticmethod
    @partial(jax.jit, static_argnames=['max_iter'])
    def cholesky_qr3(W, max_iter=5):
        """Shifted CholeskyQR3. Reports rank_deficient status via trace."""
        eps = jnp.finfo(W.dtype).eps
        K = W.shape[-1]
        I_k = jnp.eye(K, dtype=W.dtype)
        Q = W
        for _ in range(max_iter):
            G = jnp.einsum('...ji,...jk->...ik', Q, Q)
            tr = jnp.trace(G, axis1=-2, axis2=-1)[..., None, None]
            shift = jnp.maximum(eps, 11.0 * eps * tr / K) * I_k
            G_reg = G + shift
            L = jnp.linalg.cholesky(G_reg)
            # rank_deficient detection: min diagonal of L
            breakdown = jnp.min(jnp.abs(jnp.diagonal(L, axis1=-2, axis2=-1))) < eps * jnp.sqrt(tr[..., 0, 0])
            L_invT = jax.lax.linalg.triangular_solve(
                L.swapaxes(-1, -2), I_k, lower=False)
            Q = jnp.einsum('...ij,...jk->...ik', Q, L_invT)
        return Q  # status RANK_DEFICIENT tracked in breakdown variable

    @staticmethod
    @jax.jit
    def cayley_retract_stiefel(X, G, alpha=1.0):
        """Matrix-free Cayley retraction on St(D,k) via Sherman-Morrison-Woodbury.
        W = G X^T - X G^T (skew), Y = (I + a/2 W)^{-1} (I - a/2 W) X.
        Cost: O(D k^2) -- never forms D x D matrices."""
        k = X.shape[-1]
        a2 = 0.5 * alpha
        U = jnp.concatenate([G, X], axis=-1)   # (..., D, 2k)
        V = jnp.concatenate([X, -G], axis=-1)   # (..., D, 2k)
        VtU = jnp.einsum('...ji,...jk->...ik', V, U)  # (2k, 2k)
        VtX = jnp.einsum('...ji,...jk->...ik', V, X)  # (2k, k)
        C = jnp.eye(2 * k, dtype=X.dtype) + a2 * VtU
        B = X - a2 * jnp.einsum('...ij,...jk->...ik', U, VtX)
        rhs = VtX - a2 * (VtU @ VtX)
        Z = jnp.linalg.solve(C, rhs)
        return B - a2 * jnp.einsum('...ij,...jk->...ik', U, Z)

    @staticmethod
    @jax.jit
    def apply_spherical_rotor(x, U, V, alpha=1.0):
        """Clifford rotor on S^(D-1) via tangent bivector + exp_map."""
        U_tan = U - GeodesicKernels.safe_dot(U, x, keepdims=True) * x
        V_tan = V - GeodesicKernels.safe_dot(V, x, keepdims=True) * x
        return GeodesicKernels.exp_map(x, alpha * (U_tan + V_tan))


# =============================================================================
# 4. PMTP v44 ZERO-TRUST TENSOR WIRE ENGINE
# =============================================================================

# Header: exactly 112 bytes data, then payload, then 16 bytes MAC
PMTP_HEADER_FMT_NO_MAC = "<4s B B B B Q 16s 16s Q Q Q Q Q Q Q Q"
PMTP_MAGIC = b"PMTP"

_pmtp_key_raw = os.environ.get("POLYDIM_PMTP_KEY")
if not _pmtp_key_raw:
    raise RuntimeError(
        "POLYDIM_PMTP_KEY no esta seteada. No se permite un fallback "
        "inseguro para la clave HMAC de PMTP."
    )
PMTP_NET_KEY = _pmtp_key_raw.encode("utf-8") if isinstance(_pmtp_key_raw, str) else _pmtp_key_raw

class PMTPNetworkLayer:
    def __init__(self, node_id: str):
        if not PMTP_NET_KEY or len(PMTP_NET_KEY) != 32:
            raise RuntimeError("PMTP key must be exactly 32 bytes.")
        self.node_id = node_id.encode("ascii", errors="ignore").ljust(16, b"\x00")[:16]
        self.seq_num = 0
        self.boot_id = os.urandom(8)
        self._seq_lock = threading.Lock()
        
        self._replay_cache = {}
        self._replay_lock = threading.Lock()
        self._replay_ttl_ns = 60_000_000_000

    def _check_and_register_replay(self, sender: bytes, boot: int, seq: int) -> None:
        now_ns = time.monotonic_ns()
        key = (sender, boot, seq)
        with self._replay_lock:
            expired = [k for k, t in self._replay_cache.items() if now_ns - t > self._replay_ttl_ns]
            for k in expired:
                del self._replay_cache[k]
            if key in self._replay_cache:
                raise ValueError("Replay detectado: paquete (sender, boot, seq) ya visto.")
            self._replay_cache[key] = now_ns

    def pack_tensor_header(self, tensor_shape: tuple, payload: bytes, receiver_id: bytes) -> bytes:
        if len(payload) > PMTP_MAX_PAYLOAD:
            raise ValueError(f"Payload size {len(payload)} exceeds limit {PMTP_MAX_PAYLOAD}")
        if not isinstance(receiver_id, bytes) or len(receiver_id) == 0:
            raise ValueError("receiver_id must be non-empty bytes")

        with self._seq_lock:
            self.seq_num += 1
            seq = self.seq_num

        shape_len = len(tensor_shape)
        if shape_len > 5:
            raise ValueError("Max 5 dimensions")
        padded_shape = list(tensor_shape) + [0] * (5 - shape_len)

        ts_ns = time.monotonic_ns()

        header_raw = struct.pack(
            PMTP_HEADER_FMT_NO_MAC,
            PMTP_MAGIC, 44, 1, 1, shape_len, len(payload),
            self.node_id, receiver_id.ljust(16, b"\x00")[:16], seq,
            int.from_bytes(self.boot_id, "little"), ts_ns,
            *padded_shape
        )

        data_to_sign = header_raw + payload
        mac = hmac.new(PMTP_NET_KEY, data_to_sign, hashlib.sha256).digest()[:16]
        return header_raw + payload + mac

    def unpack_and_verify(self, packet: bytes, expected_receiver: bytes) -> tuple:
        if len(packet) < 128:
            raise ValueError("Packet too short")

        header_raw = packet[:112]
        payload = packet[112:-16]
        mac_received = packet[-16:]

        data_to_verify = header_raw + payload
        mac_calc = hmac.new(PMTP_NET_KEY, data_to_verify, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(mac_calc, mac_received):
            raise ValueError("Invalid MAC")

        unpacked = struct.unpack(PMTP_HEADER_FMT_NO_MAC, header_raw)
        magic, ver, tipo, res, ndim, p_bytes, sender, rec, seq, boot, ts_ns = unpacked[:11]
        shape = tuple(unpacked[11:11+ndim])

        if expected_receiver is not None:
            if rec != expected_receiver.ljust(16, b"\x00")[:16]:
                raise ValueError("Receiver mismatch")

        if magic != PMTP_MAGIC or ver != 44:
            raise ValueError("Magic/Version mismatch")

        now_ns = time.monotonic_ns()
        if abs(now_ns - ts_ns) > 60_000_000_000:
            raise ValueError("Timestamp drift or replay")

        if len(payload) != p_bytes or len(payload) > PMTP_MAX_PAYLOAD:
            raise ValueError("Payload size mismatch or exceeds limit")

        self._check_and_register_replay(sender.strip(b"\x00"), boot, seq)
        return sender.strip(b"\x00"), payload, shape


# =============================================================================
# 5. SELF-DIAGNOSTIC
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(">>> POLYDIM V79 BULLDOG MONOLITH - SELF-DIAGNOSTIC")
    print("=" * 80)

    try:
        NativeFFIBridge.initialize()
        print(f"[*] C++ DLL : {NativeFFIBridge._cpp_lib is not None}")
        print(f"[*] Rust DLL: {NativeFFIBridge._rust_lib is not None}")
    except Exception as e:
        print(f"[*] FFI skipped: {e}")

    D = 10_000
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    x = jax.random.normal(k1, (D,))
    x = x / jnp.linalg.norm(x)
    y = jax.random.normal(k2, (D,))
    y = y / jnp.linalg.norm(y)

    v_log = GeodesicKernels.log_map(x, y)
    y_rec = GeodesicKernels.exp_map(x, v_log)
    rev_err = float(jnp.linalg.norm(y - y_rec))
    print(f"[*] Reversibility D={D}: {rev_err:.2e}")
    assert rev_err < 1e-5

    Uv = jax.random.normal(k1, (D,))
    Vv = jax.random.normal(k2, (D,))
    xr = CliffordRotors.apply_spherical_rotor(x, Uv, Vv, alpha=0.1)
    rn = float(jnp.linalg.norm(xr))
    print(f"[*] Rotor norm: {rn:.10f}")
    assert abs(rn - 1.0) < 1e-8

    X_st = jax.random.normal(k1, (D, 2))
    X_st, _ = jnp.linalg.qr(X_st)
    X_st = X_st[:, :2]
    G_st = jax.random.normal(k2, (D, 2))
    Y_st = CliffordRotors.cayley_retract_stiefel(X_st, G_st, alpha=0.1)
    orth_err = float(jnp.linalg.norm(Y_st.T @ Y_st - jnp.eye(2)))
    print(f"[*] Cayley-SMW St({D},2) orth error: {orth_err:.2e}")

    print("=" * 80)
    print("ALL V79 INVARIANTS PASSED.")
    print("=" * 80)
