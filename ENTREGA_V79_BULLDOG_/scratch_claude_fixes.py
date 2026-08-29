import os
import sys

with open("E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. JAX x64 global side-effect warning
jax_import = 'jax.config.update("jax_enable_x64", True)'
jax_fix = '''if not jax.config.jax_enable_x64:
    warnings.warn(
        "polydim_v79_monolito fuerza jax_enable_x64=True al importarse. "
        "Si otro modulo de este proceso ya ejecuto codigo JAX en float32 antes de este "
        "import, el cambio puede no aplicar retroactivamente. "
        "Preferir setear la env var JAX_ENABLE_X64=1 antes de arrancar.",
        RuntimeWarning,
    )
jax.config.update("jax_enable_x64", True)'''
code = code.replace(jax_import, jax_fix)


# 2. NativeFFIBridge dll vs so vs dylib
dll_load = '''            cpp_dll = os.path.join(curr_dir, "polydim_kernel_cpp_v79.dll")
            rust_dll = os.path.join(curr_dir, "polydim_kernel_rust_v79.dll")'''
dll_fix = '''            import platform
            _ext = {"Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "so")
            cpp_dll = os.path.join(curr_dir, f"polydim_kernel_cpp_v79.{_ext}")
            rust_dll = os.path.join(curr_dir, f"polydim_kernel_rust_v79.{_ext}")'''
code = code.replace(dll_load, dll_fix)


# 3. tau_geom hardcoded in log_map
log_map_def = '''    def log_map(x: jnp.ndarray, y: jnp.ndarray, tau_geom: float = 1e-12) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps'''
log_map_fix = '''    def log_map(x: jnp.ndarray, y: jnp.ndarray, tau_geom: float = None) -> jnp.ndarray:
        eps = jnp.finfo(x.dtype).eps
        if tau_geom is None:
            tau_geom = eps * 10.0'''
code = code.replace(log_map_def, log_map_fix)


# 4. PMTP Layer rewrite
import re
pmtp_layer = r"""class PMTPNetworkLayer:.*?return sender\.strip\(b"\\x00"\), p_bytes, shape"""
pmtp_fix = r"""class PMTPNetworkLayer:
    def __init__(self, node_id: str):
        if not PMTP_NET_KEY or len(PMTP_NET_KEY) != 32:
            raise RuntimeError("PMTP key must be exactly 32 bytes.")
        self.node_id = node_id.encode("utf-8").ljust(16, b"\x00")[:16]
        self.seq_num = 0
        self.boot_id = os.urandom(8)
        self._seq_lock = threading.Lock()
        
        self._replay_cache = {}
        self._replay_lock = threading.Lock()
        self._replay_ttl = 60.0

    def _check_and_register_replay(self, sender: bytes, boot: int, seq: int) -> None:
        now = time.time()
        key = (sender, boot, seq)
        with self._replay_lock:
            expired = [k for k, t in self._replay_cache.items() if now - t > self._replay_ttl]
            for k in expired:
                del self._replay_cache[k]
            if key in self._replay_cache:
                raise ValueError("Replay detectado: paquete (sender, boot, seq) ya visto.")
            self._replay_cache[key] = now

    def pack_tensor_header(self, tensor_shape, payload_bytes, receiver_id):
        if payload_bytes > PMTP_MAX_PAYLOAD:
            raise ValueError(f"Anti-DoS: payload {payload_bytes} > {PMTP_MAX_PAYLOAD}")
        with self._seq_lock:
            self.seq_num += 1
            seq = self.seq_num
        ndim = len(tensor_shape)
        if ndim > 5:
            raise ValueError("PMTP: max 5 dims")
        padded = list(tensor_shape) + [0] * (5 - ndim)
        ts = time.time()
        rec = receiver_id if isinstance(receiver_id, bytes) else receiver_id.encode("utf-8")
        rec = rec.ljust(16, b"\x00")[:16]
        
        full = struct.pack(
            PMTP_BODY_FMT,
            PMTP_MAGIC, 44, 1, 0, ndim, payload_bytes,
            self.node_id, rec, seq,
            int.from_bytes(self.boot_id, "little"), ts,
            *padded,
        )
        data = full[:112]
        mac = hmac.new(PMTP_NET_KEY, data, hashlib.sha256).digest()[:16]
        return data + mac

    def unpack_and_verify(self, header_bytes, expected_receiver=None):
        if len(header_bytes) != 128:
            raise ValueError(f"PMTP header length {len(header_bytes)} != 128")
        data = header_bytes[:112]
        mac_recv = header_bytes[112:]
        mac_calc = hmac.new(PMTP_NET_KEY, data, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(mac_calc, mac_recv):
            raise ValueError("HMAC-SHA256 rejected")
            
        fields = struct.unpack(PMTP_BODY_FMT, data)
        magic, ver, typ, res, ndim, p_bytes = fields[:6]
        sender, rec, seq, boot, ts = fields[6:11]
        
        if expected_receiver is not None:
            exp_rec = expected_receiver if isinstance(expected_receiver, bytes) else expected_receiver.encode("utf-8")
            if not hmac.compare_digest(rec.strip(b"\x00"), exp_rec.strip(b"\x00")):
                raise ValueError("Paquete no destinado a este receptor.")
                
        if magic != PMTP_MAGIC or ver != 44:
            raise ValueError("Magic/Version mismatch")
        if abs(time.time() - ts) > 60.0:
            raise ValueError("Anti-Replay: drift > 60s")
        if p_bytes > PMTP_MAX_PAYLOAD:
            raise ValueError("Anti-DoS: payload too large")
            
        self._check_and_register_replay(sender.strip(b"\x00"), boot, seq)
        shape = tuple(fields[11:11 + ndim])
        return sender.strip(b"\x00"), p_bytes, shape"""
code = re.sub(pmtp_layer, pmtp_fix.replace("\\", "\\\\"), code, flags=re.DOTALL)


# 5. Fix PMTP string key
key_load = '''_pmtp_key_raw = os.environ.get("POLYDIM_PMTP_KEY", "")
PMTP_NET_KEY = _pmtp_key_raw.encode("utf-8") if _pmtp_key_raw else b"x" * 32'''
key_fix = '''_pmtp_key_raw = os.environ.get("POLYDIM_PMTP_KEY")
if not _pmtp_key_raw:
    raise RuntimeError(
        "POLYDIM_PMTP_KEY no esta seteada. No se permite un fallback "
        "inseguro para la clave HMAC de PMTP."
    )
PMTP_NET_KEY = _pmtp_key_raw.encode("utf-8") if isinstance(_pmtp_key_raw, str) else _pmtp_key_raw'''
code = code.replace(key_load, key_fix)

with open("E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito.py", "w", encoding="utf-8") as f:
    f.write(code)
