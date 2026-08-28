
# ==============================================================================
# 2. BRIDGE FFI (ZERO-TRUST & TOCTOU-SAFE)
# ==============================================================================
class NativeFFIBridge:
    _rust_dll = None
    _cpp_dll = None
    _init_lock = threading.Lock()
    _temp_files = []  # Archivos a limpiar al salir el proceso

    @classmethod
    def initialize(cls):
        with cls._init_lock:
            if cls._rust_dll is not None and cls._cpp_dll is not None:
                return

            cache_dir = os.path.expanduser(f"~/.cache/polydim_ffi/worker_{os.getpid()}")
            os.makedirs(cache_dir, exist_ok=True)

            # ---------------------------------------------------------
            # FIX TOCTOU: Renombrado atómico con UUID.
            # ---------------------------------------------------------
            import shutil
            import subprocess

            # RUST COMPILATION
            rust_uuid = uuid.uuid4().hex
            rust_src_path = os.path.join(cache_dir, f"kernel_rust_{rust_uuid}.rs")
            rust_dll_temp = os.path.join(cache_dir, f"rust_lib_{rust_uuid}.dll")
            rust_dll_final = os.path.join(cache_dir, "rust_lib_v75.dll")
            
            with open(rust_src_path, "w") as f:
                f.write(RUST_SOURCE)
            
            rustc = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
            try:
                subprocess.run([rustc, "--crate-type", "cdylib", rust_src_path, "-o", rust_dll_temp], check=True, capture_output=True)
                os.replace(rust_dll_temp, rust_dll_final)  # Atómico
                cls._rust_dll = ctypes.CDLL(rust_dll_final)
                cls._rust_dll.polydim_rust_householder_reflect.argtypes = [
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
                ]
            except Exception as e:
                warnings.warn(f"Rust FFI no disponible. Fallback a C++. Error: {e}")

            # C++ COMPILATION
            cpp_uuid = uuid.uuid4().hex
            cpp_src_path = os.path.join(cache_dir, f"kernel_cpp_{cpp_uuid}.cpp")
            cpp_dll_temp = os.path.join(cache_dir, f"cpp_lib_{cpp_uuid}.dll")
            cpp_dll_final = os.path.join(cache_dir, "cpp_lib_v75.dll")

            with open(cpp_src_path, "w") as f:
                f.write(CPP_SOURCE)
            
            # Windows/Linux compatible compilation
            try:
                if sys.platform == "win32":
                    subprocess.run(["cl.exe", "/LD", "/O2", cpp_src_path, f"/Fe{cpp_dll_temp}"], check=True, capture_output=True)
                else:
                    subprocess.run(["g++", "-shared", "-O3", "-fPIC", cpp_src_path, "-o", cpp_dll_temp], check=True, capture_output=True)
                
                os.replace(cpp_dll_temp, cpp_dll_final) # Atómico
                cls._cpp_dll = ctypes.CDLL(cpp_dll_final)
                cls._cpp_dll.polydim_cpp_householder_reflect.argtypes = [
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
                ]
            except Exception as e:
                warnings.warn(f"C++ FFI no disponible. Fallback a Python JAX puro. Error: {e}")

            cls._temp_files.extend([rust_src_path, cpp_src_path])

    @classmethod
    def cleanup(cls):
        # FIX V74.1: NUNCA LLAMAR A dlclose() o FreeLibrary()
        # El SO reclamará la memoria virtual al terminar el proceso.
        # Esto soluciona los SIGSEGV masivos (Use-After-Unload) en concurrencia.
        for path in cls._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

    @classmethod
    def householder_reflect(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        if isinstance(x, jax.core.Tracer) or isinstance(v, jax.core.Tracer):
            return cls._jax_fallback(x, v)
        
        cls.initialize()
        
        # Sincronizamos a CPU (evitar bloqueo GIL masivo)
        x_np = jax.device_get(x).astype(np.float64)
        v_np = jax.device_get(v).astype(np.float64)
        
        x2d = x_np.reshape(-1, x_np.shape[-1])
        v2d = v_np.reshape(-1, v_np.shape[-1])
        out = np.empty_like(x_np)
        out2d = out.reshape(-1, out.shape[-1])
        
        if cls._rust_dll:
            fn = cls._rust_dll.polydim_rust_householder_reflect
        elif cls._cpp_dll:
            fn = cls._cpp_dll.polydim_cpp_householder_reflect
        else:
            return cls._jax_fallback(x, v)
            
        dim = x2d.shape[-1]
        for i in range(x2d.shape[0]):
            x_cont = np.ascontiguousarray(x2d[i])
            v_cont = np.ascontiguousarray(v2d[i])
            out_cont = np.empty_like(x_cont)
            
            ret = fn(
                x_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                v_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_size_t(dim)
            )
            if ret != 0:
                return cls._jax_fallback(x, v)
            out2d[i] = out_cont
            
        return jax.device_put(out).astype(x.dtype)

    @classmethod
    def _jax_fallback(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Fallback diferenciable puro en JAX
        v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)
        dot_xv = jnp.sum(x * v, axis=-1, keepdims=True)
        
        # FIX V74.1: Asesinato de gradientes en XLA mitigado con where() en lugar de maximum()
        safe_v_norm_sq = jnp.where(v_norm_sq < 1e-30, 1.0, v_norm_sq)
        scale = 2.0 * dot_xv / safe_v_norm_sq
        reflection = x - scale * v
        
        # Silenciar la salida completamente si el vector es infinitesimal
        return jnp.where(v_norm_sq < 1e-30, x, reflection)
