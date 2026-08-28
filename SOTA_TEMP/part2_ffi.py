
# ==============================================================================
# 2. BRIDGE FFI (ZERO-COPY JAX.FFI & TOCTOU-SAFE)
# ==============================================================================
class NativeFFIBridge:
    _rust_dll = None
    _cpp_dll = None
    _init_lock = threading.Lock()
    _temp_files = []
    _xla_registered = False

    @classmethod
    def initialize(cls):
        with cls._init_lock:
            if cls._xla_registered:
                return

            cache_dir = os.path.expanduser(f"~/.cache/polydim_ffi/worker_{os.getpid()}")
            os.makedirs(cache_dir, exist_ok=True)

            import shutil
            import subprocess

            # RUST COMPILATION
            rust_uuid = uuid.uuid4().hex
            rust_src_path = os.path.join(cache_dir, f"kernel_rust_{rust_uuid}.rs")
            rust_dll_temp = os.path.join(cache_dir, f"rust_lib_{rust_uuid}.dll")
            rust_dll_final = os.path.join(cache_dir, "rust_lib_v75_xla.dll")
            
            with open(rust_src_path, "w") as f:
                f.write(RUST_SOURCE)
            
            rustc = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
            try:
                subprocess.run([rustc, "--crate-type", "cdylib", rust_src_path, "-o", rust_dll_temp], check=True, capture_output=True)
                os.replace(rust_dll_temp, rust_dll_final)
                cls._rust_dll = ctypes.CDLL(rust_dll_final)
            except Exception as e:
                warnings.warn(f"Rust FFI no disponible. Fallback a C++. Error: {e}")

            # C++ COMPILATION
            cpp_uuid = uuid.uuid4().hex
            cpp_src_path = os.path.join(cache_dir, f"kernel_cpp_{cpp_uuid}.cpp")
            cpp_dll_temp = os.path.join(cache_dir, f"cpp_lib_{cpp_uuid}.dll")
            cpp_dll_final = os.path.join(cache_dir, "cpp_lib_v75_xla.dll")

            with open(cpp_src_path, "w") as f:
                f.write(CPP_SOURCE)
            
            if not cls._rust_dll:
                try:
                    if sys.platform == "win32":
                        subprocess.run(["cl.exe", "/LD", "/O2", cpp_src_path, f"/Fe{cpp_dll_temp}"], check=True, capture_output=True)
                    else:
                        subprocess.run(["g++", "-shared", "-O3", "-fPIC", cpp_src_path, "-o", cpp_dll_temp], check=True, capture_output=True)
                    
                    os.replace(cpp_dll_temp, cpp_dll_final)
                    cls._cpp_dll = ctypes.CDLL(cpp_dll_final)
                except Exception as e:
                    warnings.warn(f"C++ FFI no disponible. Fallback a Python JAX puro. Error: {e}")

            cls._temp_files.extend([rust_src_path, cpp_src_path])

            # -----------------------------------------------------------------
            # XLA PYCAPSULE INJECTION (KILL THE GIL)
            # -----------------------------------------------------------------
            if cls._rust_dll or cls._cpp_dll:
                ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
                ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
                
                if cls._rust_dll:
                    ptr = ctypes.cast(cls._rust_dll.polydim_rust_householder_xla, ctypes.c_void_p)
                else:
                    ptr = ctypes.cast(cls._cpp_dll.polydim_cpp_householder_xla, ctypes.c_void_p)
                    
                capsule = ctypes.pythonapi.PyCapsule_New(ptr, b"xla._CUSTOM_CALL_TARGET", None)
                jax.ffi.register_custom_call("householder_xla", capsule, api_version=1)
                cls._xla_registered = True

    @classmethod
    def cleanup(cls):
        # FIX V74.1: NUNCA LLAMAR A dlclose() o FreeLibrary().
        for path in cls._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

    @classmethod
    @jax.jit
    def householder_reflect(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Se asume que initialize() fue llamado externamente antes del primer JIT,
        # o que el fallback puro se usa si no hay DLLs disponibles.
        if not cls._xla_registered:
            return cls._jax_fallback(x, v)
            
        dim = jnp.array([x.shape[-1]], dtype=jnp.uint64)
        out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        
        # El XLA Custom Call inyecta el Kernel nativo puro en el Grafo (Zero-Copy)
        return jax.ffi.ffi_call(
            "householder_xla",
            result_shape_dtypes=out_shape,
            x=x,
            v=v,
            dim=dim,
            has_side_effect=False
        )

    @classmethod
    def _jax_fallback(cls, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Fallback diferenciable puro en JAX
        v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)
        dot_xv = jnp.sum(x * v, axis=-1, keepdims=True)
        safe_v_norm_sq = jnp.where(v_norm_sq < 1e-30, 1.0, v_norm_sq)
        scale = 2.0 * dot_xv / safe_v_norm_sq
        reflection = x - scale * v
        return jnp.where(v_norm_sq < 1e-30, x, reflection)
