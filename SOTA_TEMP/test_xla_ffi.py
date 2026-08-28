import os
import ctypes
import jax
import jax.numpy as jnp
import numpy as np
from jax.ffi import ffi_call

# 1. Dummy C function (Adds 1.0 to each element in float64)
# Signature matching XLA FFI:
# void add_one(void* stream, void** buffers, const char* opaque, size_t opaque_len, XlaCustomCallStatus* status)
CPP_SOURCE = """
#include <cstddef>
#ifdef _WIN32
#define EXPORT_SYM __declspec(dllexport)
#else
#define EXPORT_SYM __attribute__((visibility("default")))
#endif

extern "C" {
    EXPORT_SYM void add_one(void* stream, void** buffers, const char* opaque, size_t opaque_len, void* status) {
        double* in = reinterpret_cast<double*>(buffers[0]);
        double* out = reinterpret_cast<double*>(buffers[1]);
        // Dummy size assumption for test (let's say 4)
        for (int i = 0; i < 4; ++i) {
            out[i] = in[i] + 1.0;
        }
    }
}
"""

def main():
    dll_path = "dummy_ffi.dll"
    with open("dummy_ffi.cpp", "w") as f:
        f.write(CPP_SOURCE)
    
    # Compile
    os.system(f"cl.exe /LD /O2 dummy_ffi.cpp /Fedummy_ffi.dll >nul 2>&1")
    
    if not os.path.exists(dll_path):
        print("Fallo compiland dummy dll")
        return
        
    dll = ctypes.CDLL(os.path.abspath(dll_path))
    func = dll.add_one
    
    # 2. Convert to PyCapsule using ctypes.pythonapi
    ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    
    func_ptr = ctypes.cast(func, ctypes.c_void_p)
    capsule = ctypes.pythonapi.PyCapsule_New(func_ptr, b"xla._CUSTOM_CALL_TARGET", None)
    
    # 3. Register in JAX
    jax.ffi.register_custom_call("add_one", capsule, api_version=1)
    
    # 4. Bind FFI Call
    @jax.jit
    def fast_add_one(x):
        return ffi_call("add_one", 
                        result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
                        x=x)
                        
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    print("Input:", x)
    y = fast_add_one(x)
    print("Output FFI:", y)

if __name__ == "__main__":
    main()
