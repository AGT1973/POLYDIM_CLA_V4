import ctypes
import jax
import jax.numpy as jnp
import os

dll_path = os.path.join("polydim_rust_net", "target", "release", "polydim_net.dll")
if not os.path.exists(dll_path):
    print(f"Error: No se encontro la DLL en {dll_path}")
    exit(1)

rust_lib = ctypes.CDLL(dll_path)
rust_lib.pmtp_send_tensor_dlpack.argtypes = [ctypes.c_void_p]
rust_lib.pmtp_send_tensor_dlpack.restype = ctypes.c_int

print("=== TEST DE FRONTERA ASINTOTICO: PYTHON -> JAX -> RUST (DLPack) ===")
print("ADVERTENCIA: Solicitando 80 GB de memoria RAM (float64).")
print("Generando tensor en JAX...")

key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (100000, 100000), dtype=jnp.float64)

dlpack_capsule = x.__dlpack__()
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

raw_pointer = ctypes.pythonapi.PyCapsule_GetPointer(dlpack_capsule, b"dltensor")

print(f"Puntero crudo DLPack extraido: {hex(raw_pointer)}")
print("Inyectando puntero en memoria de Rust...")
result = rust_lib.pmtp_send_tensor_dlpack(raw_pointer)

if result == 0:
    print("EXITO: Rust ha ingerido la masa asintotica de 80 GB sin problemas.")
else:
    print(f"FALLO: Rust retorno codigo {result}")
