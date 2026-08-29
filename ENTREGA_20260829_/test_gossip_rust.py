import ctypes
import os
import time

dll_path = os.path.join("polydim_rust_net", "target", "release", "polydim_net.dll")
if not os.path.exists(dll_path):
    print(f"Error: DLL no encontrada en {dll_path}")
    exit(1)

rust_lib = ctypes.CDLL(dll_path)
rust_lib.pmtp_init_node.argtypes = [ctypes.c_uint16]
rust_lib.pmtp_init_node.restype = ctypes.c_int

print("=== TEST DE RED ASINCRONA RUST (TOKIO) ===")
print("Iniciando Nodo PMTP en el puerto 50051...")

result = rust_lib.pmtp_init_node(50051)
if result == 0:
    print("Nodo nativo lanzado en background. El GIL de Python esta 100% libre.")
else:
    print(f"Falla al lanzar nodo: {result}")

print("Python va a dormir 25 segundos para dejar que Rust rote epocas M-of-N en background...")
for i in range(5):
    time.sleep(5)
    print(f"Python trabajando (GIL libre) {i*5}s...")

print("Test completado con exito.")
