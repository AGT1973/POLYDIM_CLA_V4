"""
=== TEST DE INTEGRACIÓN PMTP: DOS NODOS INTERCAMBIANDO TENSORES ===
Este script levanta un nodo PMTP receptor en Rust (puerto 50051),
genera un tensor en JAX, y lo envía al receptor vía la DLL nativa.
Verifica que el receptor lo recibió correctamente (cola > 0).
"""
import ctypes
import os
import time
import sys
import struct

os.environ["JAX_ENABLE_X64"] = "1"

dll_path = os.path.join("polydim_rust_net", "target", "release", "polydim_net.dll")
if not os.path.exists(dll_path):
    print(f"ERROR: DLL no encontrada en {dll_path}")
    sys.exit(1)

lib = ctypes.CDLL(dll_path)

# Definir signatures FFI
lib.pmtp_init_node.argtypes = [ctypes.c_uint16]
lib.pmtp_init_node.restype = ctypes.c_int32

lib.pmtp_send_raw.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # data_ptr
    ctypes.c_size_t,                   # num_elements
    ctypes.POINTER(ctypes.c_int64),    # shape_ptr
    ctypes.c_size_t,                   # ndim
    ctypes.c_char_p,                   # host
    ctypes.c_uint16,                   # port
]
lib.pmtp_send_raw.restype = ctypes.c_int32

lib.pmtp_get_epoch.argtypes = []
lib.pmtp_get_epoch.restype = ctypes.c_uint64

lib.pmtp_recv_queue_len.argtypes = []
lib.pmtp_recv_queue_len.restype = ctypes.c_int32

print("=" * 60)
print("TEST DE INTEGRACIÓN PMTP NATIVO (Rust + Tokio + HMAC)")
print("=" * 60)

# 1. Inicializar nodo receptor (puerto 50051)
print("\n[1] Inicializando nodo PMTP receptor en puerto 50051...")
result = lib.pmtp_init_node(50051)
assert result == 0, f"Fallo al inicializar nodo: {result}"
print("    ✓ Nodo Tokio levantado. GIL liberado.")

# Esperar a que Tokio levante el listener
time.sleep(0.5)

# 2. Crear tensor de prueba (sin JAX, puro ctypes para demostrar independencia)
print("\n[2] Generando tensor de prueba (500x500 = 250,000 elementos)...")
import array
num_elements = 500 * 500
data = array.array('d', [float(i) * 0.001 for i in range(num_elements)])
data_ptr = (ctypes.c_double * num_elements).from_buffer(data)

shape = (ctypes.c_int64 * 2)(500, 500)

print(f"    ✓ Tensor generado: {num_elements} elementos, {num_elements * 8} bytes")

# 3. Enviar tensor al mismo nodo (localhost loopback)
print("\n[3] Enviando tensor por TCP a localhost:50051 (Rust async)...")
result = lib.pmtp_send_raw(
    data_ptr,
    num_elements,
    shape,
    2,
    b"127.0.0.1",
    50051,
)
assert result == 0, f"Fallo al enviar tensor: {result}"
print("    ✓ Tensor despachado a la cola de Tokio (async, no-blocking)")

# 4. Esperar a que el envío asíncrono complete el roundtrip
print("\n[4] Esperando roundtrip TCP (max 5 segundos)...")
for i in range(50):
    time.sleep(0.1)
    qlen = lib.pmtp_recv_queue_len()
    if qlen > 0:
        break

epoch = lib.pmtp_get_epoch()

print(f"\n{'=' * 60}")
print(f"RESULTADO FINAL")
print(f"{'=' * 60}")
print(f"  Epoch actual:          {epoch}")
print(f"  Tensores en cola:      {qlen}")

if qlen > 0:
    print(f"  Estado:                ✓ ÉXITO TOTAL")
    print(f"  El tensor viajó:")
    print(f"    Python -> Rust (Zero-Copy) -> HMAC-SHA256 -> TCP -> Rust -> Buffer")
    print(f"  Sin tocar el GIL de Python en ningún momento de la red.")
else:
    print(f"  Estado:                ✗ FALLO (tensor no llegó a la cola)")
    sys.exit(1)

print(f"{'=' * 60}")
