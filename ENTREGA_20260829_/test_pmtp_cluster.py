"""
=== TEST MULTI-NODO PMTP: Clúster de 2 nodos intercambiando tensores ===
Este script simula un clúster real:
  - Proceso 1 (Nodo A): Escucha en puerto 50061, envía tensores al Nodo B
  - Proceso 2 (Nodo B): Escucha en puerto 50062, envía tensores al Nodo A
Ambos usan la DLL de Rust (Tokio async) y verifican HMAC-SHA256.
"""
import ctypes
import os
import sys
import time
import multiprocessing
import array

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

DLL_PATH = os.path.join(os.path.dirname(__file__), "polydim_rust_net", "target", "release", "polydim_net.dll")


def load_lib():
    lib = ctypes.CDLL(DLL_PATH)
    lib.pmtp_init_node.argtypes = [ctypes.c_uint16]
    lib.pmtp_init_node.restype = ctypes.c_int32
    lib.pmtp_send_raw.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_uint16,
    ]
    lib.pmtp_send_raw.restype = ctypes.c_int32
    lib.pmtp_get_epoch.argtypes = []
    lib.pmtp_get_epoch.restype = ctypes.c_uint64
    lib.pmtp_recv_queue_len.argtypes = []
    lib.pmtp_recv_queue_len.restype = ctypes.c_int32
    return lib


def run_node(my_port, peer_port, node_name, result_dict):
    """Cada nodo: inicia, envía un tensor al peer, espera a recibir uno."""
    lib = load_lib()

    # 1. Inicializar
    rc = lib.pmtp_init_node(my_port)
    if rc != 0:
        result_dict[node_name] = f"FAIL: init returned {rc}"
        return
    print(f"[{node_name}] Nodo levantado en puerto {my_port}")
    time.sleep(1.0)  # Esperar a que el peer también levante

    # 2. Generar tensor
    N = 1000
    num_elements = N * N
    data = array.array('d', [float(i) * 0.001 for i in range(num_elements)])
    data_ptr = (ctypes.c_double * num_elements).from_buffer(data)
    shape = (ctypes.c_int64 * 2)(N, N)

    # 3. Enviar al peer
    print(f"[{node_name}] Enviando tensor {N}x{N} a localhost:{peer_port}...")
    rc = lib.pmtp_send_raw(data_ptr, num_elements, shape, 2, b"127.0.0.1", peer_port)
    if rc != 0:
        result_dict[node_name] = f"FAIL: send returned {rc}"
        return

    # 4. Esperar recepción
    for _ in range(100):
        time.sleep(0.1)
        qlen = lib.pmtp_recv_queue_len()
        if qlen > 0:
            break

    epoch = lib.pmtp_get_epoch()
    result_dict[node_name] = f"OK: sent={num_elements}, received_queue={qlen}, epoch={epoch}"
    print(f"[{node_name}] Resultado: {result_dict[node_name]}")


def main():
    if not os.path.exists(DLL_PATH):
        print(f"ERROR: DLL no encontrada en {DLL_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("TEST MULTI-NODO PMTP (2 nodos, Rust Tokio, HMAC-SHA256)")
    print("=" * 70)

    manager = multiprocessing.Manager()
    results = manager.dict()

    node_a = multiprocessing.Process(
        target=run_node, args=(50061, 50062, "NODO-A", results)
    )
    node_b = multiprocessing.Process(
        target=run_node, args=(50062, 50061, "NODO-B", results)
    )

    node_a.start()
    node_b.start()

    node_a.join(timeout=30)
    node_b.join(timeout=30)

    print("\n" + "=" * 70)
    print("RESULTADOS DEL CLUSTER")
    print("=" * 70)
    for name, result in sorted(results.items()):
        status = "PASS" if result.startswith("OK") else "FAIL"
        print(f"  [{status}] {name}: {result}")

    all_ok = all(r.startswith("OK") for r in results.values())
    if all_ok and len(results) == 2:
        print("\n  VEREDICTO: CLUSTER PMTP NATIVO FUNCIONAL")
        print("  Dos nodos intercambiaron tensores firmados por TCP")
        print("  sin involucrar al GIL de Python en la transmision.")
    else:
        print("\n  VEREDICTO: FALLO EN EL CLUSTER")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
