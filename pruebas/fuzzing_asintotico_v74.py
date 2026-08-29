import sys
import os
import threading
import time
import uuid
import socket

# Añadir el path para importar el monolito
sys.path.append(r"E:\POLYDIM_EINSOF\ENTREGA_20260827_V74_")

try:
    import polydim_v74_monolito as pd74
except ImportError as e:
    print(f"Error importando V74: {e}")
    sys.exit(1)

import jax
import jax.numpy as jnp
import numpy as np

def redteam_sender_node(target_port: int, d_size: int, num_messages: int, thread_id: int):
    """
    Simula un agente atacando al orquestador con ráfagas asintóticas.
    """
    print(f"[Thread {thread_id}] Sender iniciado. Atacando con D={d_size}...")
    np.random.seed(42 + thread_id)
    
    # Pre-crear tensores
    tensors = []
    for _ in range(num_messages):
        # Generar ruido. Para D=10^7, esto es ~40MB por tensor.
        raw_arr = np.random.randn(d_size).astype(np.float32)
        tensors.append(jax.device_put(jnp.array(raw_arr)))
        
    start_time = time.time()
    
    for i, t in enumerate(tensors):
        try:
            success = pd74.PMTPAgentBridge.send_tensor("127.0.0.1", target_port, t, timeout=20.0)
            if not success:
                print(f"[Thread {thread_id}] ⚠️ Falló envío {i+1}")
        except Exception as e:
            print(f"[Thread {thread_id}] ❌ CRASH en envío: {e}")
            
    elapsed = time.time() - start_time
    print(f"[Thread {thread_id}] ✅ Ráfaga completada en {elapsed:.2f}s")


def run_asymptotic_swarm_test():
    """
    Prueba de Estrés Extremo (Red Team).
    - D = 10^7 (Tensores de alta dimensión reales).
    - 4 Agentes emitiendo ráfagas concurrentes.
    - Demuestra la inmunidad al colapso asíncrono y OOM de la V74.
    """
    print("==========================================================")
    print(" INICIANDO FUZZING ASINTÓTICO PMTP V74 (El Test del Bulldog) ")
    print("==========================================================")
    
    # Inicializar FFI Bridge para que el entorno V74 esté vivo
    pd74.NativeFFIBridge.initialize()
    
    # Crear el nodo receptor (Orquestador)
    receiver = pd74.PMTPAgentBridge(host="127.0.0.1", port=0)
    receiver.start_server()
    port = receiver.port
    print(f"[Orquestador] Escuchando en el puerto {port}")
    
    D_SIZE = 10**7 # ~40MB por tensor. En 4 hilos * 10 msgs = 1.6 GB de ancho de banda.
    NUM_MESSAGES = 10
    NUM_THREADS = 4
    
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=redteam_sender_node, args=(port, D_SIZE, NUM_MESSAGES, i))
        threads.append(t)
        t.start()
        
    # Monitor de recepción
    received_count = 0
    total_expected = NUM_THREADS * NUM_MESSAGES
    
    start_wait = time.time()
    while received_count < total_expected:
        try:
            # Poll del inbox (bloqueante con timeout)
            tensor_obj = receiver.inbox.get(timeout=2.0)
            received_count += 1
            print(f"[Orquestador] Recibió tensor {received_count}/{total_expected}. Shape: {tensor_obj.shape}, Dtype: {tensor_obj.dtype}")
        except Exception as e: # queue.Empty raise Exception in python if timeout
            # If it's a queue.Empty
            if type(e).__name__ == "Empty":
                if time.time() - start_wait > 90.0:
                    print(f"\n[ERROR] TIMEOUT CRITICO. El servidor se atasco. Recibidos: {received_count}/{total_expected}")
                    break
            else:
                print(f"[Orquestador] Crash en read: {e}")
                break
                
    for t in threads:
        t.join()
        
    receiver.stop_server()
    pd74.NativeFFIBridge.cleanup()
    
    if received_count == total_expected:
        print("\n[OK] VERTICE ALCANZADO: La V74 devoro el ataque concurrente asintotico sin OOM. Nucleo SOLIDO.")
    else:
        print("\n[FAIL] FALLO EN LA ARQUITECTURA: Se perdieron tensores o hubo colapso.")

if __name__ == "__main__":
    run_asymptotic_swarm_test()
