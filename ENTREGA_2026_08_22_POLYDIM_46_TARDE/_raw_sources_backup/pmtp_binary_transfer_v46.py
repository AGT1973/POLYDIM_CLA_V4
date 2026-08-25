"""
POLYDIM V45 - TRANSFERENCIA BINARIA EXACTA SOBRE TENSORES PMTP
====================================================================
Demostración empírica de transferencia zero-loss de archivos binarios 
(imágenes, ejecutables, video) empaquetados como tensores Float64 
sobre el protocolo PMTP sin serialización a JSON/texto.
"""

import os
import sys
import time
import struct
import hashlib
import mmap
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polydim_motor_v45 as mj

HEADER_SIZE = 256
SLOT_COUNT = 8

def generate_test_media() -> bytes:
    # Genera un archivo "multimedia" binario simulado (ej. 1MB de ruido incompresible y patrones)
    np.random.seed(42)
    binary_data = np.random.bytes(1024 * 1024) # 1 MB
    return binary_data

def run_binary_transfer_proof():
    print("=" * 80)
    print("  POLYDIM V45 — ZERO-LOSS MEDIA TRANSFER VIA PMTP TENSORS")
    print("=" * 80)

    # 1. Preparar archivo de prueba
    original_media = generate_test_media()
    original_md5 = hashlib.md5(original_media).hexdigest()
    print(f"[*] Archivo Original (1 MB) generado. MD5: {original_md5}")

    # 2. Convertir binario crudo a Tensor Float64 Nativo
    pad_len = (8 - (len(original_media) % 8)) % 8
    padded_media = original_media + bytes([0]*pad_len)
    
    tensor_payload = np.frombuffer(padded_media, dtype=np.float64)
    DIMENSION = len(tensor_payload)
    print(f"[*] Transformado a Tensor Nativo Float64. Dimensión (D): {DIMENSION}")
    
    SLOT_SIZE = DIMENSION * 8
    TOTAL_MMAP_SIZE = HEADER_SIZE + SLOT_COUNT * SLOT_SIZE
    
    # 3. Inicializar PMTP Shared Memory
    shm = mmap.mmap(-1, TOTAL_MMAP_SIZE)
    master_key = os.urandom(32)
    salt = os.urandom(64)
    receiver = mj.PmtpStatefulReceiver(master_key, window_size=64, salt=salt)
    
    sync_event = threading.Event()
    reconstructed_bytes = bytearray()
    
    # AGENTE A (EMISOR)
    def agent_a_emitter():
        t0 = time.perf_counter_ns()
        epoch, seq = 1, 1
        
        payload_bytes = tensor_payload.tobytes()
        epoch_key = receiver._derive_epoch_key(epoch)
        tag = receiver._make_tag(epoch, seq, payload_bytes, epoch_key)
        
        slot_idx = seq % SLOT_COUNT
        offset = HEADER_SIZE + slot_idx * SLOT_SIZE
        
        struct.pack_into("<Q", shm, 0, seq)
        struct.pack_into("<Q", shm, 64, epoch)
        shm[128:192] = tag
        shm[offset : offset + SLOT_SIZE] = payload_bytes
        struct.pack_into("<Q", shm, 192, seq)
        
        t1 = time.perf_counter_ns()
        print(f"[*] AGENTE A: Tensor escrito en memoria compartida. Tiempo de inyección: {(t1-t0)/1000:.2f} µs")
        sync_event.set()

    # AGENTE B (RECEPTOR)
    def agent_b_receiver():
        sync_event.wait()
        t0 = time.perf_counter_ns()
        
        seq = struct.unpack_from("<Q", shm, 0)[0]
        epoch = struct.unpack_from("<Q", shm, 64)[0]
        tag = bytes(shm[128:192])
        
        slot_idx = seq % SLOT_COUNT
        offset = HEADER_SIZE + slot_idx * SLOT_SIZE
        payload_raw = bytes(shm[offset : offset + SLOT_SIZE])
        
        post_seq = struct.unpack_from("<Q", shm, 192)[0]
        
        if seq == post_seq:
            accepted, reason = receiver.verify_and_accept(epoch, seq, payload_raw, tag)
            if accepted:
                recovered_tensor = np.frombuffer(payload_raw, dtype=np.float64)
                recovered_bytes = recovered_tensor.tobytes()
                reconstructed_bytes.extend(recovered_bytes[:len(original_media)])
                
                t1 = time.perf_counter_ns()
                print(f"[*] AGENTE B: Tensor extraído y decodificado. Tiempo de extracción: {(t1-t0)/1000:.2f} µs")

    ta = threading.Thread(target=agent_a_emitter)
    tb = threading.Thread(target=agent_b_receiver)
    ta.start(); tb.start()
    ta.join(); tb.join()
    
    # 4. Auditoría de Cero Pérdidas
    recovered_md5 = hashlib.md5(reconstructed_bytes).hexdigest()
    print(f"[*] Archivo Recuperado (1 MB). MD5: {recovered_md5}")
    
    print("-" * 80)
    if original_md5 == recovered_md5:
        print("  >>> ÉXITO: TRANSFERENCIA DE MEDIOS PERFECTA (ZERO-LOSS) VÍA TENSORES PMTP <<<")
        print("  Ningún bit se corrompió. Ausencia total de colapso a 1D (No JSON, No String).")
    else:
        print("  >>> FALLA DE INTEGRIDAD <<<")
    print("=" * 80)

if __name__ == "__main__":
    run_binary_transfer_proof()
