# test_seqlock_atomicity.py
# DEMOSTRACIÓN DE ATOMICIDAD EMPÍRICA Y CERO TORN-READS EN PMTP V44 SEQLOCK (RUST 2024 C-ABI)
# PROTOCOLO BULLDOG CRITIC / LEY ARIEL (REGLA 17 - CERO AUDITORÍA PASIVA)
# ============================================================================

import os
import sys
import time
import ctypes
import threading
import numpy as np

PAYLOAD_DIM = 10000

dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pmtp_seqlock_native.dll"))
if not os.path.exists(dll_path):
    raise FileNotFoundError(f"DLL no encontrada en {dll_path}")

seqlock_lib = ctypes.CDLL(dll_path)

seqlock_lib.pmtp_seqlock_init.argtypes = []
seqlock_lib.pmtp_seqlock_init.restype = None

seqlock_lib.pmtp_seqlock_write.argtypes = [ctypes.c_double, ctypes.c_uint64]
seqlock_lib.pmtp_seqlock_write.restype = ctypes.c_uint64

seqlock_lib.pmtp_seqlock_read_lockfree.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64
]
seqlock_lib.pmtp_seqlock_read_lockfree.restype = ctypes.c_int32

seqlock_lib.pmtp_seqlock_init()

print("=" * 80)
print("  [DEMOSTRACION DE ATOMICIDAD EMPIRICA Y CERO TORN-READS EN PMTP V44 SEQLOCK]")
print("  REGLA 17: Cero Auditoria Pasiva - Multi-Hilo Rapido (8 Escritores, 16 Lectores)")
print("=" * 80)

num_writers = 8
num_readers = 16
duration_seconds = 3.0

stop_event = threading.Event()
written_count = 0
read_count = 0
torn_reads_count = 0
retry_conflicts_count = 0
lock = threading.Lock()

def writer_thread(writer_id: int):
    global written_count
    val = float(writer_id + 1)
    local_writes = 0
    ts = 0
    while not stop_event.is_set():
        ts += 1
        seqlock_lib.pmtp_seqlock_write(val, ts)
        local_writes += 1
    with lock:
        written_count += local_writes

def reader_thread(reader_id: int):
    global read_count, torn_reads_count, retry_conflicts_count
    buf = (ctypes.c_double * PAYLOAD_DIM)()
    buf_ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_double))
    ts_out = ctypes.c_uint64()
    
    local_reads = 0
    local_torn = 0
    
    while not stop_event.is_set():
        res = seqlock_lib.pmtp_seqlock_read_lockfree(buf_ptr, ctypes.byref(ts_out), 1000)
        if res == 0:
            local_reads += 1
            first_val = buf[0]
            np_arr = np.frombuffer(buf, dtype=np.float64, count=PAYLOAD_DIM)
            if not np.all(np_arr == first_val):
                local_torn += 1
        elif res == -2:
            with lock:
                retry_conflicts_count += 1
                
    with lock:
        read_count += local_reads
        torn_reads_count += local_torn

print(f"Iniciando prueba de esfuerzo: {num_writers} Escritores concurentes + {num_readers} Lectores Lock-Free por {duration_seconds} segundos...")

threads = []
for i in range(num_writers):
    t = threading.Thread(target=writer_thread, args=(i,))
    threads.append(t)

for i in range(num_readers):
    t = threading.Thread(target=reader_thread, args=(i,))
    threads.append(t)

t_start = time.perf_counter()
for t in threads:
    t.start()

time.sleep(duration_seconds)
stop_event.set()

for t in threads:
    t.join()

t_elapsed = time.perf_counter() - t_start

print("-" * 80)
print(f"  Tiempo transcurrido: {t_elapsed:.2f} segundos")
print(f"  Escrituras Totales Completadas: {written_count:,}")
print(f"  Lecturas Lock-Free Totales Exitosas: {read_count:,}")
print(f"  Throughput Combinado: {(written_count + read_count) / t_elapsed:,.0f} ops/segundo")
print(f"  Lecturas Interrumpidas Reintentadas (Conflictos de Secuencia): {retry_conflicts_count:,}")
print(f"  TORN READS DETECTADOS (Lecturas Corruptas Parciales): {torn_reads_count}")
print("=" * 80)

if torn_reads_count == 0:
    print("[OK] DEMOSTRACION EXITOSA: CERO TORN READS. EL SEQLOCK PMTP V44 ES REALMENTE ATOMICO Y RESISTE CAMBIOS DE CONTEXTO CPU.")
else:
    print(f"[FAIL] FALLO DE ATOMICIDAD: SE DETECTARON {torn_reads_count} TORN READS!")
    sys.exit(1)
