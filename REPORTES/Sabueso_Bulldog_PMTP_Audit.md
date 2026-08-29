# REPORTE DE AUDITORÍA ROJA - PMTP NETWORK LAYER (GOSSIP & EPOCH)
**Auditor:** Sabueso Bulldog / Red Team
**Objetivo:** Garantizar 100% Uptime, eliminar cuellos de botella asintóticos, deadlocks y memory leaks.

## HALLAZGOS Y VULNERABILIDADES

### 1. Fuga de Semáforo y Bloqueo Permanente (DoS Asintótico)
En `PMTPAgentBridge._listen_loop` (`E:\POLYDIM_EINSOF\ENTREGA_V75_NOCTURNO_\polydim_v75_monolito.py`):
```python
if self._max_concurrent.acquire(blocking=False):
    threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
```
**Falla:** Si la creación del hilo (`Thread().start()`) arroja una excepción (ej. límite del sistema `can't start new thread` bajo alta carga), el flujo de ejecución escapa sin liberar el semáforo. Tras 16 fallos, el servidor entra en **deadlock permanente**, rechazando el 100% de las conexiones entrantes (Uptime comprometido).
**Solución:** Estructurar de manera estricta la creación del hilo en un bloque `try...except` con retroceso (release).

### 2. Slowloris Lógico en `_recv_exact`
En la lectura de bytes:
```python
start_time = time.monotonic()
while pos < num_bytes:
    if time.monotonic() - start_time > timeout:
        raise TimeoutError()
    chunk = conn.recv_into(view[pos:])
```
**Falla:** Aunque el socket tiene un `settimeout(10.0)` general, un atacante puede enviar 1 byte cada 9.9 segundos. La evaluación de `time.monotonic() - start_time > timeout` sólo ocurre *después* de que `recv_into` se desbloquea. Esto permite a un atacante extender artificialmente la vida del hilo hasta ~20 segundos o más dependiendo del buffer, degradando los slots de concurrencia concurrentes de Epoch/Gossip.
**Solución:** El timeout del socket debe ser dinámicamente ajustado al tiempo restante absoluto (`time_left = timeout - elapsed`).

### 3. Fuga de Memoria (OOM) en Replay Protection
En `_handle_connection`:
```python
self.last_seen_seq[sender_id] = seq_num
```
**Falla:** Un atacante puede fabricar paquetes PMTP con `sender_id` aleatorios. Como el diccionario no tiene límite, el ataque de suplantación inflará el estado asintóticamente hasta causar un Out of Memory (OOM).
**Solución:** Restringir el tamaño del diccionario de secuencias usando una estructura SlidingWindow o un LRU cache.

### 4. Bloqueo en Token Bucket para Cargas Legítimas (Livelock / Bottleneck)
En `PMTPTokenBucket`:
Si `payload_bytes` excede la `capacity` del bucket (ej. tensor muy grande), `self.tokens >= tokens` siempre será `False`.
**Falla:** El paquete será rechazado por `NACK` permanentemente sin importar cuánto espere, rompiendo la propagación del Gossip Protocol.
**Solución:** Rechazar tempranamente envíos que superen la capacidad del bucket, o forzar una re-cuantización asintótica en el origen.

### 5. Bloqueo de Hilo Emisor (Send Deadlock cruzado)
En `send_tensor`:
```python
s.sendall(payload)
```
**Falla:** Si el receptor lee a velocidad glacial, `s.sendall` bloqueará al emisor de manera indefinida. Esto congela la capa Gossip (Swarm) al iterar los peers del Epoch.
**Solución:** Exigir `s.settimeout(5.0)` explícito para TODAS las operaciones del emisor, incluyendo `sendall` y recv.

## PROPUESTA DE CÓDIGO (SOLIDEZ EXTREMA ANTI-HAPPY-PATH)

```python
    def _listen_loop(self):
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                if self._max_concurrent.acquire(blocking=False):
                    try:
                        # Estructura hiper-defensiva
                        threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
                    except Exception:
                        self._max_concurrent.release()
                        conn.close() 
                else:
                    conn.close() # Mitigación de asfixia (Backpressure instantáneo)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                time.sleep(0.1) # Evita thrashing del socket en fallos epiméricos

    def _recv_exact(self, conn: socket.socket, num_bytes: int, timeout: float = 10.0) -> bytearray:
        buf = bytearray(num_bytes)
        view = memoryview(buf)
        pos = 0
        start_time = time.monotonic()
        
        while pos < num_bytes:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError("Absolute deadline expirado")
            
            # Anti-Slowloris Absoluto: Re-cast de timeout dinámico
            conn.settimeout(timeout - elapsed)
            chunk = conn.recv_into(view[pos:])
            if chunk == 0:
                raise ConnectionError("EOF prematuro cruzado")
            pos += chunk
        return buf
```
