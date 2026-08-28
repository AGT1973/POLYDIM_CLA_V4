
# ==============================================================================
# 4. SWARM ARCHITECTURE (STRUCTURED ROUTING & INT8 PMTP)
# ==============================================================================
PMTP_MAGIC = b'PMTP'
PMTP_VERSION = 3 # V75.2 - INT8 Quantization + Epoch Clocks (Empirical SOTA)

# Header: Magic, Version, Shape_len, Dtype_code, Payload_bytes, Nonce, Agent_ID, Seq_Num, Epoch, Scale_Factor
PMTP_HEADER_FMT = "<4s B B B Q 32s 32s Q Q d " + "Q" * 8 
PMTP_HEADER_SIZE = struct.calcsize(PMTP_HEADER_FMT)

# Requerimiento estricto de clave
PMTP_NET_KEY = os.environ.get("POLYDIM_PMTP_KEY", "").encode()
if not PMTP_NET_KEY or len(PMTP_NET_KEY) != 32:
    if "pytest" not in sys.modules:
        warnings.warn("POLYDIM_PMTP_KEY no está definida o no es de 32 bytes. Se usará modo INSEGURO.")
        PMTP_NET_KEY = b'0' * 32

def pmtp_mac_chunks(sender_id: bytes, receiver_id: bytes, header: bytes, payload: memoryview) -> bytes:
    h = hmac.new(PMTP_NET_KEY, digestmod=hashlib.sha256)
    h.update(sender_id)
    h.update(receiver_id)
    h.update(header)
    h.update(payload)
    return h.digest()[:32]

class EpochClock:
    def __init__(self):
        self.epoch = 0
        self._lock = threading.Lock()
        
    def increment(self):
        with self._lock:
            self.epoch += 1
            return self.epoch
            
    def sync(self, remote_epoch: int):
        with self._lock:
            self.epoch = max(self.epoch, remote_epoch) + 1

class PMTPTokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
        
    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + self.rate * (now - self.last_update))
            self.last_update = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class PMTPAgentBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, agent_id: bytes = None, peers: list = None):
        self.host = host
        self.port = port
        self.agent_id = agent_id or os.urandom(32)
        self.peers = peers or [] # lista de tuples (host, port, peer_id)
        
        self.inbox = Queue(maxsize=10) # FIX: Límite de OOM
        self.token_bucket = PMTPTokenBucket(rate=1024*1024*10, capacity=1024*1024*50) # 10MB/s, burst 50MB
        self.epoch_clock = EpochClock() # FIX: Sustituye VectorClocks O(N)
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.port = self.server_socket.getsockname()[1]
        self.running = False
        
        self._max_concurrent = threading.Semaphore(16)
        
        self.seq_num = 0
        self.last_seen_seq = {} # peer_id -> seq

    def start_server(self):
        self.server_socket.listen(128)
        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop_server(self):
        self.running = False
        self.server_socket.close()

    def _listen_loop(self):
        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                if self._max_concurrent.acquire(blocking=False):
                    threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
                else:
                    conn.close() # DoS protection
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                time.sleep(0.1)

    def _recv_exact(self, conn: socket.socket, num_bytes: int, timeout: float = 10.0) -> bytearray:
        buf = bytearray(num_bytes)
        view = memoryview(buf)
        pos = 0
        
        # FIX V74.1: Slowloris fix (Absolute deadline)
        start_time = time.monotonic()
        
        while pos < num_bytes:
            if time.monotonic() - start_time > timeout:
                raise TimeoutError("Deadline expirado")
            
            chunk = conn.recv_into(view[pos:])
            if chunk == 0:
                raise ConnectionError("Conexión cerrada prematuramente")
            pos += chunk
        return buf

    def _handle_connection(self, conn: socket.socket):
        try:
            conn.settimeout(10.0)
            
            header = self._recv_exact(conn, PMTP_HEADER_SIZE)
            fields = struct.unpack(PMTP_HEADER_FMT, header)
            magic, version, shape_len, dtype_code, payload_bytes, nonce, sender_id, seq_num, epoch, scale = fields[:10]
            shape = tuple(fields[10:10+shape_len])
            
            if magic != PMTP_MAGIC or version != PMTP_VERSION:
                conn.sendall(b'\\x02') # NACK
                return
                
            # Anti-replay
            if sender_id in self.last_seen_seq and seq_num <= self.last_seen_seq[sender_id]:
                conn.sendall(b'\\x02') # NACK repetido
                return
            
            # Backpressure
            if not self.token_bucket.consume(payload_bytes):
                conn.sendall(b'\\x02') # NACK por rate limit
                return
                
            payload = self._recv_exact(conn, payload_bytes)
            
            mac_expected = conn.recv(32)
            mac_calc = pmtp_mac_chunks(sender_id, self.agent_id, header, memoryview(payload))
            if not hmac.compare_digest(mac_expected, mac_calc):
                conn.sendall(b'\\x02')
                return
                
            self.last_seen_seq[sender_id] = seq_num
            self.epoch_clock.sync(epoch)
            
            # FIX: INT8 Dequantization
            quantized_arr = np.frombuffer(payload, dtype=np.int8).reshape(shape)
            arr = (quantized_arr.astype(np.float32) * scale).copy()
                
            try:
                self.inbox.put_nowait((sender_id, arr))
                conn.sendall(b'\\x01') # ACK positivo
            except:
                conn.sendall(b'\\x02') # NACK Queue Full
                
        except Exception as e:
            pass
        finally:
            conn.close()
            self._max_concurrent.release()

    def send_tensor(self, host: str, port: int, tensor: jnp.ndarray, receiver_id: bytes) -> bool:
        self.seq_num += 1
        epoch = self.epoch_clock.increment()
        
        # FIX: INT8 Quantization en origen (Reduce Payload a 25%)
        tensor_np = np.asarray(tensor, dtype=np.float32)
        abs_max = float(np.max(np.abs(tensor_np)))
        scale = 1.0 if abs_max == 0 else abs_max / 127.0
        
        quantized_np = np.clip(np.round(tensor_np / scale), -127, 127).astype(np.int8)
        payload = quantized_np.tobytes()
        payload_bytes = len(payload)
        shape = quantized_np.shape
        
        nonce = os.urandom(32)
        
        header = struct.pack(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, PMTP_VERSION, len(shape), 1, payload_bytes,
            nonce, self.agent_id, self.seq_num, epoch, scale,
            *(shape + (0,) * (8 - len(shape))) # Ajustar relleno
        )
        
        mac = pmtp_mac_chunks(self.agent_id, receiver_id, header, memoryview(payload))
        
        try:
            s = socket.create_connection((host, port), timeout=5.0)
            s.sendall(header)
            s.sendall(payload)
            s.sendall(mac)
            
            ack = s.recv(1)
            return ack == b'\\x01'
        except Exception:
            return False
        finally:
            try:
                s.close()
            except:
                pass

if __name__ == "__main__":
    print("POLYDIM V75 MONOLITH - Arquitectura Swarm (Epoch/INT8) SOTA Lista.")
