import os

with open("E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/scratch_v79.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace PMTP class
import re

old_pmtp = """class PMTPNetworkLayer:
    def __init__(self, node_id: str):
        if not PMTP_NET_KEY or len(PMTP_NET_KEY) != 32:
            raise RuntimeError("Clave PMTP debe ser exactamente 32 bytes.")
        self.node_id = node_id.encode("utf-8").ljust(16, b"\\x00")[:16]
        self.seq_num = 0
        self.boot_id = os.urandom(8)
        self._seq_lock = threading.Lock()

    def pack_tensor_header(self, tensor_shape: tuple, payload_bytes: int, receiver_id: bytes) -> bytes:
        if payload_bytes > 100 * 1024 * 1024:
            raise ValueError("Anti-DoS: Payload > 100MB")
            
        with self._seq_lock:
            self.seq_num += 1
            seq = self.seq_num

        shape_len = len(tensor_shape)
        if shape_len > 5:
            raise ValueError("PMTP Error: Máximo 5 dimensiones.")

        padded_shape = list(tensor_shape) + [0] * (5 - shape_len)
        ts = time.monotonic()

        header_raw = struct.pack(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, 44, 1, 1, shape_len, payload_bytes,
            self.node_id, receiver_id.ljust(16, b"\\x00")[:16], seq, 
            int.from_bytes(self.boot_id, "little"), ts,
            *padded_shape
        )

        mac = hmac.new(PMTP_NET_KEY, header_raw, hashlib.sha256).digest()[:16]
        return header_raw + mac

    def unpack_and_verify(self, header_bytes: bytes, expected_receiver: bytes) -> tuple:
        if len(header_bytes) != 128:
            raise ValueError(f"Longitud de header PMTP inválida: {len(header_bytes)}")
            
        header_raw = header_bytes[:112]
        mac_received = header_bytes[112:]
        mac_calc = hmac.new(PMTP_NET_KEY, header_raw, hashlib.sha256).digest()[:16]
        
        if not hmac.compare_digest(mac_calc, mac_received):
            raise ValueError("Firma HMAC-SHA256 rechazada.")
            
        unpacked = struct.unpack(PMTP_HEADER_FMT, header_raw)
        magic, ver, tipo, res, ndim, p_bytes, sender, rec, seq, boot, ts = unpacked[:11]
        
        if magic != PMTP_MAGIC or ver != 44:
            raise ValueError("Magic/Version mismatch.")
        if abs(time.monotonic() - ts) > 60.0:
            raise ValueError("Replay/Drift timeout")
        if p_bytes > 100 * 1024 * 1024:
            raise ValueError("Anti-DoS: Payload > 100MB")
            
        shape = unpacked[11:11+ndim]
        return sender.strip(b"\\x00"), p_bytes, shape"""

new_pmtp = """class PMTPNetworkLayer:
    def __init__(self, node_id: str):
        if not PMTP_NET_KEY or len(PMTP_NET_KEY) != 32:
            raise RuntimeError("Clave PMTP debe ser exactamente 32 bytes.")
        self.node_id = node_id.encode("utf-8").ljust(16, b"\\x00")[:16]
        self.seq_num = 0
        self.boot_id = os.urandom(8)
        self._seq_lock = threading.Lock()

    def pack_tensor_header(self, tensor_shape: tuple, payload_bytes: int, receiver_id: bytes) -> bytes:
        if payload_bytes > 100 * 1024 * 1024:
            raise ValueError("Anti-DoS: Payload > 100MB")
            
        with self._seq_lock:
            self.seq_num += 1
            seq = self.seq_num

        shape_len = len(tensor_shape)
        if shape_len > 5:
            raise ValueError("PMTP Error: Máximo 5 dimensiones.")

        padded_shape = list(tensor_shape) + [0] * (5 - shape_len)
        ts = time.monotonic()

        header_raw = struct.pack(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, 44, 1, 1, shape_len, payload_bytes,
            self.node_id, receiver_id.ljust(16, b"\\x00")[:16], seq, 
            int.from_bytes(self.boot_id, "little"), ts,
            *padded_shape,
            b"\\x00"*16
        )

        mac = hmac.new(PMTP_NET_KEY, header_raw[:112], hashlib.sha256).digest()[:16]
        return header_raw[:112] + mac

    def unpack_and_verify(self, header_bytes: bytes, expected_receiver: bytes) -> tuple:
        if len(header_bytes) != 128:
            raise ValueError(f"Longitud de header PMTP inválida: {len(header_bytes)}")
            
        unpacked = struct.unpack(PMTP_HEADER_FMT, header_bytes)
        
        header_raw = header_bytes[:112]
        mac_received = unpacked[-1]
        mac_calc = hmac.new(PMTP_NET_KEY, header_raw, hashlib.sha256).digest()[:16]
        
        if not hmac.compare_digest(mac_calc, mac_received):
            raise ValueError("Firma HMAC-SHA256 rechazada.")
            
        magic, ver, tipo, res, ndim, p_bytes, sender, rec, seq, boot, ts = unpacked[:11]
        
        if magic != PMTP_MAGIC or ver != 44:
            raise ValueError("Magic/Version mismatch.")
        if abs(time.monotonic() - ts) > 60.0:
            raise ValueError("Replay/Drift timeout")
        if p_bytes > 100 * 1024 * 1024:
            raise ValueError("Anti-DoS: Payload > 100MB")
            
        shape = unpacked[11:11+ndim]
        return sender.strip(b"\\x00"), p_bytes, shape"""

code = code.replace(old_pmtp, new_pmtp)

with open("E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/scratch_v79.py", "w", encoding="utf-8") as f:
    f.write(code)
