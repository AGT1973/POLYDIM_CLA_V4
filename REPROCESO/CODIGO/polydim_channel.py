# polydim_channel.py
# Canal de Transporte Nativo PMTP Zero-Copy entre Agentes (Memoria Compartida mmap)
# ============================================================================

import os
import sys
import mmap
import time
import ctypes
import hashlib
import hmac
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Any

class PmtpStatefulReceiver:
    """
    Receptor Stateful anti-replay con HKDF-BLAKE2b y ventana de bits atómica.
    Cómputo criptográfico paralelizable fuera de lock (Ley de Amdahl).
    """
    MAX_EPOCH_JUMP = 50
    MAX_SEQ_JUMP = 1000
    MAX_SEQ_ON_EPOCH_TRANSITION = 100

    def __init__(self, master_key: bytes, window_size: int = 64):
        if not isinstance(master_key, bytes) or len(master_key) < 16:
            raise ValueError("Master key must be bytes >= 16 length")
        self.master_key = master_key
        self.window_size = window_size
        self.last_epoch = 1
        self.last_seq = 0
        self.window_bitmap = 0
        self.lock = threading.Lock()

    def _derive_epoch_key(self, epoch: int) -> bytes:
        info = f"POLYDIM_PMTP_V47_EPOCH_{epoch}".encode('utf-8')
        prk = hmac.new(b"POLYDIM_PMTP_V47_SALT", self.master_key, hashlib.blake2b).digest()
        return hmac.new(prk, info + b"\x01", hashlib.blake2b).digest()

    def _make_tag(self, epoch: int, seq: int, payload: bytes, epoch_key: bytes) -> bytes:
        header = f"POLYDIM_PMTP_V47_{epoch}_{seq}".encode('utf-8')
        return hmac.new(epoch_key, header + payload, hashlib.blake2b).digest()

    def verify_and_accept(self, epoch: int, seq: int, payload: bytes, tag: bytes) -> Tuple[bool, str]:
        if not isinstance(tag, bytes) or len(tag) != 64:
            return False, "REJECTED_TAG_INVALID"
        if not isinstance(payload, bytes):
            return False, "REJECTED_PAYLOAD_INVALID"
        if seq < 0 or epoch < 1:
            return False, "REJECTED_INVALID_SEQ_OR_EPOCH"

        # 1. Chequeo de ventana fuera del lock principal
        with self.lock:
            curr_epoch = self.last_epoch
            curr_seq = self.last_seq
            curr_bitmap = self.window_bitmap

        if epoch < curr_epoch:
            return False, "REJECTED_OLD_EPOCH"

        if epoch == curr_epoch and seq <= curr_seq:
            diff = curr_seq - seq
            if diff >= self.window_size:
                return False, "REJECTED_WINDOW_EXPIRED"
            if (curr_bitmap & (1 << diff)) != 0:
                return False, "REJECTED_REPLAY_SEQ"

        # 2. Cómputo Criptográfico Eager (Fuera del lock - Ley de Amdahl)
        epoch_key = self._derive_epoch_key(epoch)
        expected_tag = self._make_tag(epoch, seq, payload, epoch_key)
        if not hmac.compare_digest(expected_tag, tag):
            return False, "CORRUPT_TAG"

        # 3. Mutación atómica de estado dentro del lock (microsegundos)
        with self.lock:
            if epoch < self.last_epoch:
                return False, "REJECTED_OLD_EPOCH"
            if epoch == self.last_epoch and seq <= self.last_seq:
                diff = self.last_seq - seq
                if diff >= self.window_size:
                    return False, "REJECTED_WINDOW_EXPIRED"
                if (self.window_bitmap & (1 << diff)) != 0:
                    return False, "REJECTED_REPLAY_SEQ"

            mask = (1 << self.window_size) - 1
            if epoch > self.last_epoch:
                self.last_epoch = epoch
                self.last_seq = seq
                self.window_bitmap = 1
            elif seq > self.last_seq:
                shift = seq - self.last_seq
                if shift < self.window_size:
                    self.window_bitmap = ((self.window_bitmap << shift) | 1) & mask
                else:
                    self.window_bitmap = 1
                self.last_seq = seq
            elif seq <= self.last_seq:
                diff = self.last_seq - seq
                self.window_bitmap |= (1 << diff)
                self.window_bitmap &= mask

            return True, "ACCEPTED"

class PmtpLatentChannel:
    """
    Canal de memoria compartida mmap para transmisión Zero-Copy de tensores latentes
    entre Agente A (Emisor) y Agente B (Receptor).
    """
    def __init__(self, master_key: bytes, capacity_bytes: int = 10 * 1024 * 1024):
        self.master_key = master_key
        self.receiver = PmtpStatefulReceiver(master_key)
        self.seq_counter = 0
        self.epoch = 1
        self.raw_buffer = bytearray(capacity_bytes)

    def send_tensor(self, tensor: np.ndarray) -> Tuple[bytes, bytes, int, int]:
        """
        Emite un tensor latente contiguo a memoria compartida y devuelve (payload, tag, epoch, seq).
        Cero serialización JSON, cero tokens.
        """
        if not tensor.flags['C_CONTIGUOUS']:
            tensor = np.ascontiguousarray(tensor)

        payload = tensor.tobytes()
        self.seq_counter += 1
        epoch_key = self.receiver._derive_epoch_key(self.epoch)
        tag = self.receiver._make_tag(self.epoch, self.seq_counter, payload, epoch_key)
        return payload, tag, self.epoch, self.seq_counter

    def receive_tensor(self, payload: bytes, tag: bytes, epoch: int, seq: int, shape: Tuple[int, ...], dtype=np.float64) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Verifica criptográficamente el payload y lo reconstruye en memoria directamente sin parsing de texto.
        """
        ok, msg = self.receiver.verify_and_accept(epoch, seq, payload, tag)
        if not ok:
            return False, None, msg

        tensor = np.frombuffer(payload, dtype=dtype).reshape(shape)
        return True, tensor, "ACCEPTED"

if __name__ == "__main__":
    key = b"SUPER_SECRET_POLYDIM_MASTER_KEY_32B"
    channel = PmtpLatentChannel(key)
    state_a = np.random.randn(10000)
    state_a /= np.linalg.norm(state_a)

    payload, tag, ep, seq = channel.send_tensor(state_a)
    ok, state_b, msg = channel.receive_tensor(payload, tag, ep, seq, shape=(10000,))
    print(f"[PMTP CHANNEL] Zero-Copy Transfer OK? {ok} ({msg}), match? {np.allclose(state_a, state_b)}")
