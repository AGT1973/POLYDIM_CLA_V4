import os
import re

V63_MONOLITO = 'polydim_v63_monolito.py'

def read_file(f):
    with open(f, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(f, content):
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

# -----------------------------------------------------------------------------
# PIECE 5: FFI Bridge
# -----------------------------------------------------------------------------
FFI_BRIDGE_CODE = '''
# ------------------------------------------------------------------------------
# PIECE 5: NATIVE FFI BRIDGE (C++ AVX-512 & RUST C-ABI)
# ------------------------------------------------------------------------------
class NativeFFIBridge:
    _cpp_dll = None
    _rust_dll = None

    @classmethod
    def initialize(cls):
        """Extrae, compila (si es necesario) y carga las DLLs nativas."""
        import subprocess
        import sys
        
        # Guardar fuentes
        with open("polydim_cpp_kernel.cpp", "w") as f: f.write(CPP_SOURCE)
        with open("polydim_rust_kernel.rs", "w") as f: f.write(RUST_SOURCE)
        
        # Cargar o Compilar C++
        if not os.path.exists("polydim_cpp_kernel.dll"):
            vcvars = r"C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
            cmd = f'"{vcvars}" && cl.exe /LD /EHsc polydim_cpp_kernel.cpp'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
        
        # Cargar o Compilar Rust
        if not os.path.exists("polydim_rust_kernel.dll"):
            subprocess.run(["rustc", "--crate-type", "cdylib", "polydim_rust_kernel.rs"], stdout=subprocess.DEVNULL)
            
        cls._cpp_dll = ctypes.CDLL(os.path.abspath("polydim_cpp_kernel.dll"))
        cls._rust_dll = ctypes.CDLL(os.path.abspath("polydim_rust_kernel.dll"))
        
        # Firmas C++
        cls._cpp_dll.polydim_cpp_householder_reflect.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        ]
        
    @classmethod
    def householder_reflect_cpp(cls, x_np, v_np):
        dim = len(x_np)
        out_np = np.zeros_like(x_np)
        
        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        v_ptr = v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        cls._cpp_dll.polydim_cpp_householder_reflect(x_ptr, v_ptr, out_ptr, dim)
        return out_np
'''

# -----------------------------------------------------------------------------
# PIECE 2: PERSISTENCE
# -----------------------------------------------------------------------------
PERSISTENCE_CODE = '''
# ------------------------------------------------------------------------------
# PIECE 2: PMTP PERSISTENCE (DISK I/O)
# ------------------------------------------------------------------------------
class PMTPPersistentStorage:
    @staticmethod
    def save_tensor(path: str, tensor: np.ndarray, metadata_generation: int = 1):
        """Serializa de forma pura un tensor ND al disco usando el Header PMTP (C-ABI)."""
        dim = tensor.shape[-1] if len(tensor.shape) > 0 else 1
        # dtype code: 1=float32, 2=float64
        dtype_code = 2 if tensor.dtype == np.float64 else 1
        
        header = struct.pack(
            "<QQIIIIQQ16s",
            0,                  # seq_word
            0x504F4C5944494D34, # MAGIC "POLYDIM4"
            57,                 # version
            dim,
            dtype_code,
            tensor.nbytes,
            int(time.time_ns()),
            metadata_generation,
            b'\\x00' * 16         # reserved
        )
        with open(path, "wb") as f:
            f.write(header)
            f.write(tensor.tobytes())
            
    @staticmethod
    def load_tensor(path: str) -> np.ndarray:
        """Carga un tensor ND desde disco validando el Header PMTP."""
        with open(path, "rb") as f:
            header_bytes = f.read(64)
            if len(header_bytes) < 64:
                raise ValueError("Archivo demasiado corto para ser PMTP")
            fields = struct.unpack("<QQIIIIQQ16s", header_bytes)
            
            magic = fields[1]
            if magic != 0x504F4C5944494D34:
                raise ValueError("Magic PMTP incorrecto")
                
            dim = fields[3]
            dtype_code = fields[4]
            payload_bytes = fields[5]
            
            payload = f.read(payload_bytes)
            dtype_str = '<f8' if dtype_code == 2 else '<f4'
            return np.frombuffer(payload, dtype=dtype_str).reshape(-1)
'''

# -----------------------------------------------------------------------------
# PIECE 1 & 7: NETWORK TRANSPORT & AGENT BRIDGE
# -----------------------------------------------------------------------------
NETWORK_CODE = '''
# ------------------------------------------------------------------------------
# PIECE 1 & 7: PMTP NETWORK TRANSPORT & AGENT PROTOCOL
# ------------------------------------------------------------------------------
import socket
import threading

class PMTPAgentBridge:
    """
    Protocolo de Agente a Agente (Zero-JSON, Nativo).
    Permite enviar y recibir tensores S^{D-1} sobre TCP/IP sin colapsar a 1D/Texto.
    """
    def __init__(self, host='127.0.0.1', port=50051):
        self.host = host
        self.port = port
        self.server_socket = None
        self._running = False
        self.inbox = []
        
    def start_listening(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self._running = True
        
        def listener():
            while self._running:
                try:
                    conn, addr = self.server_socket.accept()
                    header_bytes = conn.recv(64)
                    if len(header_bytes) == 64:
                        fields = struct.unpack("<QQIIIIQQ16s", header_bytes)
                        payload_size = fields[5]
                        dtype_code = fields[4]
                        
                        payload = bytearray()
                        while len(payload) < payload_size:
                            packet = conn.recv(payload_size - len(payload))
                            if not packet: break
                            payload.extend(packet)
                            
                        dtype_str = '<f8' if dtype_code == 2 else '<f4'
                        tensor = np.frombuffer(payload, dtype=dtype_str)
                        self.inbox.append(tensor)
                    conn.close()
                except Exception:
                    pass
        threading.Thread(target=listener, daemon=True).start()
        
    def send_latent(self, target_host: str, target_port: int, tensor: np.ndarray):
        """Transfiere un tensor nativo a otro agente en la red."""
        dim = tensor.shape[-1] if len(tensor.shape) > 0 else 1
        dtype_code = 2 if tensor.dtype == np.float64 else 1
        
        header = struct.pack(
            "<QQIIIIQQ16s", 0, 0x504F4C5944494D34, 57, dim, dtype_code, tensor.nbytes, int(time.time_ns()), 1, b'\\x00' * 16
        )
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((target_host, target_port))
            s.sendall(header)
            s.sendall(tensor.tobytes())
            
    def stop(self):
        self._running = False
        if self.server_socket:
            self.server_socket.close()
'''

content = read_file(V63_MONOLITO)

# Insert the code blocks right before the verification suite
verification_marker = "# SUITE DE VERIFICACIÓN AUTÓNOMA EN CALIENTE"

if verification_marker in content:
    idx = content.find(verification_marker)
    new_content = content[:idx] + FFI_BRIDGE_CODE + PERSISTENCE_CODE + NETWORK_CODE + "\\n\\n" + content[idx:]
    write_file(V63_MONOLITO, new_content)
    print(f"Piezas insertadas con éxito en {V63_MONOLITO}")
else:
    print("Marker no encontrado")

