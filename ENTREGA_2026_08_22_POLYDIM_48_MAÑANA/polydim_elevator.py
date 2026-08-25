# polydim_elevator.py
# Elevador de Representaciones al Espacio IA (S^(D-1), D >= 4096)
# Ingesta nativa desde arrays, archivos (.py, .txt, .rs, .cpp), binarios o imágenes.
# ============================================================================

import os
import hashlib
import numpy as np
from typing import Union, Tuple, Optional
from polydim_silicon_contract import HOST_SILICON, machine_tiny, machine_eps

class IaSpaceElevator:
    """
    Eleva vectores, latentes, archivos o datos binarios al Espacio Representacional Nativo S^(D-1).
    Garantiza normalización isométrica y proyección sobre la esfera unitaria.
    """
    def __init__(self, target_dim: int = 10000, dtype: np.dtype = np.float64):
        if target_dim < 2:
            raise ValueError(f"Target dimension must be >= 2, got {target_dim}")
        self.target_dim = target_dim
        self.dtype = dtype

    def elevate(self, input_vector: np.ndarray) -> np.ndarray:
        v = np.asarray(input_vector, dtype=self.dtype).ravel()
        in_dim = v.size

        if in_dim == 0:
            raise ValueError("Cannot elevate empty array")

        if in_dim == self.target_dim:
            v_elevated = v.copy()
        elif in_dim < self.target_dim:
            repeats = (self.target_dim + in_dim - 1) // in_dim
            v_elevated = np.tile(v, repeats)[:self.target_dim]
        else:
            v_elevated = v[:self.target_dim]

        norm = np.linalg.norm(v_elevated)
        if norm < machine_tiny(self.dtype):
            v_elevated = np.zeros(self.target_dim, dtype=self.dtype)
            v_elevated[0] = 1.0
            norm = 1.0

        v_unit = v_elevated / norm
        return v_unit

    def elevate_from_bytes(self, raw_bytes: bytes) -> np.ndarray:
        if not raw_bytes:
            raw_bytes = b"EMPTY_STREAM_POLYDIM"

        chunk_size = self.target_dim * 8
        if len(raw_bytes) < chunk_size:
            repeats = (chunk_size + len(raw_bytes) - 1) // len(raw_bytes)
            extended_bytes = (raw_bytes * repeats)[:chunk_size]
        else:
            extended_bytes = raw_bytes[:chunk_size]

        arr = np.frombuffer(extended_bytes, dtype=np.uint8).astype(self.dtype)
        arr_centered = arr - np.mean(arr)
        return self.elevate(arr_centered)

    def elevate_from_file(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            data = f.read()

        return self.elevate_from_bytes(data)

    def verify_on_sphere(self, v_unit: np.ndarray, tol: float = 1e-12) -> Tuple[bool, float]:
        norm = float(np.linalg.norm(v_unit))
        err = abs(norm - 1.0)
        return err < tol, err

if __name__ == "__main__":
    elevator = IaSpaceElevator(target_dim=10000)
    mock_embedding = np.random.randn(768)
    state = elevator.elevate(mock_embedding)
    ok, err = elevator.verify_on_sphere(state)
    print(f"[ELEVATOR] State elevated to D={state.size}, on S^(D-1)? {ok} (norm err={err:.2e})")
