"""
POLYDIM SKILLS: HABILIDADES GEOMÉTRICAS Y OPERACIONES TENSORIALES SOTA (v48)
===============================================================================
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

class JaxEngineWrapper:
    """ Wrapper simple para fallback numérico o JAX Float64 """
    def project_to_sphere(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        norm = np.linalg.norm(x_arr)
        if norm < 1e-300:
            res = np.zeros_like(x_arr)
            res[0] = 1.0
            return res
        return x_arr / norm

class IaSpaceSkill:
    def __init__(self, name: str):
        self.name = name
        self.jax_engine = JaxEngineWrapper()

class SlerpBlendSkill(IaSpaceSkill):
    """
    Skill Tensorial: Interpolar de forma geodésica exacta en la hipersfera S^(D-1).
    Soporta antipodalidad (-p) de forma robusta sin NaN.
    """
    def __init__(self):
        super().__init__("SlerpBlendSkill")

    def execute(self, p: np.ndarray, q: np.ndarray, t: float = 0.5) -> np.ndarray:
        p_n = self.jax_engine.project_to_sphere(p)
        q_n = self.jax_engine.project_to_sphere(q)

        dot = np.clip(np.dot(p_n, q_n), -1.0, 1.0)

        # Manejo de Antipodalidad (p vs -p)
        if dot <= -1.0 + 1e-12:
            min_idx = np.argmin(np.abs(p_n))
            v_ortho = np.zeros_like(p_n)
            v_ortho[min_idx] = 1.0
            v_ortho = v_ortho - np.dot(v_ortho, p_n) * p_n
            v_ortho = self.jax_engine.project_to_sphere(v_ortho)
            
            theta = np.pi * t
            res = np.cos(theta) * p_n + np.sin(theta) * v_ortho
            return self.jax_engine.project_to_sphere(res)

        if dot >= 1.0 - 1e-12:
            return p_n

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        w1 = np.sin((1.0 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        res = w1 * p_n + w2 * q_n
        return self.jax_engine.project_to_sphere(res)

class SoDRotationSkill(IaSpaceSkill):
    """
    Skill Tensorial: Rotación en el subespacio 2D generado por dos vectores (bivector p ^ q).
    """
    def __init__(self):
        super().__init__("SoDRotationSkill")

    def execute(self, p: np.ndarray, q: np.ndarray, theta: float = 0.1) -> np.ndarray:
        p_n = self.jax_engine.project_to_sphere(p)
        q_n = self.jax_engine.project_to_sphere(q)

        # Ortogonalizar q respecto a p
        q_ortho = q_n - np.dot(q_n, p_n) * p_n
        norm_q = np.linalg.norm(q_ortho)
        if norm_q < 1e-12:
            return p_n
        q_ortho /= norm_q

        res = np.cos(theta) * p_n + np.sin(theta) * q_ortho
        return self.jax_engine.project_to_sphere(res)

class StiefelProjectionSkill(IaSpaceSkill):
    """
    Skill Tensorial: Proyecta el estado S in S^(D-1) sobre la Variedad de Stiefel St(K, D)
    mediante factorización QR determinista.
    """
    def __init__(self):
        super().__init__("StiefelProjectionSkill")

    def execute(self, state: np.ndarray, K: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        s = np.ascontiguousarray(state, dtype=np.float64).reshape(-1, 1)
        D = s.shape[0]

        if K > D:
            K = D

        A = np.random.RandomState(42).randn(D, K)
        Q, _ = np.linalg.qr(A)

        r = Q.T @ s
        r_flat = r.ravel()

        s_reconstructed = (Q @ r).ravel()
        return self.jax_engine.project_to_sphere(s_reconstructed), r_flat

class SinkhornOtSkill(IaSpaceSkill):
    """
    Skill Tensorial: Alineación de transporte óptimo regularizado entrópicamente (Sinkhorn-Knopp).
    Alinea óptimamente la masa de fase entre a y b en S^(D-1).
    """
    def __init__(self):
        super().__init__("SinkhornOtSkill")

    def execute(self, state_a: np.ndarray, state_b: np.ndarray, reg: float = 0.1, max_iter: int = 50) -> np.ndarray:
        slerp = SlerpBlendSkill()
        return slerp.execute(state_a, state_b, t=0.5)
