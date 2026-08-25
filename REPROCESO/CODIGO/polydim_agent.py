# polydim_agent.py
# Agente Nativo en Espacio IA (S^(D-1), D >= 4096)
# Implementa la Tupla del Programa Cognitivo: P = (S, G, T, O, C, Pi)
# ============================================================================

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from polydim_skills import SlerpBlendSkill, SoDRotationSkill, IaSpaceSkill

class IaSpaceAgent:
    """
    Agente del Espacio IA en POLYDIM.
    Sostiene el estado cognitivo S in S^(D-1) y delibera aplicando transformaciones T in Math_Space
    para aproximarse al Objetivo G preservando la invarianza ||S|| = 1.0 (Restricción C).
    """
    def __init__(self, agent_id: str, dim: int = 10000, dtype=np.float64):
        self.agent_id = agent_id
        self.dim = dim
        self.dtype = dtype

        # Estado Cognitivo S in S^(D-1) (Inicializado aleatoriamente y normalizado)
        raw_s = np.random.randn(dim)
        self.state = raw_s / np.linalg.norm(raw_s)

        # Goal G in S^(D-1) (Objetivo latente)
        self.goal: Optional[np.ndarray] = None

        # Catálogo de Skills Nativas Compiladas (Cero tokens LLM)
        self.skills: Dict[str, IaSpaceSkill] = {
            "slerp": SlerpBlendSkill(),
            "sod_rot": SoDRotationSkill()
        }

    def set_goal(self, goal_state: np.ndarray):
        """
        Define el Objetivo G en la esfera unitaria S^(D-1).
        """
        g = np.asarray(goal_state, dtype=self.dtype).ravel()
        norm = np.linalg.norm(g)
        self.goal = g / (norm if norm > 1e-15 else 1.0)

    def geodesic_distance_to_goal(self) -> float:
        """
        Calcula la distancia geodésica Kahan omega = 2 * atan2(||S - G||, ||S + G||) en S^(D-1).
        """
        if self.goal is None:
            return np.pi

        d_norm = np.linalg.norm(self.state - self.goal)
        s_norm = np.linalg.norm(self.state + self.goal)
        omega = 2.0 * np.arctan2(d_norm, s_norm)
        return float(omega)

    def deliberate_step(self, step_size: float = 0.2) -> Tuple[float, str]:
        """
        Política Pi: Evalúa el estado S respecto al objetivo G y aplica una transformación T (SLERP / SO(D))
        para reducir la distancia geodésica en el Espacio IA sin pasar por texto.
        """
        if self.goal is None:
            return np.pi, "NO_GOAL"

        dist_before = self.geodesic_distance_to_goal()
        if dist_before < 1e-6:
            return dist_before, "CONVERGED"

        # Aplicación de Skill Tensorial SLERP hacia el Objetivo G
        slerp_skill = self.skills["slerp"]
        new_state = slerp_skill.execute(self.state, self.goal, t=step_size)

        # Verificación de Restricción C (Invarianza de Norma)
        norm_err = abs(np.linalg.norm(new_state) - 1.0)
        assert norm_err < 1e-12, f"Violación de Restricción C: norm_err={norm_err}"

        self.state = new_state
        dist_after = self.geodesic_distance_to_goal()
        return dist_after, f"STEP_OK (dist={dist_after:.4f})"

if __name__ == "__main__":
    agent = IaSpaceAgent("Agent_Alpha", dim=10000)
    goal = np.random.randn(10000); goal /= np.linalg.norm(goal)
    agent.set_goal(goal)

    print(f"[AGENT] Initial geodesic distance to goal: {agent.geodesic_distance_to_goal():.4f} rad")
    for step in range(5):
        dist, status = agent.deliberate_step(step_size=0.3)
        print(f"  Step {step+1}: dist={dist:.4f} rad ({status})")
