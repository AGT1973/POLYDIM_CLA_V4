"""
einsof_jax — Capa de investigación y GPU/TPU
Invariantes certificados: CHK_01, 07, 08, 09, 10, 11, 12, 15
"""
from .slerp import slerp_stable, slerp_batch
from .tsqr import tsqr_blocked, cholesky_qr2
from .stiefel import project_stiefel, stiefel_drift_check

__version__ = "40.0.0"
__all__ = ["slerp_stable", "slerp_batch", "tsqr_blocked", "cholesky_qr2",
           "project_stiefel", "stiefel_drift_check"]
