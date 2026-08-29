"""
POLYDIM V58 CORE & SOTA NATIVE GEOMETRIC COMPUTING SUITE
Paquete Propietario para Computabilidad Geométrica en S^{D-1} (D >= 10,000)
"""

import warnings
import jax

# Silenciamos advertencias de emulación float64 si la aceleradora no soporta FP64 nativo
warnings.filterwarnings("ignore", message=".*float64.*not supported on.*")

from .silicon import SiliconContract, configure_runtime
from .memory import (
    PMTPHeader,
    PMTPSharedMemoryBuffer,
    PolydimArenaAllocator,
    PMTPConsistencyError,
    PMTPProtocolError
)
from .geometry import GeodesicKernels
from .linear import HouseholderReflection, OrthogonalProjector, SkewLowRankUpdate
from .clifford import CliffordRotors
from .hodge import GrassmannianHodge
from .validation import assert_isometry

# Módulos Teóricos SOTA V58
from .quantum_geometry import QuantumGeodesicKernels
from .lie_groups import LieGroupOperators
from .quantum_information import QuantumInformation
from .tensor_networks import TensorNetwork, HolographicDuality
from .topological_invariants import TopologicalInvariants, KahlerGeometry
from .riemannian_learning import RiemannianLearning

__version__ = "58.0.0"
__all__ = [
    # Core Infrastructure
    "SiliconContract",
    "configure_runtime",
    "PMTPHeader",
    "PMTPSharedMemoryBuffer",
    "PolydimArenaAllocator",
    "PMTPConsistencyError",
    "PMTPProtocolError",
    "GeodesicKernels",
    "HouseholderReflection",
    "OrthogonalProjector",
    "SkewLowRankUpdate",
    "CliffordRotors",
    "GrassmannianHodge",
    "assert_isometry",
    # SOTA Modules
    "QuantumGeodesicKernels",
    "LieGroupOperators",
    "QuantumInformation",
    "TensorNetwork",
    "HolographicDuality",
    "TopologicalInvariants",
    "KahlerGeometry",
    "RiemannianLearning"
]
