"""
Zilver - MLX-native statevector quantum circuit simulator for Apple Silicon.

Sirius benchmark + Silicon hardware = Zilver.
"""

from .circuit import Circuit, GateOp
from .simulator import StateVector, apply_gate, expectation_z, expectation_pauli_sum
from .gradients import param_shift_gradient, param_shift_gradient_batched
from .landscape import LossLandscape, LandscapeResult
from . import gates

__version__ = "0.1.0"
__all__ = [
    "Circuit",
    "GateOp",
    "StateVector",
    "apply_gate",
    "expectation_z",
    "expectation_pauli_sum",
    "param_shift_gradient",
    "param_shift_gradient_batched",
    "LossLandscape",
    "LandscapeResult",
    "gates",
]
