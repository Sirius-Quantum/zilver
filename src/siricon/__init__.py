"""
Zilver — MLX-native quantum circuit simulator for Apple Silicon.

Sirius benchmark + Silicon hardware = Zilver.

Core simulation
---------------
``Circuit``, ``LossLandscape``, ``param_shift_gradient`` and the gate
library are available without any optional dependencies.

Distributed network (requires ``pip install zilver[network]``)
---------------------------------------------------------------
``NodeClient``, ``RegistryClient``, and ``NetworkCoordinator`` are imported
lazily so that the simulator remains usable without FastAPI / httpx installed.
"""

from .circuit import Circuit, GateOp
from .simulator import StateVector, apply_gate, expectation_z, expectation_pauli_sum
from .gradients import param_shift_gradient, param_shift_gradient_batched
from .landscape import LossLandscape, LandscapeResult
from . import gates

__version__ = "0.1.0"
__all__ = [
    # Core simulation
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
    # Distributed network (optional)
    "NodeClient",
    "RegistryClient",
    "NetworkCoordinator",
]


def __getattr__(name: str):
    """
    Lazy import for network-layer symbols.

    Defers importing ``fastapi`` and ``httpx`` until the caller actually
    references a network class, so that ``import zilver`` never fails on
    machines where the ``[network]`` extras are not installed.
    """
    _network = {"NodeClient", "RegistryClient", "NetworkCoordinator"}
    if name in _network:
        from .client import NodeClient, RegistryClient, NetworkCoordinator  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module 'zilver' has no attribute {name!r}")
