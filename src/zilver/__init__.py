"""
Zilver — MLX-native quantum circuit simulator for Apple Silicon.

Sirius benchmark + Silicon hardware = Zilver.

Core simulation
---------------
``Circuit``, ``LossLandscape``, ``param_shift_gradient`` and the gate
library are available without any optional dependencies.

Statevector backends
--------------------
``Circuit.statevector(method=...)`` dispatches across:

* ``method="metal"`` — custom Metal compute kernels fused via ``mx.compile``
  (this is the headline single-statevector path on Apple Silicon).
* ``method="accel"`` — multithreaded CPU path (numba + Accelerate). Requires
  ``pip install zilver[accel]``. Supports complex64 and complex128.
* ``method="mlx"`` — the original generic MLX path; still optimal for
  ``vmap``-batched workloads (parameter sweeps, gradient batches, fidelity
  kernels).
* ``method="auto"`` (default) — picks the best backend for the circuit and
  precision.

Distributed network (requires ``pip install zilver[network]``)
---------------------------------------------------------------
``NodeClient``, ``RegistryClient``, and ``NetworkCoordinator`` are imported
lazily so that the simulator remains usable without FastAPI / httpx installed.
"""

try:
    from .circuit import Circuit, GateOp
    from .simulator import StateVector, apply_gate, expectation_z, expectation_pauli_sum
    from .gradients import param_shift_gradient, param_shift_gradient_batched
    from .landscape import LossLandscape, LandscapeResult
    from . import gates
except ImportError:
    # MLX not available (e.g., registry running on Linux x86).
    # Simulation APIs are unavailable; network/registry layer still works.
    Circuit = GateOp = StateVector = apply_gate = None          # type: ignore[assignment,misc]
    expectation_z = expectation_pauli_sum = None                # type: ignore[assignment]
    param_shift_gradient = param_shift_gradient_batched = None  # type: ignore[assignment]
    LossLandscape = LandscapeResult = gates = None              # type: ignore[assignment,misc]

__version__ = "0.4.0"
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
    # Statevector backends (optional, lazily imported)
    "metal",
    "accel",
    # Distributed network (optional)
    "NodeClient",
    "RegistryClient",
    "NetworkCoordinator",
]


def __getattr__(name: str):
    """
    Lazy imports for optional / heavy-dep submodules.

    * ``metal``  — needs the MLX Metal backend (any Apple Silicon Mac).
    * ``accel``  — needs ``numba`` (install via ``pip install zilver[accel]``).
    * ``NodeClient`` / ``RegistryClient`` / ``NetworkCoordinator`` — need the
      ``[network]`` extras (FastAPI, httpx, etc.).

    Importing on attribute access keeps a plain ``import zilver`` cheap.
    """
    import importlib
    if name in {"metal", "accel"}:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    _network = {"NodeClient", "RegistryClient", "NetworkCoordinator"}
    if name in _network:
        from .client import NodeClient, RegistryClient, NetworkCoordinator  # noqa: F401
        return locals()[name]

    raise AttributeError(f"module 'zilver' has no attribute {name!r}")
