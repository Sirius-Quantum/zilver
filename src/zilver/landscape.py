"""Loss landscape sampling."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
import time
import mlx.core as mx
import numpy as np

from .gradients import param_shift_gradient, gradient_magnitude


@dataclass
class LandscapeResult:
    """
    Output of a LossLandscape.compute() call.

    Matches the output format:
        loss_landscape:     List[List[float]]  (resolution x resolution)
        gradient_landscape: List[List[float]]  (resolution x resolution)
        parameter_names:    List[str]
        parameter_ranges:   Dict[str, Tuple[float, float]]
    """
    loss_landscape: list[list[float]]
    gradient_landscape: list[list[float]]
    parameter_names: list[str]
    parameter_ranges: dict[str, tuple[float, float]]
    n_qubits: int
    n_params: int
    resolution: int
    wall_time_seconds: float
    backend: str = "zilver-mlx"
    metadata: dict = field(default_factory=dict)

    def plateau_coverage(self, threshold: float = 0.1) -> float:
        """Fraction of grid points with gradient magnitude below threshold."""
        flat = [g for row in self.gradient_landscape for g in row]
        return sum(1 for g in flat if g < threshold) / len(flat)

    def trainability_score(self, threshold: float = 0.1) -> float:
        return 1.0 - self.plateau_coverage(threshold)


class LossLandscape:
    """
    Compute 2D loss and gradient landscapes over a swept parameter grid.

    The two swept parameters scan [-pi, pi] x [-pi, pi] at `resolution` points
    each. All other parameters are fixed at `fixed_params` (or random if not
    provided). The full grid is evaluated in a single mx.vmap dispatch.

    Example:
        from zilver.circuit import hardware_efficient
        from zilver.landscape import LossLandscape

        circuit = hardware_efficient(n_qubits=6, depth=3)
        landscape = LossLandscape(circuit, sweep_params=(0, 1), resolution=20)
        result = landscape.compute()

        print(f"Plateau coverage: {result.plateau_coverage():.1%}")
    """

    def __init__(
        self,
        circuit,  # zilver.circuit.Circuit
        sweep_params: tuple[int, int] = (0, 1),
        resolution: int = 20,
        fixed_params: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.circuit = circuit
        self.sweep_params = sweep_params
        self.resolution = resolution
        self.n_params = circuit.n_params

        rng = np.random.default_rng(seed)
        if fixed_params is not None:
            self._fixed = fixed_params.astype(np.float32)
        else:
            self._fixed = rng.uniform(-np.pi, np.pi, self.n_params).astype(np.float32)

    def compute(self, observable: str = "sum_z") -> LandscapeResult:
        """
        Evaluate the 20x20 loss and gradient landscape.

        All 400 grid evaluations are batched via mx.vmap.
        Gradients (w.r.t. the two swept parameters only) use parameter shift,
        also batched over all 400 points simultaneously.
        """
        t0 = time.perf_counter()

        f = self.circuit.compile(observable=observable)
        grid_params = self._build_grid_params()  # (resolution^2, P)

        # --- Batch loss evaluation -------------------------------------------
        all_losses = mx.vmap(f)(grid_params)  # (R^2,)
        mx.eval(all_losses)

        # --- Batch gradient computation w.r.t. swept params -----------------
        # For efficiency we compute parameter shift only for the 2 swept params.
        p0_idx, p1_idx = self.sweep_params
        shift_val = np.pi / 2

        # Precompute one-hot shift vectors as constants (outside vmap closure)
        identity = mx.eye(self.n_params, dtype=mx.float32)
        e0 = identity[p0_idx] * shift_val  # (P,) shift vector for param 0
        e1 = identity[p1_idx] * shift_val  # (P,) shift vector for param 1

        def grad_2d(params: mx.array) -> mx.array:
            """Gradient magnitude via parameter shift for the two swept params."""
            g0 = 0.5 * (f(params + e0) - f(params - e0))
            g1 = 0.5 * (f(params + e1) - f(params - e1))
            return mx.sqrt(g0 ** 2 + g1 ** 2)

        all_grads = mx.vmap(grad_2d)(grid_params)  # (R^2,)
        mx.eval(all_grads)

        wall_time = time.perf_counter() - t0

        # --- Reshape to 2D grids ---------------------------------------------
        R = self.resolution
        losses_np = np.array(all_losses.tolist(), dtype=np.float32).reshape(R, R)
        grads_np  = np.array(all_grads.tolist(),  dtype=np.float32).reshape(R, R)

        axis = np.linspace(-np.pi, np.pi, R)
        p0_name = f"param_{p0_idx}"
        p1_name = f"param_{p1_idx}"

        return LandscapeResult(
            loss_landscape=losses_np.tolist(),
            gradient_landscape=grads_np.tolist(),
            parameter_names=[p0_name, p1_name],
            parameter_ranges={
                p0_name: (-float(np.pi), float(np.pi)),
                p1_name: (-float(np.pi), float(np.pi)),
            },
            n_qubits=self.circuit.n_qubits,
            n_params=self.n_params,
            resolution=R,
            wall_time_seconds=wall_time,
            metadata={
                "sweep_params": list(self.sweep_params),
                "observable": observable,
            },
        )

    def _build_grid_params(self) -> mx.array:
        """
        Build (R^2, P) parameter grid.

        The two swept parameters vary over the grid; all others are fixed.
        """
        R = self.resolution
        axis = np.linspace(-np.pi, np.pi, R, dtype=np.float32)

        p0_grid, p1_grid = np.meshgrid(axis, axis, indexing="ij")
        p0_flat = p0_grid.reshape(-1)
        p1_flat = p1_grid.reshape(-1)

        # Tile fixed params across all R^2 grid points
        params = np.tile(self._fixed, (R * R, 1))
        params[:, self.sweep_params[0]] = p0_flat
        params[:, self.sweep_params[1]] = p1_flat

        return mx.array(params)


def landscape_from_qasm(
    qasm_str: str,
    sweep_params: tuple[int, int] = (0, 1),
    resolution: int = 20,
    fixed_params: np.ndarray | None = None,
) -> LandscapeResult:
    """
    Convenience function: parse a QASM string and compute the landscape.
    Bridges the QASM circuits into Zilver.
    """
    from .qasm_bridge import circuit_from_qasm
    circuit = circuit_from_qasm(qasm_str)
    return LossLandscape(circuit, sweep_params, resolution, fixed_params).compute()
