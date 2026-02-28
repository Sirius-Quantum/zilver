"""
Batch distributor: split parameter grids across nodes for vmap evaluation.

VQA training and landscape sampling require evaluating E(θ) at many
parameter points. The batch distributor partitions a (N, n_params)
parameter grid across available nodes. Each node runs mx.vmap on its
slice, saturating Metal throughput. Results are reassembled in order.

Two execution paths:
  run_local_batch()    — single vmap dispatch on the current device
  BatchDistributor     — registry-routed slices dispatched to node pool

For cut circuits use run_cut_local_batch() and CutBatchDistributor.

Slice assignment:
  Slices are distributed round-robin across k eligible nodes, ordered by
  (jobs_in_flight ASC, stake DESC) as returned by Registry.match_all().
  Each node receives ceil(N/k) or floor(N/k) parameter sets.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BatchSlice:
    """
    Record of one node's share of a batch execution.

    node_id:      node that processed this slice
    start:        first index (inclusive) in the original batch
    end:          last index (exclusive)
    expectations: expectation values for params_batch[start:end]
    elapsed_ms:   wall-clock time for this slice
    """
    node_id:      str
    start:        int
    end:          int
    expectations: list[float]
    elapsed_ms:   float


@dataclass
class BatchResult:
    """
    Output of BatchDistributor.run() or run_local_batch().

    expectations:  flat list of length N, order matches input params_batch
    n_evals:       N (total parameter sets evaluated)
    n_nodes_used:  number of distinct nodes that executed slices
    slices:        per-node breakdown for diagnostics
    elapsed_ms:    total wall-clock time
    """
    expectations: list[float]
    n_evals:      int
    n_nodes_used: int
    slices:       list[BatchSlice]
    elapsed_ms:   float

    def as_array(self) -> mx.array:
        """Return expectations as an (N,) MLX array."""
        return mx.array(self.expectations, dtype=mx.float32)

    def reshape(self, *shape) -> list:
        """Reshape expectations into a nested list, e.g. reshape(20, 20)."""
        arr = np.array(self.expectations, dtype=np.float32).reshape(shape)
        return arr.tolist()

    def __repr__(self) -> str:
        return (
            f"BatchResult("
            f"n_evals={self.n_evals}, "
            f"n_nodes={self.n_nodes_used}, "
            f"elapsed_ms={self.elapsed_ms:.1f})"
        )


# ---------------------------------------------------------------------------
# Index splitting
# ---------------------------------------------------------------------------

def _split_indices(n: int, k: int) -> list[tuple[int, int]]:
    """
    Split n items across k workers into contiguous (start, end) pairs.

    Items are distributed as evenly as possible; the first (n % k) workers
    receive one extra item each.

        n=10, k=3  →  [(0,4), (4,7), (7,10)]
        n=6,  k=4  →  [(0,2), (2,4), (4,5), (5,6)]
    """
    chunk = n // k
    extra = n % k
    slices = []
    start = 0
    for i in range(k):
        end = start + chunk + (1 if i < extra else 0)
        slices.append((start, end))
        start = end
    return slices


# ---------------------------------------------------------------------------
# Single-device batch execution
# ---------------------------------------------------------------------------

def run_local_batch(
    circuit,
    params_batch: mx.array,
    observable: str = "sum_z",
) -> BatchResult:
    """
    Evaluate a circuit at N parameter sets using a single vmap dispatch.

    Args:
        circuit:      a zilver Circuit instance
        params_batch: (N, n_params) array of parameter sets
        observable:   passed to circuit.compile()

    Returns:
        BatchResult with expectations of length N.

    This is the single-node fast path. For multi-node distribution use
    BatchDistributor instead.
    """
    if params_batch.ndim == 1:
        params_batch = params_batch[None, :]

    n_evals = params_batch.shape[0]
    t0 = time.perf_counter()

    fn = circuit.compile(observable)
    out = mx.vmap(fn)(params_batch)  # (N,)
    mx.eval(out)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    expectations = out.tolist()

    sl = BatchSlice(
        node_id="local",
        start=0,
        end=n_evals,
        expectations=expectations,
        elapsed_ms=elapsed_ms,
    )
    return BatchResult(
        expectations=expectations,
        n_evals=n_evals,
        n_nodes_used=1,
        slices=[sl],
        elapsed_ms=elapsed_ms,
    )


def run_cut_local_batch(
    decomp,
    params_batch: mx.array,
    observable: str = "sum_z",
) -> BatchResult:
    """
    Evaluate a CutDecomposition at N parameter sets using a single device.

    For each QPD pair, both left and right subcircuits are vmapped over the
    full params_batch. Results are recombined with QPD weights per evaluation.

    Args:
        decomp:       CutDecomposition from CutCircuit.decompose()
        params_batch: (N, n_params) array of parameter sets
        observable:   passed to subcircuit.compile()

    Returns:
        BatchResult with recombined expectations of length N.
    """
    if params_batch.ndim == 1:
        params_batch = params_batch[None, :]

    n_evals = params_batch.shape[0]
    t0 = time.perf_counter()

    expectations = mx.zeros((n_evals,), dtype=mx.float32)
    for pair in decomp.pairs:
        lf = pair.left.compile(observable)
        rf = pair.right.compile(observable)
        lv = mx.vmap(lf)(params_batch)   # (N,)
        rv = mx.vmap(rf)(params_batch)   # (N,)
        expectations = expectations + pair.weight * (lv + rv)

    mx.eval(expectations)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    exp_list = expectations.tolist()

    sl = BatchSlice(
        node_id="local",
        start=0,
        end=n_evals,
        expectations=exp_list,
        elapsed_ms=elapsed_ms,
    )
    return BatchResult(
        expectations=exp_list,
        n_evals=n_evals,
        n_nodes_used=1,
        slices=[sl],
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Multi-node batch distributor
# ---------------------------------------------------------------------------

class BatchDistributor:
    """
    Distribute a parameter batch across a pool of registered nodes.

    Each node receives a contiguous slice of the (N, n_params) batch and
    executes it with mx.vmap. Results are reassembled in original order.

    In this local simulation, all computation still runs on the current
    device. Slices are dispatched sequentially with registry load tracking.
    In the full P2P network each slice becomes a SimJob sent to a remote node.

    Usage:
        from zilver.registry import Registry
        from zilver.batch_distributor import BatchDistributor

        reg = Registry()
        reg.register(node_a.caps)
        reg.register(node_b.caps)

        bd = BatchDistributor(reg)
        result = bd.run(circuit, params_batch, backend="sv")
        # result.expectations[i] == E(params_batch[i])
    """

    def __init__(self, registry):
        self.registry = registry

    def run(
        self,
        circuit,
        params_batch: mx.array,
        backend:    str = "sv",
        observable: str = "sum_z",
        max_nodes:  int = 64,
    ) -> BatchResult:
        """
        Execute a parameter batch distributed across eligible nodes.

        Args:
            circuit:      a zilver Circuit instance
            params_batch: (N, n_params) array of parameter sets
            backend:      "sv" | "dm" | "tn"
            observable:   passed to circuit.compile()
            max_nodes:    cap on the number of nodes to query from registry

        Returns:
            BatchResult with N expectations in original order.

        Raises:
            RuntimeError if no eligible node is found.
        """
        if params_batch.ndim == 1:
            params_batch = params_batch[None, :]

        n_evals = params_batch.shape[0]
        n_qubits = circuit.n_qubits

        nodes = self.registry.match_all(backend, n_qubits, count=max_nodes)
        if not nodes:
            raise RuntimeError(
                f"No eligible node for backend={backend!r} n_qubits={n_qubits}"
            )

        k = len(nodes)
        index_slices = _split_indices(n_evals, k)

        fn = circuit.compile(observable)
        t0 = time.perf_counter()

        result_slices: list[BatchSlice] = []
        all_expectations: list[float] = [0.0] * n_evals

        for node_entry, (start, end) in zip(nodes, index_slices):
            if start == end:
                continue
            nid = node_entry.caps.node_id
            self.registry.assign(nid)

            ts = time.perf_counter()
            chunk = params_batch[start:end]           # (chunk_size, n_params)
            out = mx.vmap(fn)(chunk)                  # (chunk_size,)
            mx.eval(out)
            slice_ms = (time.perf_counter() - ts) * 1000.0

            self.registry.complete(nid)

            chunk_exp = out.tolist()
            for i, val in enumerate(chunk_exp):
                all_expectations[start + i] = val

            result_slices.append(BatchSlice(
                node_id=nid,
                start=start,
                end=end,
                expectations=chunk_exp,
                elapsed_ms=slice_ms,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BatchResult(
            expectations=all_expectations,
            n_evals=n_evals,
            n_nodes_used=len(result_slices),
            slices=result_slices,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Multi-node cut circuit batch distributor
# ---------------------------------------------------------------------------

class CutBatchDistributor:
    """
    Distribute a CutDecomposition batch across a node pool.

    For each QPD pair, both subcircuits are dispatched to separate nodes
    from the registry. Each node vmaps its subcircuit over its param slice.
    QPD recombination is applied after all nodes report results.

    This enables large-circuit evaluation: a 40-qubit circuit cut at qubit
    20 generates 4 QPD pairs, each pair split across 2×k nodes. Each node
    only simulates 20 qubits, and N evaluations are parallelised across k
    slices per subcircuit.

    Usage:
        cbd = CutBatchDistributor(registry)
        result = cbd.run(decomp, params_batch)
    """

    def __init__(self, registry):
        self.registry = registry

    def run(
        self,
        decomp,
        params_batch: mx.array,
        backend:    str = "sv",
        observable: str = "sum_z",
        max_nodes:  int = 64,
    ) -> BatchResult:
        """
        Execute a cut circuit parameter batch across eligible nodes.

        Args:
            decomp:       CutDecomposition from CutCircuit.decompose()
            params_batch: (N, n_params) parameter sets
            backend:      "sv" | "dm" | "tn"
            observable:   passed to subcircuit.compile()
            max_nodes:    cap on registry query per subcircuit

        Returns:
            BatchResult with N recombined expectations.

        Raises:
            RuntimeError if no eligible node for any subcircuit.
        """
        if params_batch.ndim == 1:
            params_batch = params_batch[None, :]

        n_evals = params_batch.shape[0]
        t0 = time.perf_counter()

        expectations = mx.zeros((n_evals,), dtype=mx.float32)

        for pair in decomp.pairs:
            lv = self._run_subcircuit_batch(
                pair.left, params_batch, backend, observable, max_nodes
            )
            rv = self._run_subcircuit_batch(
                pair.right, params_batch, backend, observable, max_nodes
            )
            expectations = expectations + pair.weight * (lv + rv)

        mx.eval(expectations)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        exp_list = expectations.tolist()

        sl = BatchSlice(
            node_id="distributed",
            start=0,
            end=n_evals,
            expectations=exp_list,
            elapsed_ms=elapsed_ms,
        )
        return BatchResult(
            expectations=exp_list,
            n_evals=n_evals,
            n_nodes_used=len(decomp.pairs),
            slices=[sl],
            elapsed_ms=elapsed_ms,
        )

    def _run_subcircuit_batch(
        self,
        circuit,
        params_batch: mx.array,
        backend:    str,
        observable: str,
        max_nodes:  int,
    ) -> mx.array:
        """Run one subcircuit across all parameter sets, distributed over nodes."""
        n_evals  = params_batch.shape[0]
        n_qubits = circuit.n_qubits
        nodes = self.registry.match_all(backend, n_qubits, count=max_nodes)
        if not nodes:
            raise RuntimeError(
                f"No eligible node for backend={backend!r} n_qubits={n_qubits}"
            )

        k = len(nodes)
        index_slices = _split_indices(n_evals, k)
        fn = circuit.compile(observable)

        chunks: list[mx.array] = []
        for node_entry, (start, end) in zip(nodes, index_slices):
            if start == end:
                continue
            nid = node_entry.caps.node_id
            self.registry.assign(nid)
            out = mx.vmap(fn)(params_batch[start:end])
            mx.eval(out)
            self.registry.complete(nid)
            chunks.append(out)

        return mx.concatenate(chunks, axis=0)
