"""
Tests for batch distributor.

Correctness checks:
  - _split_indices splits evenly
  - _split_indices handles remainder
  - _split_indices handles k >= n
  - run_local_batch returns BatchResult
  - run_local_batch expectations length matches N
  - run_local_batch results match single-eval circuit
  - run_local_batch accepts 1D params (single eval)
  - run_local_batch reshape utility
  - run_cut_local_batch returns BatchResult
  - run_cut_local_batch expectations are finite
  - run_cut_local_batch matches run_local_batch on no-entanglement cut
  - BatchDistributor.run returns BatchResult
  - BatchDistributor.run expectations match run_local_batch
  - BatchDistributor.run raises when no eligible node
  - BatchDistributor.run single node uses one slice
  - BatchDistributor.run multiple nodes splits batch
  - BatchDistributor.run load tracking (assign/complete)
  - CutBatchDistributor.run returns BatchResult
  - CutBatchDistributor.run expectations are finite
  - CutBatchDistributor.run raises when no eligible node
  - BatchResult.as_array returns mx.array
  - BatchResult repr
"""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.circuit import Circuit, hardware_efficient
from zilver.cutting import CutCircuit
from zilver.node import NodeCapabilities
from zilver.registry import Registry
from zilver.batch_distributor import (
    _split_indices,
    run_local_batch,
    run_cut_local_batch,
    BatchDistributor,
    CutBatchDistributor,
    BatchResult,
    BatchSlice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ry_circuit(n: int = 4) -> Circuit:
    c = Circuit(n)
    for q in range(n):
        c.ry(q, q)
    c.n_params = n
    return c


def _params_batch(n_evals: int, n_params: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-math.pi, math.pi, (n_evals, n_params)).astype(np.float32)
    return mx.array(arr)


def _registry_with_nodes(n: int = 2, sv_max: int = 20) -> Registry:
    reg = Registry()
    for i in range(n):
        caps = NodeCapabilities(
            node_id=f"node-{i}",
            chip="Apple M4",
            ram_gb=16,
            sv_qubits_max=sv_max,
            dm_qubits_max=10,
            tn_qubits_max=50,
            backends=["sv", "dm", "tn"],
        )
        reg.register(caps)
    return reg


# ---------------------------------------------------------------------------
# _split_indices
# ---------------------------------------------------------------------------

class TestSplitIndices:
    def test_even_split(self):
        slices = _split_indices(12, 3)
        assert slices == [(0, 4), (4, 8), (8, 12)]

    def test_remainder_distributed_front(self):
        slices = _split_indices(10, 3)
        # 10 = 3+3+4? No: 10//3=3, 10%3=1, so first worker gets extra
        assert slices == [(0, 4), (4, 7), (7, 10)]

    def test_k_equals_n(self):
        slices = _split_indices(4, 4)
        assert slices == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_k_greater_than_n(self):
        slices = _split_indices(3, 5)
        # 3//5=0, so first 3 workers get 1 item, last 2 get 0
        assert slices == [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]

    def test_covers_full_range(self):
        n, k = 17, 5
        slices = _split_indices(n, k)
        assert slices[0][0] == 0
        assert slices[-1][1] == n

    def test_contiguous(self):
        slices = _split_indices(20, 6)
        for i in range(len(slices) - 1):
            assert slices[i][1] == slices[i + 1][0]

    def test_single_worker(self):
        assert _split_indices(7, 1) == [(0, 7)]


# ---------------------------------------------------------------------------
# run_local_batch
# ---------------------------------------------------------------------------

class TestRunLocalBatch:
    def test_returns_batch_result(self):
        c = _ry_circuit(4)
        pb = _params_batch(8, 4)
        result = run_local_batch(c, pb)
        assert isinstance(result, BatchResult)

    def test_expectations_length(self):
        c = _ry_circuit(4)
        pb = _params_batch(12, 4)
        result = run_local_batch(c, pb)
        assert len(result.expectations) == 12
        assert result.n_evals == 12

    def test_all_finite(self):
        c = _ry_circuit(4)
        pb = _params_batch(8, 4)
        result = run_local_batch(c, pb)
        assert all(math.isfinite(v) for v in result.expectations)

    def test_matches_single_eval(self):
        c = _ry_circuit(4)
        fn = c.compile("sum_z")
        rng = np.random.default_rng(1)
        params_np = rng.uniform(-math.pi, math.pi, (5, 4)).astype(np.float32)
        pb = mx.array(params_np)

        result = run_local_batch(c, pb)
        for i in range(5):
            ref = float(fn(pb[i]).item())
            assert abs(result.expectations[i] - ref) < 1e-5

    def test_single_eval_1d_input(self):
        c = _ry_circuit(4)
        params_1d = mx.zeros((4,), dtype=mx.float32)
        result = run_local_batch(c, params_1d)
        assert result.n_evals == 1
        assert len(result.expectations) == 1

    def test_n_nodes_used_is_one(self):
        c = _ry_circuit(4)
        result = run_local_batch(c, _params_batch(8, 4))
        assert result.n_nodes_used == 1

    def test_slice_is_local(self):
        c = _ry_circuit(4)
        result = run_local_batch(c, _params_batch(8, 4))
        assert result.slices[0].node_id == "local"

    def test_elapsed_ms_positive(self):
        c = _ry_circuit(4)
        result = run_local_batch(c, _params_batch(8, 4))
        assert result.elapsed_ms >= 0

    def test_as_array(self):
        c = _ry_circuit(4)
        result = run_local_batch(c, _params_batch(4, 4))
        arr = result.as_array()
        assert isinstance(arr, mx.array)
        assert arr.shape == (4,)

    def test_reshape(self):
        c = _ry_circuit(4)
        result = run_local_batch(c, _params_batch(9, 4))
        grid = result.reshape(3, 3)
        assert len(grid) == 3
        assert len(grid[0]) == 3

    def test_repr(self):
        c = _ry_circuit(4)
        result = run_local_batch(c, _params_batch(4, 4))
        r = repr(result)
        assert "BatchResult" in r
        assert "n_evals=4" in r


# ---------------------------------------------------------------------------
# run_cut_local_batch
# ---------------------------------------------------------------------------

class TestRunCutLocalBatch:
    def _decomp(self):
        c = Circuit(4)
        for q in range(4):
            c.h(q)
        for q in range(3):
            c.cnot(q, q + 1)
        return CutCircuit(c).add_wire_cut(partition_qubit=2).decompose()

    def test_returns_batch_result(self):
        decomp = self._decomp()
        pb = _params_batch(6, 0)
        # zero-param circuit: params_batch should still be (N, 0) shape
        pb = mx.zeros((6, 0), dtype=mx.float32)
        result = run_cut_local_batch(decomp, pb)
        assert isinstance(result, BatchResult)

    def test_expectations_length(self):
        decomp = self._decomp()
        pb = mx.zeros((8, 0), dtype=mx.float32)
        result = run_cut_local_batch(decomp, pb)
        assert len(result.expectations) == 8
        assert result.n_evals == 8

    def test_all_finite(self):
        decomp = self._decomp()
        pb = mx.zeros((6, 0), dtype=mx.float32)
        result = run_cut_local_batch(decomp, pb)
        assert all(math.isfinite(v) for v in result.expectations)

    def test_parametrized_circuit_finite(self):
        n, p = 6, 3
        c = Circuit(n)
        for q in range(n):
            c.ry(q, q)
        c.n_params = n
        decomp = CutCircuit(c).add_wire_cut(partition_qubit=p).decompose()

        rng = np.random.default_rng(42)
        pb = mx.array(rng.uniform(-math.pi, math.pi, (5, n)).astype(np.float32))
        result = run_cut_local_batch(decomp, pb)
        assert all(math.isfinite(v) for v in result.expectations)

    def test_matches_run_local_per_eval(self):
        """Each cut batch result must match a per-param run_local() call."""
        from zilver.coordinator import run_local
        n, p = 4, 2
        c = Circuit(n)
        for q in range(n):
            c.ry(q, q)
        c.n_params = n
        decomp = CutCircuit(c).add_wire_cut(partition_qubit=p).decompose()

        rng = np.random.default_rng(7)
        params_np = rng.uniform(-math.pi, math.pi, (4, n)).astype(np.float32)
        pb = mx.array(params_np)

        batch_result = run_cut_local_batch(decomp, pb)
        for i in range(4):
            ref = run_local(decomp, pb[i]).expectation
            assert abs(batch_result.expectations[i] - ref) < 1e-5, (
                f"index {i}: batch={batch_result.expectations[i]:.6f} ref={ref:.6f}"
            )


# ---------------------------------------------------------------------------
# BatchDistributor
# ---------------------------------------------------------------------------

class TestBatchDistributor:
    def test_returns_batch_result(self):
        reg = _registry_with_nodes(2)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        result = bd.run(c, _params_batch(8, 4))
        assert isinstance(result, BatchResult)

    def test_expectations_length(self):
        reg = _registry_with_nodes(2)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        result = bd.run(c, _params_batch(10, 4))
        assert len(result.expectations) == 10

    def test_all_finite(self):
        reg = _registry_with_nodes(2)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        result = bd.run(c, _params_batch(8, 4))
        assert all(math.isfinite(v) for v in result.expectations)

    def test_matches_run_local_batch(self):
        reg = _registry_with_nodes(3)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        pb = _params_batch(12, 4)

        local = run_local_batch(c, pb)
        dist  = bd.run(c, pb)

        for i, (lv, dv) in enumerate(zip(local.expectations, dist.expectations)):
            assert abs(lv - dv) < 1e-5, f"index {i}: local={lv:.6f} dist={dv:.6f}"

    def test_single_node_one_slice(self):
        reg = _registry_with_nodes(1)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        result = bd.run(c, _params_batch(6, 4))
        assert result.n_nodes_used == 1
        assert len(result.slices) == 1

    def test_two_nodes_two_slices(self):
        reg = _registry_with_nodes(2)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        result = bd.run(c, _params_batch(8, 4), max_nodes=2)
        assert result.n_nodes_used == 2
        assert len(result.slices) == 2

    def test_slices_cover_full_range(self):
        reg = _registry_with_nodes(3)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        result = bd.run(c, _params_batch(9, 4), max_nodes=3)
        indices = set()
        for sl in result.slices:
            indices.update(range(sl.start, sl.end))
        assert indices == set(range(9))

    def test_raises_no_eligible_node(self):
        reg = _registry_with_nodes(1, sv_max=2)
        bd = BatchDistributor(reg)
        c = _ry_circuit(8)  # 8 qubits, node only supports 2
        with pytest.raises(RuntimeError):
            bd.run(c, _params_batch(4, 8))

    def test_load_tracking(self):
        reg = _registry_with_nodes(1)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        # After run, jobs_in_flight should be back to 0 (assign+complete)
        bd.run(c, _params_batch(4, 4))
        assert reg.get("node-0").jobs_in_flight == 0

    def test_jobs_completed_incremented(self):
        reg = _registry_with_nodes(1)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        bd.run(c, _params_batch(4, 4))
        assert reg.get("node-0").caps.jobs_completed == 1

    def test_1d_params_accepted(self):
        reg = _registry_with_nodes(1)
        bd = BatchDistributor(reg)
        c = _ry_circuit(4)
        params_1d = mx.zeros((4,), dtype=mx.float32)
        result = bd.run(c, params_1d)
        assert result.n_evals == 1


# ---------------------------------------------------------------------------
# CutBatchDistributor
# ---------------------------------------------------------------------------

class TestCutBatchDistributor:
    def _decomp(self, n=4, p=2):
        c = Circuit(n)
        for q in range(n):
            c.h(q)
        for q in range(n - 1):
            c.cnot(q, q + 1)
        return CutCircuit(c).add_wire_cut(partition_qubit=p).decompose()

    def test_returns_batch_result(self):
        reg = _registry_with_nodes(2)
        cbd = CutBatchDistributor(reg)
        decomp = self._decomp()
        pb = mx.zeros((6, 0), dtype=mx.float32)
        result = cbd.run(decomp, pb)
        assert isinstance(result, BatchResult)

    def test_expectations_length(self):
        reg = _registry_with_nodes(2)
        cbd = CutBatchDistributor(reg)
        decomp = self._decomp()
        pb = mx.zeros((8, 0), dtype=mx.float32)
        result = cbd.run(decomp, pb)
        assert len(result.expectations) == 8

    def test_all_finite(self):
        reg = _registry_with_nodes(2)
        cbd = CutBatchDistributor(reg)
        decomp = self._decomp()
        pb = mx.zeros((6, 0), dtype=mx.float32)
        result = cbd.run(decomp, pb)
        assert all(math.isfinite(v) for v in result.expectations)

    def test_matches_run_cut_local_batch(self):
        reg = _registry_with_nodes(2)
        cbd = CutBatchDistributor(reg)
        n, p = 4, 2
        c = Circuit(n)
        for q in range(n):
            c.ry(q, q)
        c.n_params = n
        decomp = CutCircuit(c).add_wire_cut(partition_qubit=p).decompose()

        rng = np.random.default_rng(3)
        pb = mx.array(rng.uniform(-math.pi, math.pi, (5, n)).astype(np.float32))

        local  = run_cut_local_batch(decomp, pb)
        dist   = cbd.run(decomp, pb)

        for i, (lv, dv) in enumerate(zip(local.expectations, dist.expectations)):
            assert abs(lv - dv) < 1e-5, f"index {i}: local={lv:.6f} dist={dv:.6f}"

    def test_raises_no_eligible_node(self):
        reg = _registry_with_nodes(1, sv_max=1)  # too small for any subcircuit
        cbd = CutBatchDistributor(reg)
        decomp = self._decomp()
        pb = mx.zeros((4, 0), dtype=mx.float32)
        with pytest.raises(RuntimeError):
            cbd.run(decomp, pb)
