"""Price a circuit in PASSES over memory, which is what it actually costs.

A statevector gate is bandwidth-bound: the arithmetic is free and the traffic is
everything. So the honest unit is not gates and not seconds -- it is passes over
the state. Three policies, in increasing order of what they exploit:

    naive   one pass per gate. What the simulator does today.
    frame   Clifford index maps (CNOT/X/SWAP) become metadata and cost nothing;
            diagonal gates fold into a neighbouring pass and cost nothing.
    fused   plus: up to m gates on distinct qubits share one pass.

`frame` is the quantum-specific one. CNOT(c,t) is GF(2)-linear on the index, so
a whole Clifford segment composes into one matrix carried alongside the buffer
(see fused.py). Diagonal gates never pair amplitudes at all, so they ride along
in whatever pass comes next.

The point of counting before building: it says which circuits the HIP kernel
helps and by how much. A circuit that is mostly Clifford and diagonal collapses
enormously; a depth-20 Haar circuit on random pairs barely moves. Quoting one
number without the other would be quoting the planner, not the kernel.
"""

from __future__ import annotations

# Gates that permute the index over GF(2): pure bookkeeping under a frame.
_CLIFFORD_INDEX = {"cnot", "x", "swap", "cx"}
# Gates diagonal in the computational basis: no pairing, foldable into a pass.
_DIAGONAL = {"rz", "cz", "rzz", "s", "t", "phase", "z", "p", "u1"}


def count_passes(ops, n_qubits: int, m: int = 4) -> dict:
    """Passes under each policy, plus the counts that explain the difference.

    `ops` is any sequence with `.kind` and `.qubits` -- a Circuit's _ops.
    """
    naive = 0
    index_free = 0
    diag_free = 0
    real: list[list[int]] = []          # the gates that must touch memory

    for op in ops:
        naive += 1
        kind = (op.kind or "").lower()
        if kind in _CLIFFORD_INDEX:
            index_free += 1
            continue
        if kind in _DIAGONAL:
            diag_free += 1
            continue
        real.append(list(op.qubits))

    # merge consecutive gates on the SAME qubit set: two 2x2s are one 2x2
    merged: list[list[int]] = []
    for q in real:
        if merged and merged[-1] == q:
            continue
        merged.append(q)

    # greedy layering: a pass takes up to m gates, all on distinct qubits
    fused = 0
    busy: set[int] = set()
    width = 0
    for q in merged:
        if width >= m or busy.intersection(q):
            fused += 1
            busy, width = set(), 0
        busy.update(q)
        width += 1
    if width:
        fused += 1

    return {
        "gates": naive,
        "naive_passes": naive,
        "frame_passes": len(merged),
        "fused_passes": fused,
        "clifford_deferred": index_free,
        "diagonal_folded": diag_free,
        "merged_away": len(real) - len(merged),
        "m": m,
    }


def report(ops, n_qubits: int, m: int = 4, label: str = "") -> str:
    r = count_passes(ops, n_qubits, m=m)
    speed_f = r["naive_passes"] / max(r["frame_passes"], 1)
    speed_u = r["naive_passes"] / max(r["fused_passes"], 1)
    head = f"{label} ({n_qubits} qubits, {r['gates']} gates)" if label else \
           f"{n_qubits} qubits, {r['gates']} gates"
    return (
        f"{head}\n"
        f"  naive              {r['naive_passes']:>6} passes\n"
        f"  + frame            {r['frame_passes']:>6}   ({speed_f:.1f}x)  "
        f"{r['clifford_deferred']} Clifford deferred, "
        f"{r['diagonal_folded']} diagonal folded, {r['merged_away']} merged\n"
        f"  + fusion m={r['m']}       {r['fused_passes']:>6}   ({speed_u:.1f}x)"
    )
