# bench

Measurements behind the AMD GPU path: memory traffic, memory ceilings and
correctness checks for the fused HIP kernel. They were run on a Radeon 8060S
under native Windows ROCm, driven from WSL by `scripts/rocm-win-zilver.sh`.

For the comparisons most people want (Zilver against Qiskit Aer, and the noise
model against real hardware) see [`benchmarks/`](../benchmarks).

| Script | What it answers |
|---|---|
| `fused_check.py` | Does the fused algorithm match closed forms? Runs on any machine. |
| `coset_toy.py` | Hand-checked version of the fusion and bit-frame logic. |
| `hip_bench.py` | Builds the HIP kernel, checks it against a closed form, then times it. |
| `copies_per_gate.py` | Memory traffic per gate, counted in copies of the state. |
| `pass_price.py` | How many memory passes a circuit costs before and after fusion. |
| `mem_ceiling.py`, `mem_shape.py` | How much memory the GPU can actually allocate, in total and per block. |
| `cold33.py` | Whether 33 qubits fits when allocated first in a fresh process. |
| `report_qft.py` | QFT of a basis state against its closed form, at every width. |
| `report_random.py`, `report_aer.py` | A seeded random circuit on the GPU, re-run in Qiskit Aer for comparison. |
| `report_timing.py` | Memory traffic per gate at each width, against an in-place copy. |
