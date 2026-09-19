# Changelog

All notable changes to Zilver are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Work on master since 0.6.2. All of it is on the AMD GPU path; nothing here
changes the Apple silicon or CPU results.

### Added
- **Gate fusion.** Consecutive gates are merged into one memory pass instead of
  one pass per gate. Gates on the same qubit are combined first, then gates are
  packed by qubit budget (`ZILVER_FUSE`, default 4 qubits per pass), including
  gates whose qubit sets overlap, where the budget is the union.
- **Support shrinking** (`ZILVER_SHRINK=1`, off by default). A run starts at
  `|0…0>`, so after the schedule has touched only K qubits the state is exactly
  a dense 2^K block tensored with `|0>`, and a pass need only touch 2^K
  amplitudes instead of 2^n. Two things make it usable: gates are reordered to
  add the fewest new qubits, and the bit order is chosen so the live amplitudes
  are a contiguous prefix rather than strided one per page. Measured on the
  published circuit at 32 qubits: 19 passes become 1.57 equivalent full passes,
  and gate time drops from 7.35 s to 0.59 s.
- The fused HIP kernel is specialised on the pass width at compile time.
- `passes.count_passes()` and `passes.report()` price a circuit in memory passes
  before any kernel runs.

### Changed
- **Readback is chunked through a pinned buffer.** On an APU, device memory is
  host DRAM, so the bulk `.cpu()` path was staging through a pageable buffer at
  5.05 GB/s — slower than a single core manages in place. Chunking at 64 MB
  through pinned memory measured 22.22 GB/s over 4.29 GB, which takes a 6.40 s
  readback at 32 qubits to about 1.55 s.
- `|0…0>` is allocated on the device instead of on the host and copied over.
- **`python -m zilver.gpu` now runs with support shrinking on.** It also emits its
  circuit from the last qubit down, single-qubit layer included, so first touch
  runs n-1, n-2, … and the bit gauge is already the identity — the un-permutation
  then costs nothing. Emitted the other way round it would be a full bit reversal,
  which needs a second 34.36 GB buffer at 32 qubits and does not fit. The H/RY
  layer precedes the ladder and so is what sets first touch; reversing only the
  CNOTs would leave the gauge unchanged. This renames qubits and changes no
  amplitude. `ZILVER_SHRINK=0` still gives the dense timings, and because the
  variable is read at import the runner relaunches itself once to set it, exactly
  as it already did for `ZILVER_BACKEND`.
- **BREAKING: `Node.start()` refuses to start without a signing key.** Pass
  `public_key_bytes` with `private_key_bytes` or `se_label`, or pass the new
  `allow_unsigned=True` for a local node that is explicitly unverifiable. An
  unsigned result carries `node_signature = ""`, and `verify_result_signature()`
  returns False for an empty signature exactly as it does for a forged one — so
  a client could not tell an honest unsigned node from an attacker. Starting
  unsigned is now a decision someone makes rather than a state a missing module
  drops the node into silently.
- **A result that cannot be signed is no longer returned unsigned.** The signing
  path was `except Exception: pass`, so any failure while holding a key produced
  an empty signature and no error. It now raises.

### Notes
- Support shrinking stays off by default *in the library*, because reordering
  gates and relabelling bits can each produce a perfectly normalised *wrong*
  state. It is verified per amplitude, never on a norm. Its un-permutation step
  needs a second full buffer (34.36 GB at 32 qubits), so wide runs need a circuit
  whose bit order is already the identity — which is why `zilver.gpu`, the one
  place it is turned on, emits exactly such a circuit.

## [0.6.2] - 2026-09-11

### Changed
- The GPU runner moved from `scripts/gpu.py` into the package as `zilver.gpu`,
  so `pip install zilver` is enough to run it: `python -m zilver.gpu`.
  `scripts/gpu.py` still works as a thin wrapper.
- With `ZILVER_BACKEND` unset, `zilver.gpu` relaunches itself once with the GPU
  backend selected. The backend is chosen when the package is first imported, so
  without this a machine without MLX would time the CPU while naming a GPU.

## [0.6.1] - 2026-09-11

### Changed
- README and package description now say what Zilver is: a quantum circuit
  simulator that runs on Apple silicon, AMD GPUs and x86 CPUs, with a
  distributed network layer on top. No code change from 0.6.0.

## [0.6.0] - 2026-09-11

### Added
- **AMD GPU support through a fused HIP kernel.** One kernel applies a gate in
  place out of registers, so a gate needs 1× the state instead of the 2–3× every
  allocating path needs. That is what puts 32 qubits on a Radeon 8060S (gfx1151)
  under native Windows ROCm.
- **The kernel is compiled at run time with hipRTC** and launched through the HIP
  driver API via `ctypes`. No ninja, no MSVC, no pybind11, no build directory —
  the ROCm runtime that PyTorch already loaded is the whole dependency. It
  declines quietly and falls back whenever anything does not line up.
- **A portable array layer** (`zilver._array`). Zilver was written against
  `mlx.core`, which ships wheels for Apple silicon only, so the package was
  previously unimportable elsewhere. One seam now exports `mx`: MLX on Apple
  silicon, otherwise a PyTorch or NumPy backend covering the surface Zilver uses.
  MLX is an optional dependency, selected by platform marker.
- **A PyTorch backend** so the statevector can live on a GPU: CUDA, ROCm (which
  reports as CUDA), Apple MPS, or DirectML for AMD and Intel GPUs on Windows and
  in WSL2. Selected with `ZILVER_BACKEND=torch`, device with `ZILVER_DEVICE`.
- `python scripts/gpu.py` (now `python -m zilver.gpu`) runs one circuit per width
  and prints the device, the time and the state norm.
- `scripts/x86-node-check.sh` runs a self-contained correctness check on a
  non-Apple machine; `scripts/unified-memory-ceiling.sh` finds the real memory
  ceiling; the Windows ROCm setup scripts drive a box from WSL.
- A `bench/` suite: the closed-form check, the Qiskit Aer comparison, a planted
  fault, copies-per-gate, the memory ceiling, and where the time goes.

### Changed
- Gates are applied through a strided view rather than an n-dimensional tensor.
  The old form built a `[2]*n` tensor — 20 axes at 20 qubits — and GPU backends
  cap tensor rank well below that (MPS refuses above 16). The strided path uses
  3 and 5 axes at any width, and matches the old form to 1.7e-07.
- The `accel` CPU path applies one- and two-qubit gates in place, without index
  arrays, and no longer requires numba: the strided path is reachable without it.
- The simulator no longer routes the statevector through a Python list.
- The density-matrix backend works on the PyTorch backend.

### Fixed
- Complex tensors are refused on a device that has no complex dtype. DirectML
  aborts the process rather than raising on one, which made a stray complex
  tensor invisible until it killed a run, and untestable on MPS, which tolerates
  complex. Both routes onto the device are now guarded, and the state is carried
  as a real pair where needed.

## [0.5.0] - 2026-06-07

### Added
- **Noisy simulation API.** `NoisyCircuit` on the density-matrix backend, with a
  declarative `NoiseModel` that applies Kraus channels automatically after every
  gate, selected by gate arity.
- `NoiseModel.depolarizing(p1, p2)` and `NoiseModel.thermal_relaxation(t1, t2,
  gate_time_1q, gate_time_2q)` factories — the latter builds amplitude- and
  phase-damping channels directly from device `T1`/`T2` and gate times.
- Kraus channel factories exported from the package root: `depolarizing_kraus`,
  `amplitude_damping_kraus`, `phase_damping_kraus`, `bit_flip_kraus`,
  `phase_flip_kraus`.
- `NoisyCircuit.run()` and `.compile()` accept a `noise_model` argument.

### Validated against real hardware
- **Sim-to-real benchmark** (`benchmarks/ghz_hellinger.py`): GHZ + Hellinger-distance
  validation of the noise model against real IBM and IQM hardware, with the noise
  model built from each device's live calibration. Reproducible with your own
  account; published reference runs in `benchmarks/results/`.
- **`benchmarks/noise_demo.py`**: offline check that the noise model reproduces the
  analytic T1/T2 decay laws to ~1e-6 — no account needed.

### Docs & examples
- **[Quickstart](QUICKSTART.md)** — a local getting-started guide: install → first
  circuit → training → quantum kernels → noisy simulation.
- **`examples/noisy_simulation.py`** — noise channels and a depolarizing sweep.
- **`examples/noisy_vqe.py`** — VQE ground-state energy, ideal vs noisy.

## [0.4.0]

- Statevector backends (`metal`, `accel`, `mlx`), parameter-shift gradients,
  fidelity kernels, and loss-landscape analysis.
