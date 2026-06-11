# Changelog

All notable changes to Zilver are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

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
