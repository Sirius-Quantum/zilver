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

## [0.4.0]

- Statevector backends (`metal`, `accel`, `mlx`), parameter-shift gradients,
  fidelity kernels, and loss-landscape analysis.
