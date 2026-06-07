# Reference results — GHZ-4 sim-to-real

These are runs we published as reference points. Each was produced by
`ghz_hellinger.py --provider <ibm|iqm> -n 4 --shots 4096` against the live
calibration of the day. **Reproduce them on your own account** — your numbers
will differ with the device and the calibration, which is exactly what the bench
is for.

The metric that matters: **Hellinger distance between noisy Zilver and the real
QPU**, compared against Qiskit Aer built from the same calibration. Lower = the
noise model is closer to the hardware; < 0.1 is the conventional "matches
hardware" threshold.

## IBM Marrakesh (Heron r2) — 2026-06

Live calibration: T1 ≈ 268 µs, T2 ≈ 186 µs, 2q RB fidelity 99.76%.

| metric | Zilver | Qiskit Aer | QPU |
|---|---|---|---|
| gate fidelity | 1-F = 6e-08 (machine precision) | exact | 2q RB 99.76% |
| accuracy (Hellinger vs QPU) | **0.105** | 0.111 | reference |
| speed | ~7 ms | ~10 ms | ~7 s (queue+run) |
| cost | $0 | $0 | free (Open plan) |

Noisy Zilver predicted the real GHZ-4 output to **H = 0.105** — at the "matches
hardware" line — and closer than Aer (0.111).

## IQM Garnet — 2026-06

Live calibration: T1 ≈ 28 µs, T2-echo ≈ 11 µs, 2q RB fidelity 98.95%.

| metric | Zilver | Qiskit Aer | QPU |
|---|---|---|---|
| gate fidelity | 1-F = 6e-08 | exact | 2q RB 98.95% |
| accuracy (Hellinger vs QPU) | **0.169** | 0.186 | reference |
| cost | $0 | $0 | ~3 credits / job |

Lower-coherence device → larger Hellinger for both simulators, and Zilver again
closer to the hardware than Aer.

## What this shows

On both machines, **noisy Zilver — running locally on an Apple Silicon Mac for
free — is at least as faithful as Qiskit Aer at predicting real-hardware output.**
The residual to the < 0.1 bar is unmodeled readout/crosstalk noise (the same for
both simulators). Run it yourself and check.
