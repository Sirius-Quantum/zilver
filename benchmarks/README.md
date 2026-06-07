# Zilver benchmarks

Reproducible benchmarks you can run yourself. Nothing here is a black box — add
your own quantum account and verify the numbers on real hardware.

## `ghz_hellinger.py` — sim-to-real fidelity

The standard noise-model validation (the same GHZ + Hellinger demonstration
Qiskit Aer and others use): prepare a GHZ-N state, run it on a real QPU and on
noisy Zilver built from the device's *own* live calibration, and report the
**Hellinger distance** between the two measured distributions. One circuit, four
numbers: gate fidelity, accuracy (Hellinger), speed, cost.

### Run it — simulator only (no account, no credits)

```bash
pip install "zilver[qiskit]"
python benchmarks/ghz_hellinger.py --dry-run -n 4
```

This compares noisy Zilver vs Qiskit Aer vs the ideal GHZ state — enough to see
that Zilver's noise model agrees with Aer's, on your machine, for free.

### Run it — on real hardware (your own account)

Add a provider and the bench reads the device's live calibration, builds the
noise model from it, and fills in the QPU column:

```bash
# IBM Quantum (free Open plan).  Needs qiskit-ibm-runtime + a saved account.
pip install qiskit-ibm-runtime
python benchmarks/ghz_hellinger.py --provider ibm -n 4

# IQM Resonance.  Needs iqm-client[qiskit] + IQM_TOKEN.
pip install "iqm-client[qiskit]"
export IQM_TOKEN=...
python benchmarks/ghz_hellinger.py --provider iqm --backend garnet -n 4
```

> **Dependency note:** `qiskit-ibm-runtime` needs `qiskit>=2.2` and
> `iqm-client[qiskit]` needs `qiskit<2.2` — they cannot share one environment.
> Use a separate virtualenv per provider. The bench code is provider-agnostic;
> only the interpreter changes.

Your numbers will differ from ours — coherence times and gate errors change with
the device and the calibration of the day. That is the point: run it on *your*
target and see what Zilver predicts before you spend QPU time.

## Adding another provider

`_providers.py` implements `IBMProvider` and `IQMProvider` behind a small
interface (`calibration`, `gate_errors`, `run`). A new backend is one more class
plus an entry in `make_provider`.

## Results

See [`results/`](results/) for runs we have published as reference points — each
records the device, the live calibration used, and the four metrics, so you can
reproduce the methodology on your own hardware.
