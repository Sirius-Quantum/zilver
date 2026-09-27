# Contributing

## Setup

```bash
git clone https://github.com/sirius-quantum/zilver
cd zilver
pip install -e ".[dev,qiskit]"
python examples/vqa_optimization.py
```

Every script in `examples/` should run to completion before you start.

---

## Workflow

1. Fork and create a branch from `master`.
2. Make your change. One logical change per commit.
3. Show that it works. New behaviour needs a runnable check that compares against
   a known answer: a closed form, or Qiskit Aer. `benchmarks/statevector_vs_aer.py`
   is an example of the pattern.
4. Re-run the examples your change touches.
5. Open a PR with a clear description of what changed and why.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security problems privately, as described in [SECURITY.md](SECURITY.md).

---

## Code style

- Python 3.10+, type-annotated public APIs
- Names over comments — add comments only where logic is non-obvious
- Parameterized gates must be MLX-native (`mx.cos` / `mx.sin`) — no `float()` calls inside `mx.vmap`
- No hardcoded secrets or absolute paths

The pre-commit guard blocks API keys, tokens, and files over 500 KB:

```bash
python scripts/guard.py --all
```

---

## Good first issues

- Additional noise models for the density matrix backend
- OpenQASM 3.0 import (the bridge reads OpenQASM 2.0 today)
- Benchmark comparisons against PennyLane

---

## License

Contributions are licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
