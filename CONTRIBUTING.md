# Contributing

## Setup

```bash
git clone https://github.com/deryakarl/zilver
cd zilver
pip install -e ".[dev,network]"
pytest tests/
```

All 464 tests must pass before you start.

---

## Workflow

1. Fork and create a branch from `master`.
2. Make your change. One logical change per commit.
3. Add tests — new behaviour needs coverage.
4. `pytest tests/` — all green.
5. Open a PR with a clear description of what changed and why.

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
- QASM / OpenQASM 2.0 import bridge
- Benchmark comparisons against Qiskit Aer and PennyLane

---

## License

Contributions are licensed under Apache 2.0.
