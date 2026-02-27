.PHONY: install install-hooks test bench guard clean

install:
	pip install -e ".[dev]"

install-hooks:
	@cp -f scripts/guard.py .git/hooks/_guard.py
	@printf '#!/bin/sh\npython "$(shell git rev-parse --show-toplevel)/.git/hooks/_guard.py"\nexit $$?' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "pre-commit hook installed"

setup: install install-hooks
	@echo "Zilver ready"

test:
	python -m pytest tests/ -q

bench:
	python benchmarks/vs_qiskit_aer.py --qubits 6 --depth 3 --resolution 20

guard:
	python scripts/guard.py --all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
