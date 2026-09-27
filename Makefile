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
	@for f in examples/*.py; do case $$f in *network_submit*) continue;; esac; \
	  echo "== $$f"; MPLBACKEND=Agg python $$f > /dev/null || exit 1; done; echo "examples: all ran"

bench:
	python benchmarks/statevector_vs_aer.py

guard:
	python scripts/guard.py --all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
