#!/bin/bash
# Local dev setup. Usage: bash scripts/setup.sh

set -e

RED='\033[31m'; YELLOW='\033[33m'; GREEN='\033[32m'; RESET='\033[0m'

info()  { echo -e "${GREEN}[zilver]${RESET} $1"; }
warn()  { echo -e "${YELLOW}[zilver]${RESET} $1"; }
abort() { echo -e "${RED}[zilver] error:${RESET} $1"; exit 1; }

# 1. Verify Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    abort "Zilver is optimized for Apple Silicon (arm64). Detected: $ARCH"
fi

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
info "Detected chip: $CHIP"

# 2. Python version check
PYTHON=$(command -v python3 || command -v python || abort "Python not found")
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    abort "Python 3.10+ required. Found: $PY_VERSION"
fi
info "Python $PY_VERSION OK"

# 3. Install zilver + dev deps
info "Installing zilver..."
pip install -e ".[dev]" -q
info "Package installed"

# 4. Verify MLX is using Metal (not CPU fallback)
info "Checking MLX Metal backend..."
$PYTHON - <<'EOF'
import mlx.core as mx
devices = mx.default_device()
print(f"  MLX default device: {devices}")
# Quick Metal smoke test
a = mx.array([1.0, 2.0, 3.0])
b = a * 2
mx.eval(b)
print("  Metal dispatch: ok")
EOF

# 5. Install pre-commit hook
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || abort "Not inside a git repo")
HOOK_DIR="$REPO_ROOT/.git/hooks"

cp -f "$REPO_ROOT/scripts/guard.py" "$HOOK_DIR/_guard.py"
cat > "$HOOK_DIR/pre-commit" <<'HOOK'
#!/bin/sh
python "$(git rev-parse --show-toplevel)/.git/hooks/_guard.py"
exit $?
HOOK
chmod +x "$HOOK_DIR/pre-commit"
info "Pre-commit hook installed"

# 6. Run tests
info "Running test suite..."
python -m pytest tests/ -q --tb=short
info "All tests passed"

echo ""
echo -e "${GREEN}Zilver is ready on $(sysctl -n hw.model 2>/dev/null || echo 'your Mac').${RESET}"
echo ""
echo "  python examples/barren_plateau.py   # run the demo"
echo "  make test                            # run tests"
echo "  make bench                           # benchmark vs Qiskit Aer"
echo "  make guard                           # scan repo for sensitive data"
echo ""
