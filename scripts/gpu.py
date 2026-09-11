"""Run the simulator on the GPU from a clone. The runner itself is zilver.gpu.

    python3 scripts/gpu.py            # 20 -> 32 qubits, stops early where memory runs out
    FROM=24 TO=30 python3 scripts/gpu.py

Installed users run the same thing with `python -m zilver.gpu`.
"""
import os
import sys

# Relative to this file, not the working directory, so it runs from anywhere, and
# the backend is chosen before zilver is imported.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("ZILVER_BACKEND", "torch")

from zilver.gpu import main  # noqa: E402

main()
