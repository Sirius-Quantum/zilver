"""Run the simulator on the GPU and print one line per width.

    pip install zilver
    python -m zilver.gpu                  # 20 -> 32 qubits, stops early where memory runs out
    FROM=24 TO=30 python -m zilver.gpu    # PowerShell: $env:FROM=24; $env:TO=30

The circuit at each width is a Hadamard and a Y-rotation on every qubit, then a
CNOT chain. Prints the device it got, then seconds and the state norm per width.
Norm is the correctness check that costs nothing -- a unitary circuit ends at
1.0, so anything else means the arithmetic drifted.
"""
import faulthandler
import os
import subprocess
import sys
import time


def _relaunch_on_gpu_backend():
    """Make sure the backend is chosen as the GPU one, before anything is timed.

    zilver picks its array backend when the package is first imported, and for
    `python -m zilver.gpu` that happens before this module runs. With
    ZILVER_BACKEND unset, a machine without MLX falls back to the numpy CPU
    path, and this runner would time the CPU while claiming the GPU. So when the
    variable is unset, set it and run once more in a fresh process, where the
    choice is made correctly from the start. An explicit setting is respected.
    """
    if "ZILVER_BACKEND" in os.environ:
        return
    env = dict(os.environ, ZILVER_BACKEND="torch")
    sys.exit(subprocess.call([sys.executable, "-m", "zilver.gpu", *sys.argv[1:]], env=env))


def main():
    # DirectML aborts the process on an unsupported dtype rather than raising, so
    # a normal traceback never appears. faulthandler prints the Python stack on a
    # fatal signal, which is the only way to see WHERE from outside the box.
    faulthandler.enable()
    _relaunch_on_gpu_backend()

    import numpy as np
    import zilver._array as _a
    from zilver.circuit import Circuit

    def _free():
        """Release cached device blocks between widths, whatever the backend."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            _a.mx.clear_cache()          # MLX's own pool, on Apple silicon
        except Exception:
            pass

    print(f"\n  device      : {getattr(_a, 'TORCH_DEVICE', 'cpu')}")
    print(f"  complex64   : {_a.HAS_COMPLEX}")
    print(f"\n{'qubits':>7}{'state GB':>10}{'seconds':>10}{'norm':>12}")

    for n in range(int(os.environ.get("FROM", "20")), int(os.environ.get("TO", "32")) + 1):
        gb = (2 ** n) * 8 / 1e9
        c = Circuit(n); pi = 0
        for q in range(n):
            c.h(q); c.ry(q, pi); pi += 1
        for q in range(n - 1):
            c.cnot(q, q + 1)
        p = [0.1 * (i + 1) for i in range(pi)]
        try:
            t = time.perf_counter()
            # method="mlx" is the array-layer path -- the one that runs on the
            # device. "auto" picks accel/numba when it is installed, which is a
            # CPU path, so the default would time the CPU while printing a GPU name.
            v = np.asarray(c.statevector(p, method="mlx").numpy())
            dt = time.perf_counter() - t
        except Exception as e:
            print(f"{n:>7}{gb:>10.2f}   {type(e).__name__}: {str(e)[:50]}")
            break
        if v.ndim == 2 and v.shape[0] == 2:
            v = v[0] + 1j * v[1]
        print(f"{n:>7}{gb:>10.2f}{dt:>10.2f}{np.linalg.norm(v):>12.7f}", flush=True)
        # Hand the pool back before the next width. A caching allocator keeps
        # freed blocks, so without this the run dies with "free: 0" out of a
        # mostly empty 49 GiB -- fragmentation, not capacity, and it costs a qubit.
        del v
        _free()


if __name__ == "__main__":
    main()
