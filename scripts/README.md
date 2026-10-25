# scripts

Project tooling and machine checks. None of these are needed to use Zilver.

**Development and release**

| Script | Purpose |
|---|---|
| `setup.sh` | Local development setup on Apple silicon. |
| `guard.py` | Pre-commit check for secrets (`make install-hooks` installs it). |
| `check_artifacts.py` | Checks that the built wheel and sdist contain exactly what `hatch.toml` lists. |
| `gpu.py` | Runs `python -m zilver.gpu` from a clone. |

**Machine checks**

| Script | Purpose |
|---|---|
| `x86-node-check.sh` | Correctness and size check on a non-Apple machine. |
| `unified-memory-ceiling.sh` | How many qubits this machine's memory holds. |
| `cpu_vs_gpu.py`, `cpu-vs-igpu.sh` | The same statevector on the CPU and the GPU, compared element by element. |
| `rocm-win.sh`, `rocm-win-zilver.sh`, `windows-rocm.ps1` | Set up native Windows ROCm on a Radeon 8060S from WSL, and run Zilver there. |
| `win-facts.sh`, `dml-probe.sh` | Report what a Windows GPU setup supports (ROCm, DirectML). |
