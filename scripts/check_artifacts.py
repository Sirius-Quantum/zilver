#!/usr/bin/env python3
"""Check that the built wheel and sdist contain exactly what hatch.toml allows.

    python3 scripts/check_artifacts.py              # build into a temp dir, then check
    python3 scripts/check_artifacts.py --dist DIR   # check artifacts already built in DIR

WHY THIS EXISTS. What reaches PyPI is decided by the BUILD, not by git. A file
sitting in the package directory is a candidate for the wheel whether or not it
is tracked, so a pre-commit hook cannot see it: hooks only ever look at staged
files. The one channel that actually publishes has had no check on it at all.

WHY IT ASKS THE QUESTION THIS WAY. It does not carry a list of things to look
for. A list of what-must-not-ship, living in a public repository, has to spell
out what it is hiding -- which is the problem it would be there to prevent.
Asking instead "is everything in here allowed?" needs no such list, and it also
catches files nobody thought to name.

Exit status is 0 when both artifacts match the allowlist, 1 otherwise.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        sys.exit("needs Python 3.11+, or `pip install tomli` on 3.10")

ROOT = Path(__file__).resolve().parent.parent

# Files the backend adds to an sdist on its own. They are not in hatch.toml's
# lists and their presence is expected, not a finding.
SDIST_AUTO = {"PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "hatch.toml", ".gitignore"}


def allowed(target: str) -> set[str]:
    """The only-include entries for one build target, as package-relative paths.

    A missing or reshaped hatch.toml is a hard failure, never a pass. If this
    file is deleted, or its settings are moved into pyproject.toml, hatchling
    falls back to publishing the whole package directory -- which is the exact
    accident this script exists to catch, so it must not be the case that the
    check quietly succeeds when its own input has gone.
    """
    path = ROOT / "hatch.toml"
    if not path.is_file():
        sys.exit(f"{path} is missing: without it the build publishes the whole "
                 f"package directory, and there is no allowlist to check against.")
    cfg = tomllib.loads(path.read_text())
    try:
        entries = cfg["build"]["targets"][target]["only-include"]
    except KeyError:
        sys.exit(f"hatch.toml has no [build.targets.{target}] only-include list. "
                 f"An exclude list is not equivalent: it publishes new files by "
                 f"default, which is what this check is here to prevent.")
    return {e[len("src/"):] if e.startswith("src/") else e for e in entries}


def _is_under(path: str, allowlist: set[str]) -> bool:
    """True when path is an allowed file, or sits under an allowed directory."""
    return path in allowlist or any(path.startswith(a.rstrip("/") + "/") for a in allowlist)


def check_wheel(whl: Path) -> list[str]:
    ok = allowed("wheel")
    bad = []
    with zipfile.ZipFile(whl) as z:
        for name in z.namelist():
            if name.endswith("/") or ".dist-info/" in name:
                continue
            if not _is_under(name, ok):
                bad.append(name)
    return bad


def check_sdist(tgz: Path) -> list[str]:
    ok = allowed("sdist")
    bad = []
    with tarfile.open(tgz) as t:
        for member in t.getmembers():
            if not member.isfile():
                continue
            # strip the leading "zilver-<version>/" directory
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if rel in SDIST_AUTO:
                continue
            probe = rel[len("src/"):] if rel.startswith("src/") else rel
            if not (_is_under(probe, ok) or _is_under(rel, ok)):
                bad.append(rel)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, help="check artifacts already built here")
    args = ap.parse_args()

    tmp = None
    if args.dist:
        dist = args.dist
    else:
        tmp = tempfile.mkdtemp(prefix="zilver-artifact-check-")
        dist = Path(tmp)
        print(f"building into {dist} …")
        r = subprocess.run([sys.executable, "-m", "build", "--outdir", str(dist)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            return 1

    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if not wheels or not sdists:
        print(f"no wheel and sdist pair in {dist}")
        return 1

    failures = 0
    for whl in wheels:
        bad = check_wheel(whl)
        print(f"  wheel {whl.name}: {'OK' if not bad else f'{len(bad)} not allowed'}")
        for b in bad:
            print(f"    NOT ALLOWED: {b}")
        failures += len(bad)

    for tgz in sdists:
        bad = check_sdist(tgz)
        print(f"  sdist {tgz.name}: {'OK' if not bad else f'{len(bad)} not allowed'}")
        for b in bad:
            print(f"    NOT ALLOWED: {b}")
        failures += len(bad)

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\nartifact check FAILED: the build contains files the allowlist does "
              "not name. Add them to hatch.toml deliberately, or find out why they "
              "are in the package directory.")
        return 1
    print("\nartifact check passed: both artifacts match the allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
