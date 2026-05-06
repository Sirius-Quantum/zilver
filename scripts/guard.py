#!/usr/bin/env python3
"""Pre-commit secret scanner."""

import re
import sys
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Rules: (name, regex_pattern, severity)
# severity: "block" aborts the commit; "warn" prints but allows it
# ---------------------------------------------------------------------------
RULES = [
    # Secrets
    ("aws_access_key",      r"AKIA[0-9A-Z]{16}",                          "block"),
    ("aws_secret_key",      r"(?i)aws[_\-\s]?secret[_\-\s]?key\s*[=:]\s*['\"][^'\"]{20,}", "block"),
    ("generic_api_key",     r"(?i)api[_\-]?key\s*[=:]\s*['\"][^'\"]{16,}['\"]", "block"),
    ("generic_token",       r"(?i)(access|auth|bearer|secret)[_\-]?token\s*[=:]\s*['\"][^'\"]{16,}['\"]", "block"),
    ("private_key_header",  r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "block"),
    ("password_literal",    r"(?i)password\s*[=:]\s*['\"][^'\"]{4,}['\"]", "block"),
    ("database_url",        r"(?i)(postgres|mysql|mongodb|redis):\/\/[^:]+:[^@]+@", "block"),

    # Credentials pattern in env-style files
    ("dotenv_secret",       r"(?m)^[A-Z_]+(SECRET|KEY|TOKEN|PASSWORD)\s*=\s*\S+", "block"),
]

# File extensions to skip (binary / compiled)
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".pkl", ".pickle", ".npz", ".npy", ".h5",
}

# Files to always skip regardless of content
SKIP_FILENAMES = {"guard.py"}  # don't scan ourselves for rule patterns


def get_staged_files() -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True
    )
    return [Path(p) for p in result.stdout.splitlines() if p.strip()]


def get_all_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True
    )
    return [Path(p) for p in result.stdout.splitlines() if p.strip()]


def scan_file(path: Path) -> list[dict]:
    if path.suffix in SKIP_EXTENSIONS:
        return []
    if path.name in SKIP_FILENAMES:
        return []
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    findings = []
    for name, pattern, severity in RULES:
        for match in re.finditer(pattern, content):
            line_num = content[: match.start()].count("\n") + 1
            findings.append({
                "file": str(path),
                "line": line_num,
                "rule": name,
                "severity": severity,
                "snippet": match.group(0)[:80],
            })
    return findings


def check_file_size(path: Path, limit_kb: int = 500) -> dict | None:
    if not path.exists():
        return None
    size_kb = path.stat().st_size / 1024
    if size_kb > limit_kb:
        return {
            "file": str(path),
            "line": 0,
            "rule": "large_file",
            "severity": "block",
            "snippet": f"{size_kb:.0f} KB (limit {limit_kb} KB)",
        }
    return None


def run_scan(files: list[Path]) -> tuple[list[dict], list[dict]]:
    blocks, warns = [], []
    for path in files:
        for finding in scan_file(path):
            (blocks if finding["severity"] == "block" else warns).append(finding)
        size_issue = check_file_size(path)
        if size_issue:
            blocks.append(size_issue)
    return blocks, warns


def print_findings(findings: list[dict], label: str, color: str) -> None:
    reset = "\033[0m"
    for f in findings:
        loc = f"{f['file']}:{f['line']}" if f['line'] else f['file']
        print(f"  {color}[{label}]{reset} {f['rule']} in {loc}")
        print(f"         {f['snippet']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",  action="store_true", help="Scan all tracked files")
    parser.add_argument("--file", type=Path,           help="Scan a specific file")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    elif args.all:
        files = get_all_tracked_files()
    else:
        files = get_staged_files()

    if not files:
        print("guard: nothing to scan")
        return 0

    blocks, warns = run_scan(files)

    red    = "\033[31m"
    yellow = "\033[33m"
    green  = "\033[32m"
    reset  = "\033[0m"

    if warns:
        print(f"\n{yellow}guard: warnings ({len(warns)}){reset}")
        print_findings(warns, "WARN", yellow)

    if blocks:
        print(f"\n{red}guard: commit blocked ({len(blocks)} issue(s)){reset}")
        print_findings(blocks, "BLOCK", red)
        print(f"\n{red}Fix the issues above before committing.{reset}\n")
        return 1

    if not blocks and not warns:
        print(f"{green}guard: clean{reset}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
