#!/usr/bin/env bash
# Report, do not fix. One run, every fact I have been guessing at.
set -uo pipefail
ps() { powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$1" 2>&1 | tr -d '\r'; }

echo "1  USERPROFILE      : [$(ps '$env:USERPROFILE')]"
echo "2  windows cwd      : [$(ps '(Get-Location).Path')]"
echo "3  /mnt/c exists    : $([ -d /mnt/c ] && echo yes || echo NO)"
echo "4  /mnt/c/Users     : $(ls /mnt/c/Users 2>/dev/null | tr '\n' ' ')"
echo "5  python on PATH   : [$(ps 'try{(Get-Command python -ErrorAction Stop).Source}catch{"none"}')]"
echo "6  python version   : [$(ps 'python --version')]"
echo "7  py launcher      : [$(ps 'try{(Get-Command py -ErrorAction Stop).Source}catch{"none"}')]"
echo "8  python3.12 path  : [$(ps 'if(Test-Path "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"){"$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"}else{"none"}')]"
echo "9  git on windows   : [$(ps 'try{(Get-Command git -ErrorAction Stop).Source}catch{"none"}')]"

echo
echo "10 WSL -> Windows write test"
probe="/mnt/c/Users/$(ls /mnt/c/Users 2>/dev/null | grep -iv 'public\|default\|all users' | head -1)"
echo "   writing to: $probe/_wsl_probe.txt"
echo hello > "$probe/_wsl_probe.txt" 2>/dev/null \
  && echo "   wsl wrote it   : yes" || echo "   wsl wrote it   : NO"
echo "   windows sees it: [$(ps 'if(Test-Path "$env:USERPROFILE\_wsl_probe.txt"){Get-Content "$env:USERPROFILE\_wsl_probe.txt"}else{"NOT VISIBLE"}')]"

echo
echo "11 does the repo copy exist on the windows side?"
echo "   [$(ps 'if(Test-Path "$env:USERPROFILE\zilver-rocm\scripts\gpu.py"){"yes"}else{"no"}')]"
