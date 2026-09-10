#!/usr/bin/env bash
set -o pipefail

report_dir="docs/reports/postfix_fix_validation_2026-09-10"
mkdir -p "$report_dir"

{
    date --iso-8601=seconds
    printf 'cwd: '
    pwd
    printf 'head: '
    git rev-parse HEAD
    printf 'branch: '
    git branch --show-current
    printf 'git-status:\n'
    git status --short --branch
    printf 'uname: '
    uname -a
    printf 'python-path: '
    command -v python
    python --version
    python -c 'import numpy, onnx, torch; print("numpy:", numpy.__version__); print("onnx:", onnx.__version__); print("torch:", torch.__version__)'
    gcc --version | head -n 1
    nvcc --version | tail -n 1
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} 2>&1 | tee "$report_dir/environment.stdout_stderr.log"
environment_rc=${PIPESTATUS[0]}
printf '%s\n' "$environment_rc" > "$report_dir/environment.rc"
exit "$environment_rc"
