#!/usr/bin/env bash
set -o pipefail

report_dir="docs/reports/postfix_fix_validation_2026-09-10"
mkdir -p "$report_dir"

{
    date --iso-8601=seconds
    printf 'command: python -u -m pytest -q tests/test_operator_sequence_control.py\n'
    printf 'cwd: '
    pwd
    printf 'head: '
    git rev-parse HEAD
    printf 'branch: '
    git branch --show-current
    printf 'python: '
    command -v python
    python --version
    printf 'nvcc: '
    command -v nvcc
    nvcc --version | tail -n 1
    printf 'gpu: '
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
} > "$report_dir/control_targeted.meta.log" 2>&1

python -u -m pytest -q tests/test_operator_sequence_control.py \
    2>&1 | tee "$report_dir/control_targeted.stdout_stderr.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$report_dir/control_targeted.rc"
exit "$rc"
