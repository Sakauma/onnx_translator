#!/usr/bin/env bash
set -o pipefail

report_dir="docs/reports/postfix_fix_validation_2026-09-10"
mkdir -p "$report_dir"

{
    date --iso-8601=seconds
    printf 'command: python -u tools/verify_all.py --skip-cuda --keep-artifacts\n'
    printf 'head: '
    git rev-parse HEAD
    printf 'branch: '
    git branch --show-current
    printf 'tracked-status-before:\n'
    git status --short --untracked-files=no
} > "$report_dir/gate_cpu.meta.log" 2>&1

python -u tools/verify_all.py --skip-cuda --keep-artifacts \
    2>&1 | tee "$report_dir/gate_cpu.stdout_stderr.log"
gate_rc=${PIPESTATUS[0]}
{
    printf 'rc=%s\n' "$gate_rc"
    printf 'finished_at='
    date --iso-8601=seconds
} > "$report_dir/gate_cpu.rc"
exit "$gate_rc"
