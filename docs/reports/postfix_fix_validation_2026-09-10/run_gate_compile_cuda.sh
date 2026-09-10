#!/usr/bin/env bash
set -o pipefail

report_dir="docs/reports/postfix_fix_validation_2026-09-10"
mkdir -p "$report_dir"
source_count=$(find cuda -maxdepth 1 -type f -name 'verify_*.cu' | wc -l)

{
    date --iso-8601=seconds
    printf 'command: python -u tools/cli.py compile-cuda\n'
    printf 'head: '
    git rev-parse HEAD
    printf 'cuda-source-count=%s\n' "$source_count"
    printf 'cache-verifier-count-before='
    find cache -maxdepth 1 -type f -name 'verify_*' 2>/dev/null | wc -l
} > "$report_dir/gate_compile_cuda.meta.log" 2>&1

python -u tools/cli.py compile-cuda \
    2>&1 | tee "$report_dir/gate_compile_cuda.stdout_stderr.log"
gate_rc=${PIPESTATUS[0]}
verifier_count=$(find cache -maxdepth 1 -type f -name 'verify_*' 2>/dev/null | wc -l)
compiled=$(sed -n 's/.*compilation succeeded\. compiled=\([0-9][0-9]*\) skipped=.*/\1/p' "$report_dir/gate_compile_cuda.stdout_stderr.log" | tail -n 1)
skipped=$(sed -n 's/.*compilation succeeded\. compiled=[0-9][0-9]* skipped=\([0-9][0-9]*\).*/\1/p' "$report_dir/gate_compile_cuda.stdout_stderr.log" | tail -n 1)
{
    printf 'rc=%s\n' "$gate_rc"
    printf 'source_count=%s\n' "$source_count"
    printf 'compiled=%s\n' "$compiled"
    printf 'skipped=%s\n' "$skipped"
    printf 'cache_verifier_count=%s\n' "$verifier_count"
    printf 'finished_at='
    date --iso-8601=seconds
} > "$report_dir/gate_compile_cuda.rc"
exit "$gate_rc"
