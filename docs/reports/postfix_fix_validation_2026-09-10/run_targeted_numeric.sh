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
    printf 'python: '
    command -v python
    python --version
} > "$report_dir/numeric_targeted.meta.log" 2>&1

printf '%s\n' 'command: python -u -m pytest -q tests/test_operator_misc_semantics.py tests/test_operator_reduce_semantics.py' \
    > "$report_dir/numeric_semantics.command"
python -u -m pytest -q \
    tests/test_operator_misc_semantics.py \
    tests/test_operator_reduce_semantics.py \
    2>&1 | tee "$report_dir/numeric_semantics.stdout_stderr.log"
pytest_rc=${PIPESTATUS[0]}
printf '%s\n' "$pytest_rc" > "$report_dir/numeric_semantics.rc"

if [[ "${1:-}" == "semantics-only" ]]; then
    exit "$pytest_rc"
fi

printf '%s\n' 'command: python -u tools/cli.py numerical --op dynamic_quantize_linear --iterations 1' \
    > "$report_dir/dql_numerical.command"
python -u tools/cli.py numerical --op dynamic_quantize_linear --iterations 1 \
    2>&1 | tee "$report_dir/dql_numerical.stdout_stderr.log"
numerical_rc=${PIPESTATUS[0]}
printf '%s\n' "$numerical_rc" > "$report_dir/dql_numerical.rc"

if (( pytest_rc != 0 )); then
    exit "$pytest_rc"
fi
exit "$numerical_rc"
