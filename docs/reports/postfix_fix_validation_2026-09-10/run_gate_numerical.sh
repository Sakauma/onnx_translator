#!/usr/bin/env bash
set -o pipefail

report_dir="docs/reports/postfix_fix_validation_2026-09-10"
mkdir -p "$report_dir"

{
    date --iso-8601=seconds
    printf 'command: python -u tools/cli.py numerical --iterations 3 --skip-plots\n'
    printf 'head: '
    git rev-parse HEAD
    printf 'branch: '
    git branch --show-current
    printf 'tracked-status-before:\n'
    git status --short --untracked-files=no
} > "$report_dir/gate_numerical.meta.log" 2>&1

python -u tools/cli.py numerical --iterations 3 --skip-plots \
    2>&1 | tee "$report_dir/gate_numerical.stdout_stderr.log"
gate_rc=${PIPESTATUS[0]}
plan_count=$(grep -c '🧪 Testing ' "$report_dir/gate_numerical.stdout_stderr.log" || true)
pass_count=$(grep -c '✅ Pass (3/3)' "$report_dir/gate_numerical.stdout_stderr.log" || true)
failed_iteration_count=$(grep -c '❌ Iter ' "$report_dir/gate_numerical.stdout_stderr.log" || true)
plan_exception_count=$(grep -c 'ERROR: numerical plan failed before completion' "$report_dir/gate_numerical.stdout_stderr.log" || true)
{
    printf 'rc=%s\n' "$gate_rc"
    printf 'plan_count=%s\n' "$plan_count"
    printf 'pass_count=%s\n' "$pass_count"
    printf 'iteration_count=%s\n' "$((pass_count * 3))"
    printf 'failed_iteration_count=%s\n' "$failed_iteration_count"
    printf 'plan_exception_count=%s\n' "$plan_exception_count"
    printf 'finished_at='
    date --iso-8601=seconds
} > "$report_dir/gate_numerical.rc"
exit "$gate_rc"
