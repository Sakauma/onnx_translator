#!/usr/bin/env bash
set -u

repo=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$repo/docs/reports/reaudit_fix_2026-09-11"
expected_source_sha=b8d90098176f0bec35c1ddea334781697f9881b7

cd "$repo" || exit 125
actual_source_sha=$(git rev-parse HEAD)
if [[ "$actual_source_sha" != "$expected_source_sha" ]]; then
    printf '%s\n' "126" >"$report/numerical_gate_attempt2.rc.txt"
    printf '%s\n' \
        "expected_source_sha=$expected_source_sha" \
        "actual_source_sha=$actual_source_sha" \
        "wrapper_rc=126" >"$report/numerical_gate_attempt2.meta.txt"
    printf '%s\n' \
        "refusing numerical attempt2: source SHA mismatch" \
        "expected: $expected_source_sha" \
        "actual:   $actual_source_sha" >"$report/numerical_gate_attempt2.stderr.txt"
    : >"$report/numerical_gate_attempt2.stdout.txt"
    exit 126
fi

exec bash "$report/run_numerical_gate.sh" numerical_gate_attempt2
