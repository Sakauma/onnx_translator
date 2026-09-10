#!/usr/bin/env bash
set -o pipefail
report_dir="docs/reports/postfix_fix_validation_2026-09-10"

{
    date --iso-8601=seconds
    printf 'head: '
    git rev-parse HEAD
    printf 'command: python -u -m pytest -q tests/test_cuda_verifier_protocol.py\n'
} > "$report_dir/gate_protocol.meta.log"
python -u -m pytest -q tests/test_cuda_verifier_protocol.py 2>&1 | tee "$report_dir/gate_protocol.stdout_stderr.log"
protocol_rc=${PIPESTATUS[0]}
printf '%s\n' "$protocol_rc" > "$report_dir/gate_protocol.rc"
if (( protocol_rc != 0 )); then
    exit "$protocol_rc"
fi

{
    date --iso-8601=seconds
    printf 'head: '
    git rev-parse HEAD
    printf 'command: python -u -m pytest -q tests/test_numerical_harness_regression.py::test_real_gpu_add_and_unique_multioutput_cleanup\n'
} > "$report_dir/gate_gpu_harness.meta.log"
python -u -m pytest -q tests/test_numerical_harness_regression.py::test_real_gpu_add_and_unique_multioutput_cleanup 2>&1 | tee "$report_dir/gate_gpu_harness.stdout_stderr.log"
gpu_rc=${PIPESTATUS[0]}
printf '%s\n' "$gpu_rc" > "$report_dir/gate_gpu_harness.rc"
exit "$gpu_rc"
