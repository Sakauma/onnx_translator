#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
cd /mnt/d/workspace/onnx_translator_bugfix_worktree

report=docs/reports/dql_zero_reference_2026-09-11
command=(
  /home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q
  tests/test_numerical_dql_zero_reference.py
  tests/test_reaudit_special_output_contracts.py
  tests/test_numerical_runner_architecture.py
  tests/test_numerical_harness_regression.py
)

printf '%q ' "${command[@]}" > "$report/harness_targeted.command.txt"
printf '\n' >> "$report/harness_targeted.command.txt"
printf '%s\n' \
  'Non-GPU targeted scope: new DQL zero Reference profile; mocked special-output contracts; runner architecture; numerical harness regressions. Real GPU tests excluded.' \
  > "$report/harness_targeted.scope.txt"

"${command[@]}" > "$report/harness_targeted.stdout.txt" 2> "$report/harness_targeted.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/harness_targeted.rc.txt"
exit "$rc"
