#!/usr/bin/env bash
set -u
export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
main=/mnt/d/workspace/onnx_translator
report=/mnt/d/workspace/onnx_translator_bugfix_worktree/docs/reports/dql_zero_reference_2026-09-11
cd "$main"
pytest_cmd=(/home/sakauma/data/miniconda3/envs/egor/bin/python -u -m pytest -q tests/test_dynamic_quantize_zero_reference.py)
printf '%q ' "${pytest_cmd[@]}" > "$report/main_reference_pytest.command.txt"; printf '\n' >> "$report/main_reference_pytest.command.txt"
"${pytest_cmd[@]}" > "$report/main_reference_pytest.stdout.txt" 2> "$report/main_reference_pytest.stderr.txt"
pytest_rc=$?; printf '%s\n' "$pytest_rc" > "$report/main_reference_pytest.rc.txt"
[[ "$pytest_rc" -eq 0 ]] || exit "$pytest_rc"
cli_cmd=(/home/sakauma/data/miniconda3/envs/egor/bin/python -u tools/cli.py numerical --op dynamic_quantize_linear --iterations 3 --skip-plots)
printf '%q ' "${cli_cmd[@]}" > "$report/main_dql_cli.command.txt"; printf '\n' >> "$report/main_dql_cli.command.txt"
"${cli_cmd[@]}" > "$report/main_dql_cli.stdout.txt" 2> "$report/main_dql_cli.stderr.txt"
cli_rc=$?; printf '%s\n' "$cli_rc" > "$report/main_dql_cli.rc.txt"
exit "$cli_rc"