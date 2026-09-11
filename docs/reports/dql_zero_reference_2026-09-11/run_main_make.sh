#!/usr/bin/env bash
set -u
export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
main=/mnt/d/workspace/onnx_translator
report=/mnt/d/workspace/onnx_translator_bugfix_worktree/docs/reports/dql_zero_reference_2026-09-11
cd "$main"
printf '%s\n' 'make' > "$report/main_make.command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$report/main_make.start.txt"
make > "$report/main_make.stdout.txt" 2> "$report/main_make.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_make.rc.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$report/main_make.end.txt"
exit "$rc"