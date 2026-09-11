#!/usr/bin/env bash
set -u
export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
main=/mnt/d/workspace/onnx_translator
report=/mnt/d/workspace/onnx_translator_bugfix_worktree/docs/reports/dql_zero_reference_2026-09-11
cd "$main"
cmd=(/home/sakauma/data/miniconda3/envs/egor/bin/python -u tools/cli.py compile-cuda --op dynamic_quantize_linear --force)
printf '%q ' "${cmd[@]}" > "$report/main_compile_cuda.command.txt"; printf '\n' >> "$report/main_compile_cuda.command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$report/main_compile_cuda.start.txt"
"${cmd[@]}" > "$report/main_compile_cuda.stdout.txt" 2> "$report/main_compile_cuda.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_compile_cuda.rc.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$report/main_compile_cuda.end.txt"
sha256sum tensor_ops.so cache/verify_dynamic_quantize_linear > "$report/main_binaries.sha256"
stat -c '%n %s bytes' tensor_ops.so cache/verify_dynamic_quantize_linear > "$report/main_binaries.stat.txt"
exit "$rc"