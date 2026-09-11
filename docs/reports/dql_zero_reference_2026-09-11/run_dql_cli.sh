#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
cd /mnt/d/workspace/onnx_translator_bugfix_worktree

report=docs/reports/dql_zero_reference_2026-09-11
command=(
  /home/sakauma/data/miniconda3/envs/egor/bin/python
  -u
  tools/cli.py numerical
  --op dynamic_quantize_linear
  --iterations 3
  --skip-plots
)

printf '%q ' "${command[@]}" > "$report/dql_cli.command.txt"
printf '\n' >> "$report/dql_cli.command.txt"
printf '%s\n' \
  'Exact CLI filter: dynamic_quantize_linear only; three registered plans (one regular and two zero-reference plans); three iterations per plan; no rebuild.' \
  > "$report/dql_cli.scope.txt"

{
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'source_head=%s\n' "$(git rev-parse HEAD)"
  printf 'tensor_ops_sha256=%s\n' "$(sha256sum tensor_ops.so | cut -d' ' -f1)"
  printf 'dql_cuda_sha256=%s\n' "$(sha256sum cache/verify_dynamic_quantize_linear | cut -d' ' -f1)"
} > "$report/dql_cli.meta.txt"

"${command[@]}" > "$report/dql_cli.stdout.txt" 2> "$report/dql_cli.stderr.txt"
rc=$?
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/dql_cli.meta.txt"
printf '%s\n' "$rc" > "$report/dql_cli.rc.txt"
exit "$rc"
