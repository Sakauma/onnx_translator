#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
workspace=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$workspace/docs/reports/post_push_fixes_2026-09-11"
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
expected_head=fa095a914ac71ef05650806f76f813b94ac8c198
expected_tensor_ops_sha=e6a720b448b03113da11e7310ce15bcdd326c026c2147d2aabe187b577c046a9
cd "$workspace"

command=("$python" -u -m pytest -q -ra tests)
printf '%q ' "${command[@]}" > "$report/final_pytest.command.txt"
printf '\n' >> "$report/final_pytest.command.txt"
{
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'tensor_ops_sha256=%s\n' "$(sha256sum tensor_ops.so | cut -d' ' -f1)"
} > "$report/final_pytest.meta.txt"
actual_head=$(git rev-parse HEAD)
actual_tensor_ops_sha=$(sha256sum tensor_ops.so | cut -d' ' -f1)
if [[ "$actual_head" != "$expected_head" || "$actual_tensor_ops_sha" != "$expected_tensor_ops_sha" ]]; then
  printf '%s\n' 'Source or tensor_ops.so identity mismatch; pytest was not started.' \
    > "$report/final_pytest.stderr.txt"
  : > "$report/final_pytest.stdout.txt"
  printf '%s\n' '125' > "$report/final_pytest.rc.txt"
  exit 125
fi
"${command[@]}" > "$report/final_pytest.stdout.txt" 2> "$report/final_pytest.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/final_pytest.rc.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/final_pytest.meta.txt"
exit "$rc"
