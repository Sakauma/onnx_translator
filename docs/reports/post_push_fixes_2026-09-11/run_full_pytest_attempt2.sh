#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
workspace=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$workspace/docs/reports/post_push_fixes_2026-09-11"
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
expected_head=e0a03c6b3e71ba5c9562624f783c3ac6e4165322
expected_tensor_ops_sha=6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89
cd "$workspace"

command=("$python" -u -m pytest -q -ra tests)
printf '%q ' "${command[@]}" > "$report/final_pytest_attempt2.command.txt"
printf '\n' >> "$report/final_pytest_attempt2.command.txt"
{
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'tensor_ops_sha256=%s\n' "$(sha256sum tensor_ops.so | cut -d' ' -f1)"
} > "$report/final_pytest_attempt2.meta.txt"
actual_head=$(git rev-parse HEAD)
actual_tensor_ops_sha=$(sha256sum tensor_ops.so | cut -d' ' -f1)
if [[ "$actual_head" != "$expected_head" || "$actual_tensor_ops_sha" != "$expected_tensor_ops_sha" ]]; then
  printf '%s\n' 'Source or tensor_ops.so identity mismatch; pytest was not started.' \
    > "$report/final_pytest_attempt2.stderr.txt"
  : > "$report/final_pytest_attempt2.stdout.txt"
  printf '%s\n' '125' > "$report/final_pytest_attempt2.rc.txt"
  exit 125
fi
"${command[@]}" > "$report/final_pytest_attempt2.stdout.txt" 2> "$report/final_pytest_attempt2.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/final_pytest_attempt2.rc.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/final_pytest_attempt2.meta.txt"
exit "$rc"
