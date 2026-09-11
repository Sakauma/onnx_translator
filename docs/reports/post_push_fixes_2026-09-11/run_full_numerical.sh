#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
workspace=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$workspace/docs/reports/post_push_fixes_2026-09-11"
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
expected_head=e0a03c6b3e71ba5c9562624f783c3ac6e4165322
expected_tensor_ops_sha=6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89
expected_layernorm_sha=cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074
expected_inventory_sha=0ee9df10a281dd557d1615a5781a980de0c79943a5b7853dc8f7471830a2ed8a
cd "$workspace"

command=("$python" -u tools/cli.py numerical --iterations 3 --skip-plots)
printf '%q ' "${command[@]}" > "$report/final_numerical.command.txt"
printf '\n' >> "$report/final_numerical.command.txt"
{
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'tensor_ops_sha256=%s\n' "$(sha256sum tensor_ops.so | cut -d' ' -f1)"
  printf 'verifier_count=%s\n' "$(find cache -maxdepth 1 -type f -name 'verify_*' | wc -l)"
  printf 'iterations_per_plan=3\n'
} > "$report/final_numerical.meta.txt"
find cache -maxdepth 1 -type f -name 'verify_*' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$report/final_numerical.preflight_inventory.sha256"
actual_head=$(git rev-parse HEAD)
actual_tensor_ops_sha=$(sha256sum tensor_ops.so | cut -d' ' -f1)
actual_layernorm_sha=$(sha256sum cache/verify_layer_normalization | cut -d' ' -f1)
actual_inventory_sha=$(sha256sum "$report/final_numerical.preflight_inventory.sha256" | cut -d' ' -f1)
if [[ "$actual_head" != "$expected_head" \
  || "$actual_tensor_ops_sha" != "$expected_tensor_ops_sha" \
  || "$actual_layernorm_sha" != "$expected_layernorm_sha" \
  || "$actual_inventory_sha" != "$expected_inventory_sha" ]]; then
  printf '%s\n' 'Source or binary inventory identity mismatch; numerical validation was not started.' \
    > "$report/final_numerical.stderr.txt"
  : > "$report/final_numerical.stdout.txt"
  printf '%s\n' '125' > "$report/final_numerical.rc.txt"
  exit 125
fi
"${command[@]}" > "$report/final_numerical.stdout.txt" 2> "$report/final_numerical.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/final_numerical.rc.txt"
printf 'actual_plan_count=%s\n' "$(grep -c 'Testing ' "$report/final_numerical.stdout.txt" || true)" >> "$report/final_numerical.meta.txt"
printf 'actual_pass_count=%s\n' "$(grep -c 'Pass (3/3)' "$report/final_numerical.stdout.txt" || true)" >> "$report/final_numerical.meta.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/final_numerical.meta.txt"
exit "$rc"
