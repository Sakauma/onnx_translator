#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
cd /mnt/d/workspace/onnx_translator_bugfix_worktree

report=docs/reports/dql_zero_reference_2026-09-11
expected_source_head=a1d4940cc694ac65fbb77a993b7b800fcafd6ee5
expected_tensor_ops_sha=4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6
expected_dql_cuda_sha=2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e
expected_verifier_count=178
expected_plan_count=725
iterations=3
expected_iteration_count=2175

command=(
  /home/sakauma/data/miniconda3/envs/egor/bin/python
  -u
  tools/cli.py numerical
  --iterations "$iterations"
  --skip-plots
)

printf '%q ' "${command[@]}" > "$report/full_numerical.command.txt"
printf '\n' >> "$report/full_numerical.command.txt"
printf '%s\n' \
  'Full default numerical inventory; no CUDA rebuild; three iterations per registered plan; ONNX ReferenceEvaluator compatibility profile for finite non-empty all-zero DynamicQuantizeLinear.' \
  > "$report/full_numerical.scope.txt"

start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_head=$(git rev-parse HEAD)
tensor_ops_sha=$(sha256sum tensor_ops.so | cut -d' ' -f1)
dql_cuda_sha=$(sha256sum cache/verify_dynamic_quantize_linear | cut -d' ' -f1)

find cache -maxdepth 1 -type f -name 'verify_*' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$report/full_numerical.cuda_inventory.txt"
verifier_count=$(wc -l < "$report/full_numerical.cuda_inventory.txt")

{
  printf 'start_utc=%s\n' "$start_utc"
  printf 'source_head=%s\n' "$source_head"
  printf 'expected_source_head=%s\n' "$expected_source_head"
  printf 'tensor_ops_sha256=%s\n' "$tensor_ops_sha"
  printf 'expected_tensor_ops_sha256=%s\n' "$expected_tensor_ops_sha"
  printf 'dql_cuda_sha256=%s\n' "$dql_cuda_sha"
  printf 'expected_dql_cuda_sha256=%s\n' "$expected_dql_cuda_sha"
  printf 'verifier_count=%s\n' "$verifier_count"
  printf 'expected_verifier_count=%s\n' "$expected_verifier_count"
  printf 'expected_plan_count=%s\n' "$expected_plan_count"
  printf 'iterations_per_plan=%s\n' "$iterations"
  printf 'expected_iteration_count=%s\n' "$expected_iteration_count"
} > "$report/full_numerical.meta.txt"

preflight_rc=0
[[ "$source_head" == "$expected_source_head" ]] || preflight_rc=1
[[ "$tensor_ops_sha" == "$expected_tensor_ops_sha" ]] || preflight_rc=1
[[ "$dql_cuda_sha" == "$expected_dql_cuda_sha" ]] || preflight_rc=1
[[ "$verifier_count" -eq "$expected_verifier_count" ]] || preflight_rc=1
printf '%s\n' "$preflight_rc" > "$report/full_numerical.preflight.rc.txt"

if [[ "$preflight_rc" -ne 0 ]]; then
  printf '%s\n' 'Full numerical preflight mismatch; Python was not started.' > "$report/full_numerical.stderr.txt"
  : > "$report/full_numerical.stdout.txt"
  printf '%s\n' 'NOT_RUN' > "$report/full_numerical.python.rc.txt"
  printf '%s\n' '1' > "$report/full_numerical.wrapper.rc.txt"
  printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/full_numerical.meta.txt"
  exit 1
fi

"${command[@]}" > "$report/full_numerical.stdout.txt" 2> "$report/full_numerical.stderr.txt"
python_rc=$?
printf '%s\n' "$python_rc" > "$report/full_numerical.python.rc.txt"

actual_plan_count=$(grep -c 'Testing ' "$report/full_numerical.stdout.txt" || true)
actual_pass_count=$(grep -c 'Pass (3/3)' "$report/full_numerical.stdout.txt" || true)
actual_iteration_count=$((actual_pass_count * iterations))
printf 'actual_plan_count=%s\n' "$actual_plan_count" >> "$report/full_numerical.meta.txt"
printf 'actual_pass_count=%s\n' "$actual_pass_count" >> "$report/full_numerical.meta.txt"
printf 'actual_passing_iteration_count=%s\n' "$actual_iteration_count" >> "$report/full_numerical.meta.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/full_numerical.meta.txt"

wrapper_rc="$python_rc"
if [[ "$python_rc" -eq 0 ]] && {
  [[ "$actual_plan_count" -ne "$expected_plan_count" ]] \
    || [[ "$actual_pass_count" -ne "$expected_plan_count" ]] \
    || [[ "$actual_iteration_count" -ne "$expected_iteration_count" ]];
}; then
  wrapper_rc=1
  printf '%s\n' 'Full numerical count mismatch despite Python RC 0.' >> "$report/full_numerical.stderr.txt"
fi

printf '%s\n' "$wrapper_rc" > "$report/full_numerical.wrapper.rc.txt"
exit "$wrapper_rc"
