#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
workspace=/mnt/d/workspace/onnx_translator
report=/mnt/d/workspace/onnx_translator_bugfix_worktree/docs/reports/post_push_fixes_2026-09-11/evidence/main
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
expected_head=e0a03c6b3e71ba5c9562624f783c3ac6e4165322
mkdir -p "$report"
cd "$workspace"

actual_head=$(git rev-parse HEAD)
printf 'start_utc=%s\nhead=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$actual_head" > "$report/main_sync.meta.txt"
if [[ "$actual_head" != "$expected_head" ]]; then
  printf '%s\n' 'Main checkout source identity mismatch; no gate was started.' > "$report/main_sync.stderr.txt"
  printf '%s\n' '125' > "$report/main_sync.rc.txt"
  exit 125
fi

printf '%s\n' 'make' > "$report/main_make.command.txt"
make > "$report/main_make.stdout.txt" 2> "$report/main_make.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_make.rc.txt"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi

printf '%q ' "$python" -u tools/cli.py compile-cuda --op layer_normalization --force > "$report/main_layernorm_compile.command.txt"
printf '\n' >> "$report/main_layernorm_compile.command.txt"
"$python" -u tools/cli.py compile-cuda --op layer_normalization --force > "$report/main_layernorm_compile.stdout.txt" 2> "$report/main_layernorm_compile.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_layernorm_compile.rc.txt"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi

{
  printf 'tensor_ops_sha256=%s\n' "$(sha256sum tensor_ops.so | cut -d' ' -f1)"
  printf 'layernorm_verifier_sha256=%s\n' "$(sha256sum cache/verify_layer_normalization | cut -d' ' -f1)"
  printf 'verifier_count=%s\n' "$(find cache -maxdepth 1 -type f -name 'verify_*' | wc -l)"
} >> "$report/main_sync.meta.txt"

printf '%q ' "$python" -u -m pytest -q tests/test_post_push_shape_semantics.py tests/test_post_push_layernorm_semantics.py > "$report/main_targeted_pytest.command.txt"
printf '\n' >> "$report/main_targeted_pytest.command.txt"
"$python" -u -m pytest -q tests/test_post_push_shape_semantics.py tests/test_post_push_layernorm_semantics.py > "$report/main_targeted_pytest.stdout.txt" 2> "$report/main_targeted_pytest.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_targeted_pytest.rc.txt"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi

printf '%q ' "$python" -u tools/cli.py numerical --op layer_normalization --iterations 3 --skip-plots > "$report/main_layernorm_numerical.command.txt"
printf '\n' >> "$report/main_layernorm_numerical.command.txt"
"$python" -u tools/cli.py numerical --op layer_normalization --iterations 3 --skip-plots > "$report/main_layernorm_numerical.stdout.txt" 2> "$report/main_layernorm_numerical.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_layernorm_numerical.rc.txt"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi

printf '%q ' "$python" -u tools/cli.py numerical --op resize --iterations 3 --skip-plots > "$report/main_resize_numerical.command.txt"
printf '\n' >> "$report/main_resize_numerical.command.txt"
"$python" -u tools/cli.py numerical --op resize --iterations 3 --skip-plots > "$report/main_resize_numerical.stdout.txt" 2> "$report/main_resize_numerical.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/main_resize_numerical.rc.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/main_sync.meta.txt"
printf '%s\n' "$rc" > "$report/main_sync.rc.txt"
exit "$rc"
