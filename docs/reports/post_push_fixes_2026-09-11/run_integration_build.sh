#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
workspace=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$workspace/docs/reports/post_push_fixes_2026-09-11"
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
cd "$workspace"

printf '%s\n' 'make' > "$report/integration_make.command.txt"
printf '%s\n' "$python -u tools/cli.py compile-cuda" > "$report/integration_compile_cuda.command.txt"

{
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'workspace=%s\n' "$workspace"
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'path=%s\n' "$PATH"
  printf '\n[gcc]\n'
  gcc --version
  printf '\n[nvcc]\n'
  nvcc --version
  printf '\n[python-packages]\n'
  "$python" - <<'PY'
import platform
import numpy
import onnx
import torch
print(f"python={platform.python_version()}")
print(f"numpy={numpy.__version__}")
print(f"onnx={onnx.__version__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
  printf '\n[nvidia-smi]\n'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
} > "$report/integration_environment.txt" 2> "$report/integration_environment.stderr.txt"
environment_rc=$?
printf '%s\n' "$environment_rc" > "$report/integration_environment.rc.txt"

make > "$report/integration_make.stdout.txt" 2> "$report/integration_make.stderr.txt"
make_rc=$?
printf '%s\n' "$make_rc" > "$report/integration_make.rc.txt"

if [[ "$make_rc" -eq 0 ]]; then
  "$python" -u tools/cli.py compile-cuda \
    > "$report/integration_compile_cuda.stdout.txt" \
    2> "$report/integration_compile_cuda.stderr.txt"
  cuda_rc=$?
else
  : > "$report/integration_compile_cuda.stdout.txt"
  printf '%s\n' 'NOT_RUN: make failed' > "$report/integration_compile_cuda.stderr.txt"
  cuda_rc=125
fi
printf '%s\n' "$cuda_rc" > "$report/integration_compile_cuda.rc.txt"

if [[ "$make_rc" -eq 0 && "$cuda_rc" -eq 0 ]]; then
  sha256sum tensor_ops.so > "$report/integration_tensor_ops.sha256"
  find cache -maxdepth 1 -type f -name 'verify_*' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$report/integration_cuda_inventory.sha256"
  stat -c '%n %s bytes' tensor_ops.so > "$report/integration_artifacts.stat.txt"
  find cache -maxdepth 1 -type f -name 'verify_*' -printf '%f %s bytes\n' \
    | sort \
    >> "$report/integration_artifacts.stat.txt"
  printf 'verifier_count=%s\n' "$(wc -l < "$report/integration_cuda_inventory.sha256")" \
    > "$report/integration_inventory.meta.txt"
else
  : > "$report/integration_tensor_ops.sha256"
  : > "$report/integration_cuda_inventory.sha256"
  : > "$report/integration_artifacts.stat.txt"
  printf '%s\n' 'verifier_count=NOT_RECORDED' > "$report/integration_inventory.meta.txt"
fi

printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/integration_environment.txt"
if [[ "$make_rc" -ne 0 ]]; then
  exit "$make_rc"
fi
exit "$cuda_rc"
