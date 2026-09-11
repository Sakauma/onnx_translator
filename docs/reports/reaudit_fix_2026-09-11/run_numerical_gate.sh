#!/usr/bin/env bash
set -u
set -o pipefail

repo=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$repo/docs/reports/reaudit_fix_2026-09-11"
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export PYTHONIOENCODING=utf-8

cd "$repo" || exit 125

log_prefix=${1:-numerical_gate}
stdout="$report/$log_prefix.stdout.txt"
stderr="$report/$log_prefix.stderr.txt"
rc_file="$report/$log_prefix.rc.txt"
meta="$report/$log_prefix.meta.txt"
inventory="$report/$log_prefix.cuda_inventory.txt"

start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_sha=$(git rev-parse HEAD)
so_sha=$(sha256sum tensor_ops.so | awk '{print $1}')
mapfile -t verifier_paths < <(find cache -maxdepth 1 -type f -name 'verify_*' -perm -u+x -print | LC_ALL=C sort)

: >"$inventory"
for verifier in "${verifier_paths[@]}"; do
    sha256sum "$verifier" >>"$inventory"
done
verifier_count=${#verifier_paths[@]}

printf '%s\n' \
    "command=$python -u tools/cli.py numerical --iterations 3 --skip-plots" \
    "start_utc=$start_utc" \
    "source_sha=$source_sha" \
    "tensor_ops_sha256=$so_sha" \
    "compiled_verifier_count=$verifier_count" \
    "inventory_file=$(basename "$inventory")" >"$meta"

set +e
"$python" -u tools/cli.py numerical --iterations 3 --skip-plots >"$stdout" 2>"$stderr"
python_rc=$?
set -e

end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$python_rc" >"$rc_file"
printf '%s\n' "end_utc=$end_utc" "python_rc=$python_rc" >>"$meta"
exit "$python_rc"
