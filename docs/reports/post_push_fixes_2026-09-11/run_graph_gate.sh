#!/usr/bin/env bash
set -u

export PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
export LC_ALL=C.UTF-8
workspace=/mnt/d/workspace/onnx_translator_bugfix_worktree
report="$workspace/docs/reports/post_push_fixes_2026-09-11"
python=/home/sakauma/data/miniconda3/envs/egor/bin/python
cd "$workspace"

head_short=$(git rev-parse --short=12 HEAD)
model="/tmp/onnx_translator_resize_graph_gate_${head_short}.onnx"
task_name="post_push_fixes_graph_gate_${head_short}"
result_path="$workspace/result/$task_name"
if [[ -e "$result_path" ]]; then
  printf 'Refusing to overwrite existing result path: %s\n' "$result_path" \
    > "$report/graph_gate.stderr.txt"
  printf '%s\n' '125' > "$report/graph_gate.rc.txt"
  exit 125
fi

"$python" - "$model" <<'PY'
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

path = sys.argv[1]
graph = helper.make_graph(
    [helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], mode="nearest")],
    "resize_default_graph_gate",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 3])],
    initializer=[numpy_helper.from_array(np.array([1, 1, 1, 3], dtype=np.int64), "sizes")],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, path)
PY

command=("$python" -u tools/cli.py verify-graph --model "$model" --task-name "$task_name" --no-clean)
printf '%q ' "${command[@]}" > "$report/graph_gate.command.txt"
printf '\n' >> "$report/graph_gate.command.txt"
{
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'model=%s\n' "$model"
  printf 'task_name=%s\n' "$task_name"
  printf 'result_path=%s\n' "$result_path"
  printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$report/graph_gate.meta.txt"
"${command[@]}" > "$report/graph_gate.stdout.txt" 2> "$report/graph_gate.stderr.txt"
rc=$?
printf '%s\n' "$rc" > "$report/graph_gate.rc.txt"
printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$report/graph_gate.meta.txt"
exit "$rc"
