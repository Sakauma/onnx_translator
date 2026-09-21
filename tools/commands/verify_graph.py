# /**
#   ******************************************************************************
#   * @file        verify_graph.py
#   * @author      Egor Izmaylov
#   * @brief       加载指定 ONNX 模型，验证导入、图构建、形状推断和前向执行流程。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import onnx

import nn
import nn.ModelInitParas
from nn import Graph
from nn.GraphVisualization import GraphGenerate
from nn.ONNXImport import ONNXImport


RESULT_ROOT = Path(__file__).resolve().parents[2] / "result"


def _resolve_result_dir(task_name):
    if (
        not task_name
        or task_name in {".", ".."}
        or "/" in task_name
        or "\\" in task_name
        or Path(task_name).is_absolute()
    ):
        raise ValueError("task name must be a non-empty directory basename")

    result_root = RESULT_ROOT.resolve()
    lexical_result_dir = result_root / task_name
    try:
        result_dir = lexical_result_dir.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"task result path cannot be resolved safely: {exc}") from exc
    if result_dir.parent != result_root:
        raise ValueError("task name must resolve to a direct child of the result directory")
    if result_dir != lexical_result_dir:
        raise ValueError("task result directory must not be a symbolic link or filesystem alias")
    return result_dir


def _declared_outputs(model):
    outputs = []
    for value_info in model.graph.output:
        tensor_type = value_info.type.tensor_type
        shape = None
        if tensor_type.HasField("shape"):
            dims = []
            for dim in tensor_type.shape.dim:
                dims.append(dim.dim_value if dim.HasField("dim_value") else None)
            shape = tuple(dims)
        outputs.append((value_info.name, tensor_type.elem_type, shape))
    return outputs


def _validate_declared_outputs(declarations, inferred):
    actual = list(inferred) if isinstance(inferred, (list, tuple)) else [inferred]
    if len(actual) != len(declarations):
        raise ValueError(
            f"Declared output count mismatch: expected {len(declarations)}, got {len(actual)}"
        )
    for index, ((name, elem_type, expected_shape), tensor) in enumerate(zip(declarations, actual)):
        actual_dtype = getattr(tensor, "dtype", None)
        expected_dtype = nn.onnx_dtype_mapping.get(elem_type)
        if actual_dtype != expected_dtype:
            raise TypeError(
                f"Output #{index} {name!r} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
            )
        if expected_shape is None:
            continue
        actual_shape = tuple(getattr(tensor, "size", ()))
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Output #{index} {name!r} rank mismatch: expected {len(expected_shape)}, "
                f"got {len(actual_shape)}"
            )
        for axis, (expected_dim, actual_dim) in enumerate(zip(expected_shape, actual_shape)):
            if expected_dim is not None and expected_dim != actual_dim:
                raise ValueError(
                    f"Output #{index} {name!r} dimension {axis} mismatch: "
                    f"expected {expected_dim}, got {actual_dim}"
                )


# 实现 `run_verification` 步骤，规范化输入并返回下游期望的数据或元信息。
def run_verification(onnx_file_path, task_name, strict=True, allow_generic=False, clean=True):
    result_dir = _resolve_result_dir(task_name)
    if clean and result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建结果目录: {result_dir}")

    print(f"\n开始验证模型: {onnx_file_path}")

    try:
        model = onnx.load_model(onnx_file_path, load_external_data=False)
        output_declarations = _declared_outputs(model)
        output_names = [item[0] for item in output_declarations]
    except Exception as e:
        print(f"错误: 无法读取模型输出声明: {e}")
        return 1

    print("\n[Step 1] 正在运行 ONNXImport 导入算子...")
    try:
        ops_list = ONNXImport(onnx_file_path, strict=strict)
    except Exception:
        print("导入严重失败! 无法继续。错误堆栈:")
        traceback.print_exc()
        return 1

    if not ops_list:
        print("错误: 未导入任何算子。")
        return 1

    op_types = {}
    generic_nodes = []
    for op in ops_list:
        name = op.__class__.__name__
        if name == "GenericNode":
            generic_nodes.append(op)
            name = f"Generic({op.op_type})"
        op_types[name] = op_types.get(name, 0) + 1

    print(f"成功导入 {len(ops_list)} 个算子节点。")
    print(f"算子统计: {op_types}")

    if generic_nodes and not allow_generic:
        print("错误: 存在 GenericNode，占位节点会掩盖真实导入问题。")
        for op in generic_nodes[:20]:
            print(f"  - {op.op_type} {op.name}: {op.error}")
        if len(generic_nodes) > 20:
            print(f"  ... 还有 {len(generic_nodes) - 20} 个")
        return 1

    print("\n[Step 2] 解析模型初始输入参数...")
    try:
        initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(onnx_file_path)
        print(f"模型输入名称: {initial_inputs}")
    except Exception as e:
        print(f"错误: 解析输入参数失败: {e}")
        traceback.print_exc()
        return 1

    print("\n[Step 3] 构建计算图并尝试形状推断...")
    try:
        graph = Graph(
            ops=ops_list,
            input_name=initial_inputs,
            output_name=output_names,
            model_name=task_name,
        )

        print("正在执行图结构推断 (forward_)...")
        placeholder_tensors = [nn.Tensor_(*t.size, dtype=t.dtype) for t in initial_tensors]
        inferred_outputs = graph.forward_(*placeholder_tensors)
        _validate_declared_outputs(output_declarations, inferred_outputs)
        print("图结构推断完成，节点连接逻辑验证通过。")
    except Exception:
        print("错误: 图构建或形状推断失败。")
        traceback.print_exc()
        return 1

    print("\n[Step 4] 生成可视化流程图...")
    try:
        GraphGenerate(graph, task_name, output_dir=result_dir, raise_on_error=True)
    except Exception:
        print("错误: 生成可视化图表失败。")
        traceback.print_exc()
        return 1

    return 0


# 作为 `tools/cli.py verify-graph` 子命令实现，解析参数、调度检查流程并返回进程退出码。
def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify ONNX import, graph construction, and visualization.")
    parser.add_argument("--model", default="./onnx_model/model.onnx", help="ONNX model path.")
    parser.add_argument("--task-name", default="nps_verification", help="Result subdirectory name.")
    parser.add_argument("--allow-generic", action="store_true", help="Allow GenericNode fallback nodes.")
    parser.add_argument("--no-strict", action="store_true", help="Let ONNXImport downgrade unsupported nodes instead of failing immediately.")
    parser.add_argument("--no-clean", action="store_true", help="Keep the previous result directory.")
    args = parser.parse_args(argv)

    try:
        _resolve_result_dir(args.task_name)
    except ValueError as exc:
        parser.error(str(exc))

    if not os.path.exists(args.model):
        print(f"找不到模型文件: {args.model}")
        return 2

    return run_verification(
        onnx_file_path=args.model,
        task_name=args.task_name,
        strict=not args.no_strict,
        allow_generic=args.allow_generic,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    sys.exit(main())
