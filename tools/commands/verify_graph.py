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

import nn
import nn.ModelInitParas
from nn import Graph
from nn.GraphVisualization import GraphGenerate
from nn.ONNXImport import ONNXImport


# 实现 `run_verification` 步骤，规范化输入并返回下游期望的数据或元信息。
def run_verification(onnx_file_path, task_name, strict=True, allow_generic=False, clean=True):
    result_dir = os.path.join("./result", task_name)
    if clean and os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f"创建结果目录: {result_dir}")

    print(f"\n开始验证模型: {onnx_file_path}")

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
            model_name=task_name,
        )

        print("正在执行图结构推断 (forward_)...")
        placeholder_tensors = [nn.Tensor_(*t.size, dtype=t.dtype) for t in initial_tensors]
        graph.forward_(*placeholder_tensors)
        print("图结构推断完成，节点连接逻辑验证通过。")
    except Exception:
        print("错误: 图构建或形状推断失败。")
        traceback.print_exc()
        return 1

    print("\n[Step 4] 生成可视化流程图...")
    try:
        GraphGenerate(graph, task_name)
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
