"""文件功能：验证指定 ONNX 模型的导入、图结构推断和可视化生成流程。
作者：Egor Izmaylov
时间：2026-06-02
"""

import nn
from nn import Graph
import nn.ModelInitParas
from nn.ONNXImport import ONNXImport
from nn.GraphVisualization import GraphGenerate
import os
import sys


# 执行图逻辑验证命令，导入模型、推断结构并生成可视化结果。
def main(model_path="./onnx_model/model.onnx", model_name="graph_logic_test"):
    result_dir = os.path.join("./result", model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建目录: {result_dir}")

    print("步骤 1: 导入 ONNX 模型并映射算子...")
    ops_list = ONNXImport(model_path)
    if not ops_list:
        print("错误：没有从 ONNX 文件中导入任何算子。")
        return 1
    print(f"成功导入 {len(ops_list)} 个算子。")

    print("\n步骤 2: 解析模型初始输入参数...")
    try:
        initial_inputs, initial_tensors = nn.ModelInitParas.ONNXParasGen(model_path)
        print(f"模型输入: {initial_inputs}")
        print(f"生成模拟输入张量 (形状): {[t.size for t in initial_tensors]}")
    except Exception as e:
        print(f"解析输入参数时出错: {e}")
        return 1

    print("\n步骤 3: 创建 Graph 对象 (用于图结构推断)...")
    graph_for_logic = Graph(
        ops=ops_list,
        input_name=initial_inputs,
        model_name=model_name,
    )

    placeholder_tensors = [nn.Tensor_(*t.size, dtype=t.dtype) for t in initial_tensors]
    graph_for_logic.forward_(*placeholder_tensors)
    print("图结构推断 (forward_) 完成。")

    print("\n步骤 4: 生成图可视化文件...")
    try:
        GraphGenerate(graph_for_logic, model_name)
        print(f"✅ 逻辑验证成功！流程图已保存在 '{result_dir}' 目录中。")
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
